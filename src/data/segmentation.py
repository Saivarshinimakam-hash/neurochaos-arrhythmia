"""
src/data/segmentation.py
-------------------------
Beat segmentation: extract fixed-length windows around annotated R-peaks.

Window definition (from config.yaml)
--------------------------------------
  beat_before_samples : 108   (0.3 s × 360 Hz)
  beat_after_samples  : 180   (0.5 s × 360 Hz)
  beat_window_size    : 288   total samples

The asymmetric window is intentional: more context is kept after the
R-peak to capture the ST segment and T-wave, which carry discriminative
information for supraventricular vs ventricular beats.

Boundary handling
-----------------
Beats whose window would extend past the signal boundary are DISCARDED
by default (``allow_padding=False``).  This is the safe default for
research: we never invent signal.  If ``allow_padding=True`` is passed,
edge-padding with the signal's edge value (``np.pad`` mode ``"edge"``)
is used and a warning is emitted.

The discarded-beat count is always reported so the caller knows the
yield loss.

Data containers
---------------
``SegmentedBeat``  — a single extracted beat with its metadata.
``BeatDataset``    — a collection of beats from one or more records,
                     with aligned NumPy arrays ready for model input.

No normalisation is performed here.  Normalisation is applied in
src/data/splitting.py after the patient-wise split to avoid leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.data.loader import ECGRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentedBeat:
    """A single segmented beat window with its provenance metadata.

    Attributes
    ----------
    waveform : np.ndarray, shape (window_size,)
        Raw (filtered but not normalised) ECG samples.
    label : int
        AAMI integer class index (0–4).
    aami_class : str
        AAMI class string (``"N"``, ``"S"``, …).
    symbol : str
        Original wfdb beat annotation symbol.
    record_id : str
        Source record identifier (e.g. ``"100"``).
    r_peak_sample : int
        Sample index of the R-peak in the original full-length signal.
    beat_index : int
        Position of this beat in the record's annotation list.
    """
    waveform: np.ndarray
    label: int
    aami_class: str
    symbol: str
    record_id: str
    r_peak_sample: int
    beat_index: int

    def __repr__(self) -> str:
        return (
            f"SegmentedBeat(record={self.record_id!r}, "
            f"beat_idx={self.beat_index}, r_peak={self.r_peak_sample}, "
            f"class={self.aami_class!r}({self.label}), "
            f"waveform_shape={self.waveform.shape})"
        )


@dataclass
class BeatDataset:
    """Collection of segmented beats from one or more ECG records.

    Attributes
    ----------
    waveforms : np.ndarray, shape (n_beats, window_size)
        All beat waveforms stacked row-wise.
    labels : np.ndarray, shape (n_beats,), dtype int32
        AAMI integer class index for each beat.
    aami_classes : list[str]
        AAMI class string for each beat.
    symbols : list[str]
        Original wfdb symbol for each beat.
    record_ids : list[str]
        Source record ID for each beat.
    r_peak_samples : np.ndarray, shape (n_beats,), dtype int64
        R-peak sample indices (within the original record signal).
    beat_indices : np.ndarray, shape (n_beats,), dtype int64
        Beat position within each record's annotation list.
    window_size : int
        Number of samples per beat window.
    n_beats : int
        Total number of beats.
    source_records : list[str]
        Unique record IDs contributing beats (sorted).
    class_distribution : dict[str, int]
        Beat counts per AAMI class.

    Notes
    -----
    All list/array attributes are always the same length (``n_beats``).
    """
    waveforms: np.ndarray
    labels: np.ndarray
    aami_classes: list[str]
    symbols: list[str]
    record_ids: list[str]
    r_peak_samples: np.ndarray
    beat_indices: np.ndarray
    window_size: int
    n_beats: int = field(init=False)
    source_records: list[str] = field(init=False)
    class_distribution: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.n_beats = len(self.labels)
        self.source_records = sorted(set(self.record_ids))

        from src.data.annotations import AAMI_CLASS_TO_INDEX
        self.class_distribution = {
            cls: int(np.sum(self.labels == idx))
            for cls, idx in AAMI_CLASS_TO_INDEX.items()
        }

        # Integrity check
        lengths = {
            "waveforms": len(self.waveforms),
            "labels": len(self.labels),
            "aami_classes": len(self.aami_classes),
            "symbols": len(self.symbols),
            "record_ids": len(self.record_ids),
            "r_peak_samples": len(self.r_peak_samples),
            "beat_indices": len(self.beat_indices),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"BeatDataset field length mismatch: {lengths}"
            )

    def __repr__(self) -> str:
        dist = ", ".join(
            f"{k}={v}" for k, v in self.class_distribution.items()
        )
        return (
            f"BeatDataset(n_beats={self.n_beats}, "
            f"window_size={self.window_size}, "
            f"records={len(self.source_records)}, "
            f"dist=[{dist}])"
        )

    def __len__(self) -> int:
        return self.n_beats

    def is_empty(self) -> bool:
        return self.n_beats == 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_record(
    record: ECGRecord,
    before: int = 108,
    after: int = 180,
    allow_padding: bool = False,
) -> BeatDataset:
    """Segment all annotated beats from a single ECG record.

    The record signal is expected to be already preprocessed (cleaned
    and band-pass filtered) by ``src.data.preprocessing.preprocess_signal``.

    Parameters
    ----------
    record:
        A loaded and validated ``ECGRecord``.
    before:
        Number of samples to include BEFORE the R-peak.
    after:
        Number of samples to include AFTER the R-peak.
    allow_padding:
        If ``True``, beats near signal boundaries are edge-padded and a
        warning is emitted.  If ``False`` (default), boundary beats are
        discarded.

    Returns
    -------
    BeatDataset
        All successfully extracted beats (may be empty if all beats hit
        boundaries and ``allow_padding=False``).
    """
    window_size = before + after
    signal = record.signal
    n_signal = len(signal)

    waveforms:     list[np.ndarray] = []
    labels:        list[int]        = []
    aami_classes:  list[str]        = []
    symbols:       list[str]        = []
    record_ids:    list[str]        = []
    r_peak_samps:  list[int]        = []
    beat_idxs:     list[int]        = []

    n_boundary = 0
    n_padded   = 0

    for beat_idx, (r_peak, sym, aami, lbl) in enumerate(zip(
        record.beat_samples,
        record.beat_symbols,
        record.aami_classes,
        record.aami_labels,
    )):
        start = int(r_peak) - before
        end   = int(r_peak) + after   # exclusive

        # ── Boundary check ───────────────────────────────────────────
        needs_padding = (start < 0) or (end > n_signal)

        if needs_padding:
            if not allow_padding:
                n_boundary += 1
                logger.debug(
                    "Record %s beat %d (r=%d): window [%d, %d) out of "
                    "bounds [0, %d) — discarded",
                    record.record_id, beat_idx, r_peak,
                    start, end, n_signal,
                )
                continue
            else:
                # Edge-pad the signal slice
                pad_left  = max(0, -start)
                pad_right = max(0, end - n_signal)
                clipped_start = max(0, start)
                clipped_end   = min(n_signal, end)
                core = signal[clipped_start:clipped_end]
                waveform = np.pad(core, (pad_left, pad_right), mode="edge")
                n_padded += 1
                logger.debug(
                    "Record %s beat %d: edge-padded (%d left, %d right)",
                    record.record_id, beat_idx, pad_left, pad_right,
                )
        else:
            waveform = signal[start:end].copy()

        # Sanity — should always hold
        if len(waveform) != window_size:
            logger.error(
                "Record %s beat %d: extracted waveform length %d != "
                "expected %d — skipping",
                record.record_id, beat_idx, len(waveform), window_size,
            )
            continue

        waveforms.append(waveform.astype(np.float32))
        labels.append(int(lbl))
        aami_classes.append(aami)
        symbols.append(sym)
        record_ids.append(record.record_id)
        r_peak_samps.append(int(r_peak))
        beat_idxs.append(beat_idx)

    n_kept = len(labels)
    n_total = record.n_beats

    if n_boundary:
        logger.info(
            "Record %s: discarded %d / %d boundary beats (window [-%d, +%d))",
            record.record_id, n_boundary, n_total, before, after,
        )
    if n_padded:
        logger.warning(
            "Record %s: edge-padded %d / %d boundary beats",
            record.record_id, n_padded, n_total,
        )

    logger.info(
        "Record %s: segmented %d / %d beats (window_size=%d)",
        record.record_id, n_kept, n_total, window_size,
    )

    if n_kept == 0:
        logger.warning(
            "Record %s: NO beats were extracted. "
            "Check beat annotations and signal length.",
            record.record_id,
        )
        # Return empty dataset with correct window_size
        return _empty_dataset(window_size)

    return BeatDataset(
        waveforms=np.stack(waveforms, axis=0),          # (n, window_size)
        labels=np.array(labels, dtype=np.int32),
        aami_classes=aami_classes,
        symbols=symbols,
        record_ids=record_ids,
        r_peak_samples=np.array(r_peak_samps, dtype=np.int64),
        beat_indices=np.array(beat_idxs, dtype=np.int64),
        window_size=window_size,
    )


def segment_records(
    records: list[ECGRecord],
    before: int = 108,
    after: int = 180,
    allow_padding: bool = False,
) -> BeatDataset:
    """Segment beats from multiple records and concatenate into one dataset.

    Parameters
    ----------
    records:
        List of loaded ``ECGRecord`` objects (already preprocessed).
    before, after:
        Window parameters (same semantics as ``segment_record``).
    allow_padding:
        Propagated to ``segment_record``.

    Returns
    -------
    BeatDataset
        All beats from all records concatenated.

    Raises
    ------
    ValueError
        If *records* is empty.
    """
    if not records:
        raise ValueError("segment_records: records list is empty")

    datasets = [
        segment_record(rec, before=before, after=after,
                       allow_padding=allow_padding)
        for rec in records
    ]

    # Filter out empty datasets before concatenation
    non_empty = [ds for ds in datasets if not ds.is_empty()]
    if not non_empty:
        logger.warning("All records produced empty datasets after segmentation")
        window_size = before + after
        return _empty_dataset(window_size)

    combined = _concatenate_datasets(non_empty)
    logger.info(
        "Combined %d record(s) → %s",
        len(non_empty), combined,
    )
    return combined


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_dataset(window_size: int) -> BeatDataset:
    """Return a BeatDataset with zero beats and the correct window size."""
    from src.data.annotations import AAMI_CLASS_TO_INDEX
    return BeatDataset(
        waveforms=np.empty((0, window_size), dtype=np.float32),
        labels=np.empty(0, dtype=np.int32),
        aami_classes=[],
        symbols=[],
        record_ids=[],
        r_peak_samples=np.empty(0, dtype=np.int64),
        beat_indices=np.empty(0, dtype=np.int64),
        window_size=window_size,
    )


def _concatenate_datasets(datasets: list[BeatDataset]) -> BeatDataset:
    """Concatenate a list of non-empty BeatDatasets."""
    window_sizes = {ds.window_size for ds in datasets}
    if len(window_sizes) > 1:
        raise ValueError(
            f"Cannot concatenate datasets with different window sizes: "
            f"{window_sizes}"
        )

    return BeatDataset(
        waveforms=np.concatenate([ds.waveforms for ds in datasets], axis=0),
        labels=np.concatenate([ds.labels for ds in datasets], axis=0),
        aami_classes=sum((ds.aami_classes for ds in datasets), []),
        symbols=sum((ds.symbols for ds in datasets), []),
        record_ids=sum((ds.record_ids for ds in datasets), []),
        r_peak_samples=np.concatenate(
            [ds.r_peak_samples for ds in datasets], axis=0
        ),
        beat_indices=np.concatenate(
            [ds.beat_indices for ds in datasets], axis=0
        ),
        window_size=datasets[0].window_size,
    )
