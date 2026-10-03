"""
src/data/loader.py
------------------
MIT-BIH Arrhythmia Database loader.

Responsibilities
----------------
1. Download records from PhysioNet via ``wfdb`` if not already present.
2. Load raw ECG signal and annotation for a single record.
3. Validate signal integrity (length, NaN/Inf, flat-line detection).
4. Provide a high-level function to load ALL records in a split.

What this module does NOT do
-----------------------------
- Filtering / preprocessing  → src/data/preprocessing.py
- Beat segmentation          → src/data/segmentation.py
- Train/val/test splitting   → src/data/splitting.py

Design notes
------------
- ``wfdb.rdrecord`` loads the physical signal (mV) directly.
- ``wfdb.rdann`` loads the annotation file (.atr).
- We load **one lead** (configurable; default MLII = channel 0).
- Signal validation raises ``SignalValidationError`` (a subclass of
  ``ValueError``) so callers can decide whether to skip a record or
  abort.
- Download is idempotent: if the ``.dat`` file already exists the
  download step is skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import wfdb

from src.data.annotations import (
    AAMI_SYMBOL_TO_CLASS,
    AAMI_CLASS_TO_INDEX,
    NON_BEAT_SYMBOLS,
    symbol_to_aami,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class SignalValidationError(ValueError):
    """Raised when a loaded ECG record fails quality checks."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ECGRecord:
    """Container for a single loaded MIT-BIH record.

    Attributes
    ----------
    record_id : str
        Record identifier (e.g. ``"100"``).
    signal : np.ndarray, shape (n_samples,)
        Raw ECG signal in physical units (mV) for the selected lead.
    fs : int
        Sampling frequency in Hz (360 for MIT-BIH).
    beat_samples : np.ndarray, shape (n_beats,)
        Sample indices of each annotated beat (R-peak positions).
    beat_symbols : list[str]
        Original wfdb beat annotation symbol for each beat.
    aami_labels : np.ndarray, shape (n_beats,)
        Integer AAMI class index for each beat.
    aami_classes : list[str]
        AAMI class string for each beat (``"N"``, ``"S"``, …).
    n_samples : int
        Total number of signal samples.
    n_beats : int
        Total number of annotated beats (after filtering non-beat symbols).
    lead_name : str
        Name of the loaded lead channel (e.g. ``"MLII"``).
    """

    record_id: str
    signal: np.ndarray
    fs: int
    beat_samples: np.ndarray
    beat_symbols: list[str]
    aami_labels: np.ndarray
    aami_classes: list[str]
    n_samples: int
    n_beats: int
    lead_name: str
    class_distribution: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.class_distribution = {
            cls: int(np.sum(self.aami_labels == idx))
            for cls, idx in AAMI_CLASS_TO_INDEX.items()
        }

    def __repr__(self) -> str:
        dist = ", ".join(f"{k}={v}" for k, v in self.class_distribution.items())
        return (
            f"ECGRecord(id={self.record_id!r}, lead={self.lead_name!r}, "
            f"samples={self.n_samples}, beats={self.n_beats}, "
            f"dist=[{dist}])"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_record(
    record_id: int | str,
    data_dir: str | Path,
    fs: int = 360,
    lead: int = 0,
    download: bool = True,
) -> ECGRecord:
    """Load a single MIT-BIH record.

    Parameters
    ----------
    record_id:
        Record number, e.g. ``100`` or ``"100"``.
    data_dir:
        Directory where the MIT-BIH ``.dat``, ``.hea``, ``.atr`` files
        are stored (or will be downloaded to).
    fs:
        Expected sampling frequency.  Used for validation only; the
        actual fs is read from the record header.
    lead:
        Channel index to load (0 = MLII, 1 = V1/V5 depending on record).
    download:
        If ``True`` and the record files are not present, download them
        from PhysioNet.

    Returns
    -------
    ECGRecord
        Validated, annotated record container.

    Raises
    ------
    FileNotFoundError
        If record files are missing and ``download=False``.
    SignalValidationError
        If the signal fails quality checks.
    """
    record_id = str(record_id)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    record_path = data_dir / record_id

    # ── Download if needed ───────────────────────────────────────────────
    if not _record_exists(record_path):
        if not download:
            raise FileNotFoundError(
                f"Record {record_id} not found at {data_dir} "
                "and download=False."
            )
        logger.info("Downloading record %s from PhysioNet ...", record_id)
        _download_record(record_id, data_dir)

    # ── Load signal ──────────────────────────────────────────────────────
    logger.debug("Loading signal for record %s (lead index %d)", record_id, lead)
    record = wfdb.rdrecord(str(record_path), channels=[lead])

    actual_fs = record.fs
    if actual_fs != fs:
        logger.warning(
            "Record %s fs=%d does not match expected fs=%d",
            record_id, actual_fs, fs,
        )

    # p_signal: shape (n_samples, n_channels); squeeze to 1D
    signal_2d: np.ndarray = record.p_signal  # physical units (mV)
    signal: np.ndarray = signal_2d[:, 0].astype(np.float32)

    lead_name: str = record.sig_name[0] if record.sig_name else f"ch{lead}"

    # ── Load annotations ─────────────────────────────────────────────────
    logger.debug("Loading annotations for record %s", record_id)
    ann = wfdb.rdann(str(record_path), extension="atr")

    # Filter to beat-only annotations
    beat_samples_all = np.array(ann.sample)
    beat_symbols_all = ann.symbol

    beat_samples, beat_symbols, aami_classes, aami_labels = _filter_beats(
        beat_samples_all, beat_symbols_all, len(signal), record_id
    )

    # ── Validate signal ──────────────────────────────────────────────────
    _validate_signal(signal, record_id, actual_fs)

    ecg = ECGRecord(
        record_id=record_id,
        signal=signal,
        fs=actual_fs,
        beat_samples=beat_samples,
        beat_symbols=beat_symbols,
        aami_labels=aami_labels,
        aami_classes=aami_classes,
        n_samples=len(signal),
        n_beats=len(beat_samples),
        lead_name=lead_name,
    )

    logger.info("Loaded %s", ecg)
    return ecg


def load_records(
    record_ids: list[int | str],
    data_dir: str | Path,
    fs: int = 360,
    lead: int = 0,
    download: bool = True,
    skip_on_error: bool = False,
) -> list[ECGRecord]:
    """Load multiple MIT-BIH records.

    Parameters
    ----------
    record_ids:
        List of record identifiers to load.
    data_dir:
        Data directory (same semantics as ``load_record``).
    fs:
        Expected sampling frequency.
    lead:
        Channel index to load.
    download:
        Download missing records from PhysioNet.
    skip_on_error:
        If ``True``, log failures and continue instead of raising.

    Returns
    -------
    list[ECGRecord]
        Successfully loaded records in the same order as *record_ids*.
    """
    records: list[ECGRecord] = []

    for rid in record_ids:
        try:
            rec = load_record(
                record_id=rid,
                data_dir=data_dir,
                fs=fs,
                lead=lead,
                download=download,
            )
            records.append(rec)
        except (SignalValidationError, FileNotFoundError, Exception) as exc:
            if skip_on_error:
                logger.error(
                    "Skipping record %s due to error: %s: %s",
                    rid, type(exc).__name__, exc,
                )
            else:
                raise

    logger.info(
        "Loaded %d / %d requested records", len(records), len(record_ids)
    )
    return records


def get_dataset_summary(records: list[ECGRecord]) -> dict[str, Any]:
    """Compute a summary statistics dictionary for a collection of records.

    Parameters
    ----------
    records:
        List of loaded ECGRecord objects.

    Returns
    -------
    dict
        Keys: ``n_records``, ``total_beats``, ``class_counts``,
        ``class_percentages``, ``records_per_class``.
    """
    total_beats = sum(r.n_beats for r in records)
    class_counts: dict[str, int] = {c: 0 for c in AAMI_CLASS_TO_INDEX}

    for rec in records:
        for cls, cnt in rec.class_distribution.items():
            class_counts[cls] += cnt

    class_pct = {
        cls: round(100.0 * cnt / total_beats, 2) if total_beats > 0 else 0.0
        for cls, cnt in class_counts.items()
    }

    return {
        "n_records": len(records),
        "total_beats": total_beats,
        "class_counts": class_counts,
        "class_percentages": class_pct,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _record_exists(record_path: Path) -> bool:
    """Return True if both the .dat and .hea files exist."""
    return record_path.with_suffix(".dat").exists() and \
           record_path.with_suffix(".hea").exists()


def _download_record(record_id: str, dest_dir: Path) -> None:
    """Download a single MIT-BIH record using wfdb.

    wfdb.dl_database downloads all files for the given record ID into
    dest_dir.  The PhysioNet database path is 'mitdb'.
    """
    try:
        wfdb.dl_database(
            db_dir="mitdb",
            dl_dir=str(dest_dir),
            records=[record_id],
            annotators=["atr"],
        )
        logger.info("Downloaded record %s to %s", record_id, dest_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download record {record_id}: {exc}\n"
            "Check your internet connection or download the MIT-BIH "
            "database manually from https://physionet.org/content/mitdb/1.0.0/"
        ) from exc


def _filter_beats(
    samples: np.ndarray,
    symbols: list[str],
    signal_length: int,
    record_id: str,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    """Filter annotation arrays to include only mappable beat annotations.

    Returns
    -------
    beat_samples : np.ndarray
    beat_symbols : list[str]
    aami_classes : list[str]
    aami_labels  : np.ndarray (int32)

    - Non-beat annotation symbols (rhythm markers, etc.) are dropped.
    - Beat symbols not in the AAMI mapping are logged as warnings and dropped.
    - Beats outside the signal boundary are dropped.
    """
    kept_samples: list[int] = []
    kept_symbols: list[str] = []
    kept_classes: list[str] = []
    kept_labels: list[int] = []

    skipped_non_beat = 0
    skipped_unknown = 0
    skipped_boundary = 0

    for sample, sym in zip(samples, symbols):
        # Skip non-beat annotations (rhythm labels, markers, etc.)
        if sym in NON_BEAT_SYMBOLS:
            skipped_non_beat += 1
            continue

        # Skip beats outside the signal boundary
        if sample < 0 or sample >= signal_length:
            skipped_boundary += 1
            logger.warning(
                "Record %s: beat at sample %d is outside signal boundary [0, %d)",
                record_id, sample, signal_length,
            )
            continue

        # Map to AAMI class
        aami = symbol_to_aami(sym)
        if aami is None:
            skipped_unknown += 1
            logger.debug(
                "Record %s: unmapped symbol '%s' at sample %d — excluded",
                record_id, sym, sample,
            )
            continue

        kept_samples.append(int(sample))
        kept_symbols.append(sym)
        kept_classes.append(aami)
        kept_labels.append(AAMI_CLASS_TO_INDEX[aami])

    if skipped_non_beat:
        logger.debug(
            "Record %s: skipped %d non-beat annotations",
            record_id, skipped_non_beat,
        )
    if skipped_unknown:
        logger.warning(
            "Record %s: skipped %d beats with unmapped symbols",
            record_id, skipped_unknown,
        )
    if skipped_boundary:
        logger.warning(
            "Record %s: skipped %d out-of-boundary beats",
            record_id, skipped_boundary,
        )

    return (
        np.array(kept_samples, dtype=np.int64),
        kept_symbols,
        kept_classes,
        np.array(kept_labels, dtype=np.int32),
    )


def _validate_signal(signal: np.ndarray, record_id: str, fs: int) -> None:
    """Run quality checks on the raw signal.

    Checks
    ------
    1. Signal is non-empty.
    2. No NaN values.
    3. No Inf values.
    4. Signal is not completely flat (std > threshold).
    5. Minimum duration check (at least 10 seconds).

    Raises
    ------
    SignalValidationError
        On any failed check.
    """
    if signal.size == 0:
        raise SignalValidationError(
            f"Record {record_id}: signal is empty."
        )

    if np.any(np.isnan(signal)):
        n_nan = int(np.sum(np.isnan(signal)))
        raise SignalValidationError(
            f"Record {record_id}: signal contains {n_nan} NaN values."
        )

    if np.any(np.isinf(signal)):
        n_inf = int(np.sum(np.isinf(signal)))
        raise SignalValidationError(
            f"Record {record_id}: signal contains {n_inf} Inf values."
        )

    std = float(np.std(signal))
    FLAT_LINE_THRESHOLD = 1e-6  # mV
    if std < FLAT_LINE_THRESHOLD:
        raise SignalValidationError(
            f"Record {record_id}: signal appears flat (std={std:.2e} < "
            f"{FLAT_LINE_THRESHOLD:.2e} mV). Possible lead-off or bad record."
        )

    min_duration_sec = 10
    min_samples = fs * min_duration_sec
    if len(signal) < min_samples:
        raise SignalValidationError(
            f"Record {record_id}: signal too short ({len(signal)} samples, "
            f"need at least {min_samples} for {min_duration_sec}s at {fs} Hz)."
        )

    logger.debug(
        "Record %s signal validated: %d samples, std=%.4f mV",
        record_id, len(signal), std,
    )
