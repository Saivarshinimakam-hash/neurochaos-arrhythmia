"""
src/data/splitting.py
----------------------
Patient-wise train / validation / test dataset assembly.

Core guarantee
--------------
Every beat in the dataset is assigned to exactly one split based solely
on which RECORD it came from.  Beats from the same record NEVER appear
in more than one split.  This is enforced by:

  1. Reading the canonical record lists from config.yaml (fixed at
     project setup time — not computed at runtime from the data).
  2. Verifying zero overlap between the three record sets after loading.
  3. Refusing to proceed if any loaded record appears in the wrong split.

Why not random beat-level splitting?
-------------------------------------
MIT-BIH recordings contain ~30-minute continuous signals from individual
patients.  Adjacent beats share morphology, noise, and recorder
characteristics.  A random beat-level 80/20 split leaks: the model sees
nearly the same patient waveform shapes in both train and test,
inflating reported performance.  Patient-wise splitting is the standard
required by ANSI/AAMI EC57 for inter-subject evaluation.

Normalisation strategy (no leakage)
------------------------------------
Per-beat z-score normalisation is self-contained (mean/std computed
from the beat window itself), so it is safe to apply identically to
all splits without fitting on train data.  Dataset-wide statistics are
never computed.

Split manifest
--------------
After assembly, a JSON manifest is written to
``results/experiments/split_manifest_<timestamp>.json`` containing:

  - record assignments (train/val/test)
  - beat counts per split and per class
  - class distribution percentage
  - window parameters
  - config seed

This makes every experiment exactly reproducible.

Limitations
-----------
- MIT-BIH has only 48 records.  Perfect stratification across 5 classes
  at the record level is mathematically impossible: the F and Q classes
  are extremely rare and not present in every record.  The split in
  config.yaml was hand-assigned to balance N/V coverage while keeping
  the test set fixed.  This limitation is documented here and in the
  manifest.
- The val and test sets each contain ~6–9 records; the class distribution
  of these small sets may differ from the training distribution.
  Weighted metrics (weighted F1, balanced accuracy) should be used in
  evaluation to account for this.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.data.loader import ECGRecord, load_records
from src.data.preprocessing import preprocess_signal, normalize_beats_zscore
from src.data.segmentation import BeatDataset, segment_record, _concatenate_datasets
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_datasets(
    cfg: dict[str, Any],
    download: bool = True,
    normalize: bool = True,
    allow_padding: bool = False,
    save_manifest: bool = True,
) -> tuple[BeatDataset, BeatDataset, BeatDataset]:
    """Load, preprocess, segment and split the MIT-BIH dataset.

    This is the top-level entry point for the full preprocessing pipeline.
    It should be called once per experiment run; the resulting datasets
    are then used directly for model training and evaluation.

    Pipeline per record
    -------------------
    1. Load raw signal + annotations (``load_record``)
    2. Preprocess signal (``preprocess_signal``: clean + band-pass filter)
    3. Segment beats (``segment_record``: fixed-length windows)
    4. Assign to split based on record ID from config

    Normalisation is applied after segmentation to the beat windows.
    Because per-beat z-score uses only each beat's own statistics, it
    is applied identically to all splits.

    Parameters
    ----------
    cfg:
        Loaded configuration dict (from ``src.utils.config.load_config``).
    download:
        Pass-through to ``load_records`` — download missing records.
    normalize:
        If ``True``, apply per-beat z-score normalisation to waveforms.
    allow_padding:
        If ``True``, edge-pad boundary beats instead of discarding them.
    save_manifest:
        Write a JSON manifest of the split to the experiments directory.

    Returns
    -------
    tuple[BeatDataset, BeatDataset, BeatDataset]
        (train_dataset, val_dataset, test_dataset)

    Raises
    ------
    ValueError
        If any record overlap is detected between splits, or if any
        split is completely empty.
    """
    ds_cfg   = cfg["dataset"]
    pp_cfg   = cfg["preprocessing"]
    bp_cfg   = pp_cfg["bandpass"]

    train_ids = [str(r) for r in ds_cfg["train_records"]]
    val_ids   = [str(r) for r in ds_cfg["val_records"]]
    test_ids  = [str(r) for r in ds_cfg["test_records"]]

    # Verify zero overlap before touching any data
    verify_no_overlap(train_ids, val_ids, test_ids)

    # ── Load all records ─────────────────────────────────────────────
    all_ids = train_ids + val_ids + test_ids
    logger.info(
        "Loading %d records (%d train / %d val / %d test) ...",
        len(all_ids), len(train_ids), len(val_ids), len(test_ids),
    )

    records: list[ECGRecord] = load_records(
        record_ids=all_ids,
        data_dir=ds_cfg["raw_dir"],
        fs=ds_cfg["fs"],
        lead=ds_cfg["lead"],
        download=download,
        skip_on_error=False,
    )

    # Build a lookup dict: record_id → ECGRecord
    record_map: dict[str, ECGRecord] = {r.record_id: r for r in records}

    # ── Preprocess + segment per split ───────────────────────────────
    train_ds = _process_split(
        record_ids=train_ids,
        record_map=record_map,
        bp_cfg=bp_cfg,
        fs=ds_cfg["fs"],
        pp_cfg=pp_cfg,
        allow_padding=allow_padding,
        split_name="train",
    )
    val_ds = _process_split(
        record_ids=val_ids,
        record_map=record_map,
        bp_cfg=bp_cfg,
        fs=ds_cfg["fs"],
        pp_cfg=pp_cfg,
        allow_padding=allow_padding,
        split_name="val",
    )
    test_ds = _process_split(
        record_ids=test_ids,
        record_map=record_map,
        bp_cfg=bp_cfg,
        fs=ds_cfg["fs"],
        pp_cfg=pp_cfg,
        allow_padding=allow_padding,
        split_name="test",
    )

    # ── Normalise ────────────────────────────────────────────────────
    if normalize:
        logger.info("Applying per-beat z-score normalisation ...")
        train_ds = _apply_normalisation(train_ds)
        val_ds   = _apply_normalisation(val_ds)
        test_ds  = _apply_normalisation(test_ds)

    # ── Final integrity checks ────────────────────────────────────────
    _check_not_empty(train_ds, "train")
    _check_not_empty(val_ds,   "val")
    _check_not_empty(test_ds,  "test")
    verify_record_level_integrity(train_ds, val_ds, test_ds)

    # ── Log summary ──────────────────────────────────────────────────
    _log_split_summary(train_ds, val_ds, test_ds)

    # ── Save manifest ────────────────────────────────────────────────
    if save_manifest:
        manifest_path = _save_manifest(
            cfg=cfg,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            normalize=normalize,
            allow_padding=allow_padding,
        )
        logger.info("Split manifest saved → %s", manifest_path)

    return train_ds, val_ds, test_ds


def verify_no_overlap(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> None:
    """Raise ValueError if any record ID appears in more than one split.

    Parameters
    ----------
    train_ids, val_ids, test_ids:
        Record ID lists (string form).

    Raises
    ------
    ValueError
        On any overlap.
    """
    train_set = set(train_ids)
    val_set   = set(val_ids)
    test_set  = set(test_ids)

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set   & test_set

    if tv or tt or vt:
        raise ValueError(
            f"Record split overlap detected — "
            f"train∩val={sorted(tv)}, "
            f"train∩test={sorted(tt)}, "
            f"val∩test={sorted(vt)}"
        )
    logger.debug(
        "Record overlap check passed: %d train / %d val / %d test (all disjoint)",
        len(train_ids), len(val_ids), len(test_ids),
    )


def verify_record_level_integrity(
    train_ds: BeatDataset,
    val_ds: BeatDataset,
    test_ds: BeatDataset,
) -> None:
    """Verify that no record ID appears in more than one BeatDataset.

    This is a post-assembly check that confirms the split was applied
    correctly to the actual extracted beats.

    Raises
    ------
    ValueError
        If any record appears in multiple splits.
    """
    train_records = set(train_ds.source_records)
    val_records   = set(val_ds.source_records)
    test_records  = set(test_ds.source_records)

    tv = train_records & val_records
    tt = train_records & test_records
    vt = val_records   & test_records

    if tv or tt or vt:
        raise ValueError(
            f"Post-assembly record overlap detected in beat datasets — "
            f"train∩val={sorted(tv)}, "
            f"train∩test={sorted(tt)}, "
            f"val∩test={sorted(vt)}"
        )
    logger.info(
        "Post-assembly record integrity: PASSED "
        "(%d train records, %d val records, %d test records — all disjoint)",
        len(train_records), len(val_records), len(test_records),
    )


def get_split_record_ids(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Return the train/val/test record ID lists from config as strings.

    Useful for scripts that need to know the split without loading data.
    """
    ds = cfg["dataset"]
    return {
        "train": [str(r) for r in ds["train_records"]],
        "val":   [str(r) for r in ds["val_records"]],
        "test":  [str(r) for r in ds["test_records"]],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_split(
    record_ids: list[str],
    record_map: dict[str, ECGRecord],
    bp_cfg: dict[str, Any],
    fs: int,
    pp_cfg: dict[str, Any],
    allow_padding: bool,
    split_name: str,
) -> BeatDataset:
    """Preprocess + segment all records assigned to one split."""
    datasets: list[BeatDataset] = []

    before = pp_cfg["beat_before_samples"]
    after  = pp_cfg["beat_after_samples"]

    for rid in record_ids:
        if rid not in record_map:
            logger.error(
                "Record %s assigned to %s split but was not loaded — skipping",
                rid, split_name,
            )
            continue

        rec = record_map[rid]

        # Preprocess signal in-place (creates a new array, not mutating)
        preprocessed_signal = preprocess_signal(
            signal=rec.signal,
            fs=fs,
            lowcut=bp_cfg["lowcut"],
            highcut=bp_cfg["highcut"],
            order=bp_cfg["order"],
            record_id=rec.record_id,
        )

        # Temporarily swap signal for segmentation (non-destructive)
        import dataclasses
        rec_processed = dataclasses.replace(rec, signal=preprocessed_signal)

        ds = segment_record(
            record=rec_processed,
            before=before,
            after=after,
            allow_padding=allow_padding,
        )

        if ds.is_empty():
            logger.warning(
                "%s split — record %s produced 0 beats after segmentation",
                split_name, rid,
            )
        else:
            datasets.append(ds)

    if not datasets:
        # Return empty dataset rather than crashing — caller will catch
        window_size = before + after
        from src.data.segmentation import _empty_dataset
        return _empty_dataset(window_size)

    return _concatenate_datasets(datasets)


def _apply_normalisation(ds: BeatDataset) -> BeatDataset:
    """Return a new BeatDataset with z-score normalised waveforms."""
    import dataclasses
    normalised = normalize_beats_zscore(ds.waveforms)
    return dataclasses.replace(ds, waveforms=normalised)


def _check_not_empty(ds: BeatDataset, split_name: str) -> None:
    """Raise ValueError if the dataset is empty."""
    if ds.is_empty():
        raise ValueError(
            f"The '{split_name}' dataset is empty after preprocessing and "
            "segmentation.  Check that the assigned records were loaded "
            "successfully and contain annotated beats."
        )


def _log_split_summary(
    train_ds: BeatDataset,
    val_ds: BeatDataset,
    test_ds: BeatDataset,
) -> None:
    """Log a compact summary table of the three splits."""
    from src.data.annotations import AAMI_CLASS_TO_INDEX

    total = train_ds.n_beats + val_ds.n_beats + test_ds.n_beats
    header = f"{'Split':<8} {'Records':>7} {'Beats':>8}  " + \
             "  ".join(f"{c:>5}" for c in AAMI_CLASS_TO_INDEX)
    logger.info(header)
    logger.info("-" * len(header))

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        counts = "  ".join(
            f"{ds.class_distribution[c]:>5}" for c in AAMI_CLASS_TO_INDEX
        )
        logger.info(
            f"{name:<8} {len(ds.source_records):>7} {ds.n_beats:>8}  {counts}"
        )

    logger.info("-" * len(header))
    logger.info(
        "%-8s %7s %8d", "total", "", total
    )


def _save_manifest(
    cfg: dict[str, Any],
    train_ds: BeatDataset,
    val_ds: BeatDataset,
    test_ds: BeatDataset,
    normalize: bool,
    allow_padding: bool,
) -> Path:
    """Write split manifest JSON and return its path."""
    from src.data.annotations import AAMI_CLASS_TO_INDEX

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg["output"]["experiments_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"split_manifest_{timestamp}.json"

    def ds_summary(ds: BeatDataset, split_name: str) -> dict:
        total = ds.n_beats
        return {
            "split": split_name,
            "record_ids": sorted(ds.source_records),
            "n_records": len(ds.source_records),
            "n_beats": total,
            "class_counts": ds.class_distribution,
            "class_percentages": {
                cls: round(100.0 * cnt / total, 2) if total > 0 else 0.0
                for cls, cnt in ds.class_distribution.items()
            },
        }

    pp = cfg["preprocessing"]
    manifest = {
        "created_at": timestamp,
        "seed": cfg["seed"],
        "dataset": {
            "name": cfg["dataset"]["name"],
            "fs": cfg["dataset"]["fs"],
            "lead": cfg["dataset"]["lead"],
        },
        "preprocessing": {
            "bandpass_lowcut_hz":  pp["bandpass"]["lowcut"],
            "bandpass_highcut_hz": pp["bandpass"]["highcut"],
            "bandpass_order":      pp["bandpass"]["order"],
            "normalization":       pp["normalization"] if normalize else "none",
            "beat_before_samples": pp["beat_before_samples"],
            "beat_after_samples":  pp["beat_after_samples"],
            "beat_window_size":    pp["beat_window_size"],
            "allow_padding":       allow_padding,
        },
        "splits": {
            "train": ds_summary(train_ds, "train"),
            "val":   ds_summary(val_ds,   "val"),
            "test":  ds_summary(test_ds,  "test"),
        },
        "leakage_notes": [
            "Split is patient/record-wise: no beat from a test record "
            "appears in train or val.",
            "Record assignments are fixed in config.yaml and not computed "
            "from the data at runtime.",
            "Per-beat z-score normalisation uses only each beat's own "
            "statistics — no dataset-wide fitting.",
            "F and Q class stratification at record level is not possible "
            "due to the small number of records and class rarity.",
        ],
    }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest_path
