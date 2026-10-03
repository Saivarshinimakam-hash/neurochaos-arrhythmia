"""
src/training/train_classical.py
--------------------------------
Training script for the three classical ML baselines.

Experiment 1 — Classical ML on raw waveforms (flattened 288-dim feature)
Experiment 4 — Classical ML on NeuroChaos features (13-dim feature)

Both experiments share this script; the ``--features`` argument selects
which feature representation to use.

Usage
-----
From project root (venv activated):

    # Experiment 1 — raw waveform features
    python -m src.training.train_classical --features waveform

    # Experiment 4 — NeuroChaos features
    python -m src.training.train_classical --features neurochaos

Output (saved to results/experiments/<timestamp>_classical_<features>/)
-----------------------------------------------------------------------
  - metrics_lr.json / metrics_rf.json / metrics_svm.json
  - model_lr.joblib / model_rf.joblib / model_svm.joblib
  - confusion_matrix_lr.png  (etc.)
  - roc_curves.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.utils.logger import setup_logging, get_logger
from src.data.splitting import build_datasets
from src.models.classical import (
    build_logistic_regression,
    build_random_forest,
    build_svm,
    flatten_waveforms,
)
from src.neurochaos.extractor import NeuroChaosExtractor, NeuroChaosImputer
from src.neurochaos.feature_cache import (
    cache_path,
    save_feature_cache,
    save_checkpoint,
    load_feature_cache,
)
from src.evaluation.metrics import compute_metrics, CLASS_NAMES
from src.evaluation.confusion_matrix import save_confusion_matrix
from src.evaluation.plots import save_roc_curves

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    features: str,
    cfg_path: Path | None = None,
    recompute: bool = False,
) -> None:
    cfg = load_config(cfg_path)
    set_global_seed(cfg["seed"])

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg["output"]["experiments_dir"]) / f"{timestamp}_classical_{features}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(cfg["output"]["logs_dir"]) / f"{timestamp}_classical_{features}.log"
    setup_logging(level="INFO", log_file=log_file)
    logger.info("=== Classical ML training | features=%s ===", features)

    # ── Load data ──────────────────────────────────────────────────────
    logger.info("Loading dataset ...")
    train_ds, val_ds, test_ds = build_datasets(cfg, download=True, normalize=True,
                                               save_manifest=False)
    logger.info("Train: %d  Val: %d  Test: %d beats",
                train_ds.n_beats, val_ds.n_beats, test_ds.n_beats)

    # ── Feature extraction ────────────────────────────────────────────
        # ── Feature extraction ────────────────────────────────────────────
    if features == "waveform":
        logger.info(
            "Using flattened waveform features (dim=%d)",
            train_ds.waveforms.shape[1],
        )

        X_train = flatten_waveforms(train_ds.waveforms)
        X_val = flatten_waveforms(val_ds.waveforms)
        X_test = flatten_waveforms(test_ds.waveforms)

    elif features == "neurochaos":
        logger.info("Using NeuroChaos features with persistent cache")

        cache_dir = Path(cfg["output"]["cache_dir"]) / "neurochaos"
        extractor = NeuroChaosExtractor(cfg)

        datasets = {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
        }

        raw_features = {}

        for split_name, ds in datasets.items():
            path = cache_path(cache_dir, split_name)

            if path.exists() and not recompute:
                logger.info(
                    "Loading cached NeuroChaos features: %s",
                    path,
                )

                cached = load_feature_cache(
                    path=path,
                    split_name=split_name,
                    cfg=cfg,
                )

                # Verify cache matches the current dataset.
                if not np.array_equal(cached["labels"], ds.labels):
                    raise ValueError(
                        f"{split_name} cache labels do not match "
                        "the current dataset."
                    )

                if cached["record_ids"] != list(ds.record_ids):
                    raise ValueError(
                        f"{split_name} cache record IDs do not match "
                        "the current dataset."
                    )

                if not np.array_equal(
                    cached["beat_indices"],
                    ds.beat_indices,
                ):
                    raise ValueError(
                        f"{split_name} cache beat indices do not match "
                        "the current dataset."
                    )

                if not np.array_equal(
                    cached["r_peak_samples"],
                    ds.r_peak_samples,
                ):
                    raise ValueError(
                        f"{split_name} cache R-peak positions do not match "
                        "the current dataset."
                    )

                raw_features[split_name] = cached["features"]

            else:
                if recompute:
                    logger.info(
                        "Recomputing NeuroChaos features for %s ...",
                        split_name,
                    )
                else:
                    logger.info(
                        "No cache found for %s — extracting features ...",
                        split_name,
                    )

                # ---------------------------------------------------------
                # Incremental NeuroChaos extraction
                # ---------------------------------------------------------
                checkpoint_path = path.with_name(
                    f"{split_name}_neurochaos_checkpoint.npz"
                )

                checkpoint_size = 1000
                n_samples = len(ds.waveforms)

                features_list = []
                start_idx = 0

                # Resume from checkpoint if available.
                if checkpoint_path.exists() and not recompute:
                    logger.info(
                        "Found NeuroChaos checkpoint: %s",
                        checkpoint_path,
                    )

                    checkpoint = load_feature_cache(
                        path=checkpoint_path,
                        split_name=split_name,
                        cfg=cfg,
                    )

                    completed = len(checkpoint["features"])

                    # Verify checkpoint belongs to the beginning
                    # of the current dataset.
                    if not np.array_equal(
                        checkpoint["labels"],
                        ds.labels[:completed],
                    ):
                        raise ValueError(
                            f"{split_name} checkpoint labels do not "
                            "match the current dataset."
                        )

                    if checkpoint["record_ids"] != list(
                        ds.record_ids[:completed]
                    ):
                        raise ValueError(
                            f"{split_name} checkpoint record IDs do not "
                            "match the current dataset."
                        )

                    if not np.array_equal(
                        checkpoint["beat_indices"],
                        ds.beat_indices[:completed],
                    ):
                        raise ValueError(
                            f"{split_name} checkpoint beat indices do not "
                            "match the current dataset."
                        )

                    if not np.array_equal(
                        checkpoint["r_peak_samples"],
                        ds.r_peak_samples[:completed],
                    ):
                        raise ValueError(
                            f"{split_name} checkpoint R-peak positions do "
                            "not match the current dataset."
                        )

                    features_list.append(checkpoint["features"])
                    start_idx = completed

                    logger.info(
                        "Resuming %s from %d/%d beats",
                        split_name,
                        start_idx,
                        n_samples,
                    )

                elif checkpoint_path.exists() and recompute:
                    checkpoint_path.unlink()

                # Extract remaining beats in chunks.
                for chunk_start in range(
                    start_idx,
                    n_samples,
                    checkpoint_size,
                ):
                    chunk_end = min(
                        chunk_start + checkpoint_size,
                        n_samples,
                    )

                    logger.info(
                        "Extracting %s beats %d-%d/%d",
                        split_name,
                        chunk_start + 1,
                        chunk_end,
                        n_samples,
                    )

                    X_chunk = extractor.extract_dataset(
                        ds.waveforms[chunk_start:chunk_end],
                        show_progress=True,
                    )

                    features_list.append(X_chunk)

                    X_partial = np.concatenate(
                        features_list,
                        axis=0,
                    )

                    save_checkpoint(
                        path=checkpoint_path,
                        features=X_partial,
                        labels=ds.labels[:chunk_end],
                        record_ids=list(
                            ds.record_ids[:chunk_end]
                        ),
                        beat_indices=ds.beat_indices[:chunk_end],
                        r_peak_samples=ds.r_peak_samples[:chunk_end],
                        split_name=split_name,
                        cfg=cfg,
                        completed=chunk_end,
                    )

                    logger.info(
                        "Checkpoint saved: %d/%d beats",
                        chunk_end,
                        n_samples,
                    )

                # Complete feature matrix.
                X_raw = np.concatenate(
                    features_list,
                    axis=0,
                )

                # Save final validated cache.
                save_feature_cache(
                    path=path,
                    features=X_raw,
                    labels=ds.labels,
                    record_ids=list(ds.record_ids),
                    beat_indices=ds.beat_indices,
                    r_peak_samples=ds.r_peak_samples,
                    split_name=split_name,
                    cfg=cfg,
                )

                # Remove temporary checkpoint after successful completion.
                if checkpoint_path.exists():
                    checkpoint_path.unlink()

                logger.info(
                    "NeuroChaos cache saved: %s",
                    path,
                )

                raw_features[split_name] = X_raw

                X_raw = extractor.extract_dataset(
                    ds.waveforms,
                    show_progress=True,
                )

                save_feature_cache(
                    path=path,
                    features=X_raw,
                    labels=ds.labels,
                    record_ids=list(ds.record_ids),
                    beat_indices=ds.beat_indices,
                    r_peak_samples=ds.r_peak_samples,
                    split_name=split_name,
                    cfg=cfg,
                )

                logger.info(
                    "NeuroChaos cache saved: %s",
                    path,
                )

                raw_features[split_name] = X_raw

        X_train_raw = raw_features["train"]
        X_val_raw = raw_features["val"]
        X_test_raw = raw_features["test"]

        # Impute NaN using training data only.
        imputer = NeuroChaosImputer()

        X_train = imputer.fit_transform(X_train_raw)
        X_val = imputer.transform(X_val_raw)
        X_test = imputer.transform(X_test_raw)

        joblib.dump(
            imputer,
            out_dir / "neurochaos_imputer.joblib",
        )

        logger.info("NeuroChaos imputer saved")

    else:
        raise ValueError(f"Unknown features mode: {features!r}")
    y_train = train_ds.labels
    y_val   = val_ds.labels
    y_test  = test_ds.labels
    # ── Train + evaluate each model ───────────────────────────────────
    models = {
        "lr":  build_logistic_regression(cfg),
        "rf":  build_random_forest(cfg),
        "svm": build_svm(cfg),
    }

    all_results: dict[str, Any] = {}

    for name, model in models.items():
        logger.info("--- Training %s ---", name.upper())
        t0 = time.time()
        model.fit(X_train, y_train)
        train_secs = time.time() - t0
        logger.info("%s trained in %.1f s", name.upper(), train_secs)

        # Val predictions
        y_val_pred  = model.predict(X_val)
        y_val_prob  = _safe_predict_proba(model, X_val)

        # Test predictions (kept separate — only for final reporting)
        y_test_pred = model.predict(X_test)
        y_test_prob = _safe_predict_proba(model, X_test)

        val_metrics  = compute_metrics(y_val,  y_val_pred,  y_val_prob,  split="val")
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob, split="test")

        logger.info("Val  accuracy=%.4f  macro-F1=%.4f  bal-acc=%.4f",
                    val_metrics["accuracy"], val_metrics["macro_f1"],
                    val_metrics["balanced_accuracy"])
        logger.info("Test accuracy=%.4f  macro-F1=%.4f  bal-acc=%.4f",
                    test_metrics["accuracy"], test_metrics["macro_f1"],
                    test_metrics["balanced_accuracy"])

        # Save model
        model_path = out_dir / f"model_{name}.joblib"
        joblib.dump(model, model_path)
        logger.info("Model saved → %s", model_path)

        # Save metrics
        result = {
            "model": name,
            "features": features,
            "train_seconds": round(train_secs, 2),
            "val": val_metrics,
            "test": test_metrics,
        }
        metrics_path = out_dir / f"metrics_{name}.json"
        with metrics_path.open("w") as fh:
            json.dump(result, fh, indent=2)

        # Confusion matrices
        save_confusion_matrix(
            y_true=y_val, y_pred=y_val_pred,
            title=f"{name.upper()} Validation Confusion Matrix",
            save_path=out_dir / f"confusion_val_{name}.png",
        )
        save_confusion_matrix(
            y_true=y_test, y_pred=y_test_pred,
            title=f"{name.upper()} Test Confusion Matrix",
            save_path=out_dir / f"confusion_test_{name}.png",
        )

        # ROC curves
        if y_val_prob is not None:
            save_roc_curves(
                y_true=y_val, y_prob=y_val_prob,
                title=f"{name.upper()} ({features}) — Val ROC",
                save_path=out_dir / f"roc_val_{name}.png",
            )
        if y_test_prob is not None:
            save_roc_curves(
                y_true=y_test, y_prob=y_test_prob,
                title=f"{name.upper()} ({features}) — Test ROC",
                save_path=out_dir / f"roc_test_{name}.png",
            )

        all_results[name] = result

    # Save combined summary
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(all_results, fh, indent=2)
    logger.info("Results saved → %s", out_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_predict_proba(model: Any, X: np.ndarray) -> np.ndarray | None:
    """Return predict_proba output or None if not available."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception as exc:
            logger.warning("predict_proba failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train classical ML baselines on MIT-BIH arrhythmia data"
    )
    parser.add_argument(
        "--features",
        choices=["waveform", "neurochaos"],
        default="waveform",
        help="Feature representation (default: waveform)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: <project_root>/config.yaml)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute NeuroChaos features even if cached features exist.",
    )
    args = parser.parse_args()
    run(
        features=args.features,
        cfg_path=args.config,
        recompute=args.recompute,
    )

if __name__ == "__main__":
    main()
