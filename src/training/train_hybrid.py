"""
src/training/train_hybrid.py
-----------------------------
Training script for Experiment 5 — Hybrid Neural + NeuroChaos model.

Usage
-----
    python -m src.training.train_hybrid                 # CNN backbone
    python -m src.training.train_hybrid --backbone bilstm

Output  (results/experiments/<timestamp>_hybrid_<backbone>/)
------------------------------------------------------------
  hybrid_<backbone>_best.keras  — best checkpoint
  metrics_val.json / metrics_test.json
  history.json
  confusion_val.png / confusion_test.png
  roc_val.png / roc_test.png
  training_history.png
  neurochaos_imputer.joblib     — fitted imputer (train stats only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.utils.logger import setup_logging, get_logger
from src.data.splitting import build_datasets
from src.models.hybrid import build_hybrid, prepare_hybrid_inputs
from src.neurochaos.extractor import NeuroChaosImputer
from src.neurochaos.feature_cache import (
    cache_path,
    load_feature_cache,
)
from src.training.keras_utils import (
    make_callbacks, compute_class_weights, save_history, keras_predict,
)
from src.evaluation.metrics import compute_metrics, metrics_summary_line
from src.evaluation.confusion_matrix import save_confusion_matrix
from src.evaluation.plots import save_roc_curves, save_pr_curves, save_training_history

logger = get_logger(__name__)


def run(backbone: str = "cnn", cfg_path: Path | None = None) -> None:
    cfg = load_config(cfg_path)
    set_global_seed(cfg["seed"])

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name  = f"{timestamp}_hybrid_{backbone}"
    out_dir   = Path(cfg["output"]["experiments_dir"]) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        level="INFO",
        log_file=Path(cfg["output"]["logs_dir"]) / f"{exp_name}.log",
    )
    logger.info("=== Experiment 5: Hybrid %s + NeuroChaos ===", backbone.upper())
    logger.info("Output: %s", out_dir)

    # ── Load datasets ──────────────────────────────────────────────────
    logger.info("Building datasets ...")
    train_ds, val_ds, test_ds = build_datasets(
        cfg, download=True, normalize=True, save_manifest=False
    )
    logger.info("Train %d  Val %d  Test %d beats",
                train_ds.n_beats, val_ds.n_beats, test_ds.n_beats)

    # ── Extract NeuroChaos features ────────────────────────────────────
    # ── Load cached NeuroChaos features ─────────────────────────────
    logger.info("Loading cached NeuroChaos features ...")

    cache_dir = Path(cfg["output"]["cache_dir"]) / "neurochaos"

    train_cache = load_feature_cache(
        cache_path(cache_dir, "train"),
        "train",
        cfg,
    )

    val_cache = load_feature_cache(
        cache_path(cache_dir, "val"),
        "val",
        cfg,
    )

    test_cache = load_feature_cache(
        cache_path(cache_dir, "test"),
        "test",
        cfg,
    )

    X_chaos_train_raw = train_cache["features"]
    X_chaos_val_raw = val_cache["features"]
    X_chaos_test_raw = test_cache["features"]

# Verify cache labels match the current datasets.
    if not np.array_equal(train_cache["labels"], train_ds.labels):
        raise ValueError("Train NeuroChaos cache labels do not match dataset.")

    if not np.array_equal(val_cache["labels"], val_ds.labels):
        raise ValueError("Validation NeuroChaos cache labels do not match dataset.")

    if not np.array_equal(test_cache["labels"], test_ds.labels):
        raise ValueError("Test NeuroChaos cache labels do not match dataset.")

# Verify record provenance.
    if not np.array_equal(train_cache["labels"], train_ds.labels):
        raise ValueError("Train NeuroChaos cache labels do not match dataset.")

    if not np.array_equal(val_cache["labels"], val_ds.labels):
        raise ValueError("Validation NeuroChaos cache labels do not match dataset.")

    if not np.array_equal(test_cache["labels"], test_ds.labels):
        raise ValueError("Test NeuroChaos cache labels do not match dataset.")
    logger.info(
    "Cached NeuroChaos features loaded: train=%s val=%s test=%s",
    X_chaos_train_raw.shape,
    X_chaos_val_raw.shape,
    X_chaos_test_raw.shape,
    )

    # Impute NaN — fitted on train only.
    imputer = NeuroChaosImputer()

    X_chaos_train = imputer.fit_transform(X_chaos_train_raw)
    X_chaos_val = imputer.transform(X_chaos_val_raw)
    X_chaos_test = imputer.transform(X_chaos_test_raw)

    imputer_path = out_dir / "neurochaos_imputer.joblib"
    joblib.dump(imputer, imputer_path)

    logger.info("Imputer saved → %s", imputer_path)

    # Impute NaN — fitted on train only (leakage-safe)
    imputer = NeuroChaosImputer()
    X_chaos_train = imputer.fit_transform(X_chaos_train_raw)
    X_chaos_val   = imputer.transform(X_chaos_val_raw)
    X_chaos_test  = imputer.transform(X_chaos_test_raw)

    imputer_path = out_dir / "neurochaos_imputer.joblib"
    joblib.dump(imputer, imputer_path)
    logger.info("Imputer saved → %s", imputer_path)

    # ── Prepare model inputs ───────────────────────────────────────────
    X_wave_train, X_chaos_train = prepare_hybrid_inputs(
        train_ds.waveforms, X_chaos_train
    )
    X_wave_val,   X_chaos_val   = prepare_hybrid_inputs(
        val_ds.waveforms,   X_chaos_val
    )
    X_wave_test,  X_chaos_test  = prepare_hybrid_inputs(
        test_ds.waveforms,  X_chaos_test
    )
    y_train = train_ds.labels
    y_val   = val_ds.labels
    y_test  = test_ds.labels

    # ── Build model ────────────────────────────────────────────────────
    logger.info("Building Hybrid model (backbone=%s) ...", backbone)
    model = build_hybrid(cfg, backbone=backbone)
    model.summary(print_fn=logger.info)

    class_weights = compute_class_weights(y_train)
    logger.info("Class weights: %s", class_weights)

    # ── Train ──────────────────────────────────────────────────────────
    h_cfg     = cfg["hybrid"]
    callbacks = make_callbacks(
        out_dir,
        model_name=f"hybrid_{backbone}",
        patience=h_cfg["early_stopping_patience"],
    )

    logger.info("Training ...")
    t0 = time.time()
    history = model.fit(
        [X_wave_train, X_chaos_train], y_train,
        validation_data=([X_wave_val, X_chaos_val], y_val),
        epochs=h_cfg["epochs"],
        batch_size=h_cfg["batch_size"],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    train_secs = time.time() - t0
    logger.info("Training completed in %.1f s", train_secs)

    save_history(history.history, out_dir / "history.json")
    save_training_history(
        history.history,
        title=f"Hybrid {backbone.upper()} Training History",
        save_path=out_dir / "training_history.png",
    )

    # ── Evaluate ───────────────────────────────────────────────────────
    eval_sets = [
        ("val",  [X_wave_val,  X_chaos_val],  y_val),
        ("test", [X_wave_test, X_chaos_test], y_test),
    ]
    for split_name, X_list, y in eval_sets:
        y_pred, y_prob = keras_predict(model, X_list)
        metrics = compute_metrics(y, y_pred, y_prob, split=split_name)
        logger.info(metrics_summary_line(metrics))
        logger.info("\n%s", metrics["classification_report"])

        with (out_dir / f"metrics_{split_name}.json").open("w") as fh:
            json.dump(
                {k: v for k, v in metrics.items() if k != "classification_report"},
                fh, indent=2,
            )

        save_confusion_matrix(
            y, y_pred,
            title=f"Hybrid {backbone.upper()} — {split_name.capitalize()}",
            save_path=out_dir / f"confusion_{split_name}.png",
        )
        save_roc_curves(
            y, y_prob,
            title=f"Hybrid {backbone.upper()} — {split_name.capitalize()} ROC",
            save_path=out_dir / f"roc_{split_name}.png",
        )
        save_pr_curves(
            y, y_prob,
            title=f"Hybrid {backbone.upper()} — {split_name.capitalize()} PR",
            save_path=out_dir / f"pr_{split_name}.png",
        )

    logger.info("All outputs saved to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Hybrid Neural+NeuroChaos model on MIT-BIH arrhythmia data"
    )
    parser.add_argument(
        "--backbone",
        choices=["cnn", "bilstm"],
        default="cnn",
        help="Neural backbone (default: cnn)",
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    run(backbone=args.backbone, cfg_path=args.config)


if __name__ == "__main__":
    main()
