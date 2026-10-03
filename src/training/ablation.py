"""
src/training/ablation.py
-------------------------
Ablation study: what does each NeuroChaos feature group contribute
to the Hybrid model?

Study design
------------
Six conditions are trained and evaluated on the SAME data split:

  Condition               Neural backbone   Chaos features used
  ──────────────────────  ────────────────  ──────────────────────────────
  neural_only             CNN               none  (waveform input only)
  neural_entropy          CNN               sample_entropy, perm_entropy,
                                            approx_entropy
  neural_recurrence       CNN               rr, det, lam
  neural_lyapunov         CNN               lle
  neural_fractal          CNN               dfa_alpha, hurst, hfd
  neural_all_chaos        CNN               all 13 NeuroChaos features

Each condition produces its own metrics JSON and confusion matrix so
the contribution of each feature group can be isolated.

Important
---------
- NeuroChaos features are extracted ONCE from the training and test sets
  and then sliced per condition — no repeated expensive extraction.
- The neural-only condition uses a standard single-input CNN (cnn.py),
  NOT the two-input Hybrid model, so the comparison is fair.
- Class weights are applied identically across all conditions.
- Early stopping uses the same patience.

Usage
-----
    python -m src.training.ablation

Output  (results/experiments/<timestamp>_ablation/)
---------------------------------------------------
  condition_<name>/metrics_val.json
  condition_<name>/metrics_test.json
  condition_<name>/confusion_val.png
  condition_<name>/confusion_test.png
  ablation_summary.json          — all conditions side-by-side
  ablation_macro_f1.png          — comparison bar chart
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.utils.logger import setup_logging, get_logger
from src.data.splitting import build_datasets
from src.models.cnn import build_cnn, prepare_cnn_inputs
from src.models.hybrid import build_hybrid, prepare_hybrid_inputs
from src.neurochaos.extractor import FEATURE_NAMES
from src.training.keras_utils import (
    make_callbacks, compute_class_weights, keras_predict,
)
from src.evaluation.metrics import compute_metrics, metrics_summary_line
from src.evaluation.confusion_matrix import save_confusion_matrix
from src.evaluation.plots import save_model_comparison_bar
from src.neurochaos.feature_cache import load_feature_cache

logger = get_logger(__name__)

# ── Feature group definitions ─────────────────────────────────────────────
# Each condition maps to a list of FEATURE_NAMES indices (or None = no chaos)

ABLATION_CONDITIONS: dict[str, list[int] | None] = {
    "neural_only":       None,
    "neural_entropy":    [
        FEATURE_NAMES.index("sample_entropy"),
        FEATURE_NAMES.index("permutation_entropy"),
        FEATURE_NAMES.index("approximate_entropy"),
    ],
    "neural_recurrence": [
        FEATURE_NAMES.index("rr"),
        FEATURE_NAMES.index("det"),
        FEATURE_NAMES.index("lam"),
    ],
    "neural_lyapunov":   [
        FEATURE_NAMES.index("lle"),
    ],
    "neural_fractal":    [
        FEATURE_NAMES.index("dfa_alpha"),
        FEATURE_NAMES.index("hurst"),
        FEATURE_NAMES.index("hfd"),
    ],
    "neural_all_chaos":  list(range(len(FEATURE_NAMES))),  # all 13
}


def run(cfg_path: Path | None = None) -> None:
    cfg = load_config(cfg_path)
    set_global_seed(cfg["seed"])

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg["output"]["experiments_dir"]) / f"{timestamp}_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        level="INFO",
        log_file=Path(cfg["output"]["logs_dir"]) / f"{timestamp}_ablation.log",
    )
    logger.info("=== Ablation Study ===")
    logger.info("Conditions: %s", list(ABLATION_CONDITIONS.keys()))
    logger.info("Output: %s", out_dir)

    # ── Load data ──────────────────────────────────────────────────────
    logger.info("Building datasets ...")
    train_ds, val_ds, test_ds = build_datasets(
        cfg, download=True, normalize=True, save_manifest=False
    )
    logger.info("Train %d  Val %d  Test %d",
                train_ds.n_beats, val_ds.n_beats, test_ds.n_beats)

    y_train = train_ds.labels
    y_val   = val_ds.labels
    y_test  = test_ds.labels
    class_weights = compute_class_weights(y_train)

    # Waveform inputs (shared across all conditions)
    X_wave_train = prepare_cnn_inputs(train_ds.waveforms)
    X_wave_val   = prepare_cnn_inputs(val_ds.waveforms)
    X_wave_test  = prepare_cnn_inputs(test_ds.waveforms)

    # ── Extract ALL NeuroChaos features once ──────────────────────────
    # ── Load cached NeuroChaos features ────────────────────────────────

    cache_dir = PROJECT_ROOT / cfg["output"]["cache_dir"] / "neurochaos"

    logger.info("Loading cached NeuroChaos features ...")

    train_cache = load_feature_cache(
        cache_dir / "train_neurochaos.npz",
        split_name="train",
        cfg=cfg,
    )

    val_cache = load_feature_cache(
        cache_dir / "val_neurochaos.npz",
        split_name="val",
        cfg=cfg,
    )

    test_cache = load_feature_cache(
        cache_dir / "test_neurochaos.npz",
        split_name="test",
        cfg=cfg,
    )
    X_chaos_train_full = train_cache["features"]
    X_chaos_val_full = val_cache["features"]
    X_chaos_test_full = test_cache["features"]

    logger.info(
        "Cached NeuroChaos features loaded: train=%s val=%s test=%s",
        X_chaos_train_full.shape,
        X_chaos_val_full.shape,
        X_chaos_test_full.shape,
    )

    # ── Run each ablation condition ───────────────────────────────────
    summary: dict[str, Any] = {}

    for cond_name, feat_indices in ABLATION_CONDITIONS.items():
        logger.info("─── Condition: %s ───", cond_name)
        set_global_seed(cfg["seed"])   # same seed for every condition

        cond_dir = out_dir / f"condition_{cond_name}"
        cond_dir.mkdir(parents=True, exist_ok=True)

        cond_summary = _run_condition(
            cond_name=cond_name,
            feat_indices=feat_indices,
            cfg=cfg,
            out_dir=cond_dir,
            X_wave_train=X_wave_train,
            X_wave_val=X_wave_val,
            X_wave_test=X_wave_test,
            X_chaos_train=X_chaos_train_full,
            X_chaos_val=X_chaos_val_full,
            X_chaos_test=X_chaos_test_full,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            class_weights=class_weights,
        )
        summary[cond_name] = cond_summary

    # ── Save aggregated summary ───────────────────────────────────────
    summary_path = out_dir / "ablation_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Ablation summary → %s", summary_path)

    # ── Comparison chart ──────────────────────────────────────────────
    cond_names   = list(summary.keys())
    test_metrics = [summary[c]["test_metrics"] for c in cond_names]

    save_model_comparison_bar(
        model_names=cond_names,
        metrics_list=test_metrics,
        metric_key="macro_f1",
        split="test",
        title="Ablation Study — Macro F1",
        save_path=out_dir / "ablation_macro_f1.png",
    )
    save_model_comparison_bar(
        model_names=cond_names,
        metrics_list=test_metrics,
        metric_key="balanced_accuracy",
        split="test",
        title="Ablation Study — Balanced Accuracy",
        save_path=out_dir / "ablation_balanced_acc.png",
    )

    # Print summary table to log
    logger.info("")
    logger.info("%-25s  %8s  %8s  %8s", "Condition", "Acc", "MacroF1", "BalAcc")
    logger.info("-" * 60)
    for cond in cond_names:
        m = summary[cond]["test_metrics"]
        logger.info(
            "%-25s  %8.4f  %8.4f  %8.4f",
            cond, m["accuracy"], m["macro_f1"], m["balanced_accuracy"],
        )
    logger.info("All ablation outputs saved to %s", out_dir)


def _run_condition(
    cond_name: str,
    feat_indices: list[int] | None,
    cfg: dict[str, Any],
    out_dir: Path,
    X_wave_train: np.ndarray,
    X_wave_val: np.ndarray,
    X_wave_test: np.ndarray,
    X_chaos_train: np.ndarray,
    X_chaos_val: np.ndarray,
    X_chaos_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    class_weights: dict[int, float],
) -> dict[str, Any]:
    """Train and evaluate one ablation condition. Returns metrics summary."""

    h_cfg = cfg["hybrid"]

    if feat_indices is None:
        # Neural-only: standard single-input CNN
        model = build_cnn(cfg)
        train_in = X_wave_train
        val_in   = X_wave_val
        test_in  = X_wave_test
        model_tag = "cnn_only"
    else:
        # Hybrid with selected chaos feature subset
        X_chaos_sub_train = X_chaos_train[:, feat_indices]
        X_chaos_sub_val   = X_chaos_val[:,   feat_indices]
        X_chaos_sub_test  = X_chaos_test[:,  feat_indices]

        # Build a hybrid model with n_chaos = len(feat_indices)
        model = _build_hybrid_custom_chaos(cfg, n_chaos=len(feat_indices))
        train_in = [X_wave_train, X_chaos_sub_train]
        val_in   = [X_wave_val,   X_chaos_sub_val]
        test_in  = [X_wave_test,  X_chaos_sub_test]
        model_tag = f"hybrid_{len(feat_indices)}chaos"

    callbacks = make_callbacks(
        out_dir, model_name=model_tag,
        patience=h_cfg["early_stopping_patience"],
    )

    t0 = time.time()
    model.fit(
        train_in, y_train,
        validation_data=(val_in, y_val),
        epochs=h_cfg["epochs"],
        batch_size=h_cfg["batch_size"],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
    )
    train_secs = time.time() - t0
    logger.info("  Trained in %.1f s", train_secs)

    val_pred,  val_prob  = keras_predict(model, val_in)
    test_pred, test_prob = keras_predict(model, test_in)

    val_metrics  = compute_metrics(y_val,  val_pred,  val_prob,  split="val")
    test_metrics = compute_metrics(y_test, test_pred, test_prob, split="test")

    logger.info("  %s", metrics_summary_line(test_metrics))

    # Save per-condition outputs
    with (out_dir / "metrics_val.json").open("w") as fh:
        json.dump({k: v for k, v in val_metrics.items()
                   if k != "classification_report"}, fh, indent=2)
    with (out_dir / "metrics_test.json").open("w") as fh:
        json.dump({k: v for k, v in test_metrics.items()
                   if k != "classification_report"}, fh, indent=2)

    save_confusion_matrix(
        y_val, val_pred,
        title=f"Ablation: {cond_name} — Val",
        save_path=out_dir / "confusion_val.png",
    )
    save_confusion_matrix(
        y_test, test_pred,
        title=f"Ablation: {cond_name} — Test",
        save_path=out_dir / "confusion_test.png",
    )

    return {
        "condition": cond_name,
        "n_chaos_features": 0 if feat_indices is None else len(feat_indices),
        "feature_names": (
            [] if feat_indices is None
            else [FEATURE_NAMES[i] for i in feat_indices]
        ),
        "train_seconds": round(train_secs, 1),
        "val_metrics":  {k: v for k, v in val_metrics.items()
                         if k != "classification_report"},
        "test_metrics": {k: v for k, v in test_metrics.items()
                         if k != "classification_report"},
    }


def _build_hybrid_custom_chaos(cfg: dict[str, Any], n_chaos: int):
    """Build a Hybrid model whose chaos input has n_chaos features (not always 13)."""
    import tensorflow as tf  # type: ignore

    hybrid_cfg = cfg["hybrid"]
    cnn_cfg    = cfg["cnn"]
    emb_dim    = hybrid_cfg["embedding_dim"]
    mlp_units  = hybrid_cfg["mlp_hidden_units"]
    dropout    = hybrid_cfg["dropout"]
    lr         = hybrid_cfg["learning_rate"]
    win_size   = cfg["preprocessing"]["beat_window_size"]

    waveform_in = tf.keras.Input(shape=(win_size, 1), name="waveform_input")
    x = waveform_in
    for f in cnn_cfg["filters"]:
        x = tf.keras.layers.Conv1D(f, cnn_cfg["kernel_size"], padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling1D(cnn_cfg["pool_size"])(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(emb_dim, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    neural_emb = tf.keras.layers.Activation("relu", name="neural_embedding")(x)

    chaos_in     = tf.keras.Input(shape=(n_chaos,), name="chaos_input")
    chaos_normed = tf.keras.layers.BatchNormalization(name="chaos_bn")(chaos_in)

    fused = tf.keras.layers.Concatenate(name="fusion")([neural_emb, chaos_normed])

    z = fused
    for i, units in enumerate(mlp_units):
        z = tf.keras.layers.Dense(units, use_bias=False, name=f"mlp_{i}")(z)
        z = tf.keras.layers.BatchNormalization(name=f"mlp_bn_{i}")(z)
        z = tf.keras.layers.Activation("relu", name=f"mlp_relu_{i}")(z)
        z = tf.keras.layers.Dropout(dropout, name=f"mlp_drop_{i}")(z)

    out = tf.keras.layers.Dense(5, activation="softmax", name="output")(z)

    model = tf.keras.Model(inputs=[waveform_in, chaos_in], outputs=out,
                           name=f"hybrid_{n_chaos}chaos")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NeuroChaos ablation study")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    run(cfg_path=args.config)


if __name__ == "__main__":
    main()
