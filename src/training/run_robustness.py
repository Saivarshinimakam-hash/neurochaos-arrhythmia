"""
src/training/run_robustness.py
-------------------------------
Phase 9 — Noise robustness experiment.

Protocol
--------
A trained model is evaluated on the test set at four noise levels:

  clean   sigma = 0.00  (no noise)
  low     sigma = 0.05  (~26 dB SNR)
  medium  sigma = 0.15  (~16 dB SNR)
  high    sigma = 0.30  (~10 dB SNR)

Noise is additive white Gaussian: x_noisy = x + N(0, σ · std(x_beat))
per beat so SNR is consistent regardless of amplitude.

The SAME test samples are used at every noise level — the only variable
is σ.  This isolates model robustness from dataset composition effects.

This script supports all four model families:
  --model cnn         → loads models/<experiment>/cnn_best.keras
  --model bilstm      → loads bilstm_best.keras
  --model hybrid      → loads hybrid_cnn_best.keras (two-input)
  --model classical   → loads model_rf.joblib (or lr / svm)

Usage
-----
    # CNN robustness (must have run train_cnn.py first)
    python -m src.training.run_robustness --model cnn --checkpoint models/cnn_best.keras

    # RandomForest robustness
    python -m src.training.run_robustness --model rf --checkpoint models/model_rf.joblib

Output  (results/experiments/<timestamp>_robustness_<model>/)
------------------------------------------------------------
  robustness_<model>.json      — metrics at all noise levels
  robustness_<model>.png       — Macro F1 vs noise level bar chart
  confusion_<level>.png        — confusion matrix per noise level
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional
import tensorflow as tf  # type: ignore
import joblib
from src.neurochaos.extractor import NeuroChaosExtractor
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.utils.logger import setup_logging, get_logger
from src.data.splitting import build_datasets
from src.evaluation.robustness import (
    add_gaussian_noise, NOISE_LEVELS, save_robustness_results,
)
from src.evaluation.metrics import compute_metrics, metrics_summary_line
from src.evaluation.confusion_matrix import save_confusion_matrix

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    model_type: str,
    checkpoint: Path,
    cfg_path: Path | None = None,
) -> None:
    """Run the robustness experiment for one trained model.

    Parameters
    ----------
    model_type : "cnn" | "bilstm" | "hybrid" | "lr" | "rf" | "svm"
    checkpoint : path to the saved model file (.keras or .joblib)
    cfg_path   : override default config.yaml location
    """
    cfg = load_config(cfg_path)
    set_global_seed(cfg["seed"])

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(cfg["output"]["experiments_dir"])
        / f"{timestamp}_robustness_{model_type}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        level="INFO",
        log_file=Path(cfg["output"]["logs_dir"]) / f"{timestamp}_robustness_{model_type}.log",
    )
    logger.info("=== Robustness Experiment: %s ===", model_type.upper())
    logger.info("Checkpoint : %s", checkpoint)
    logger.info("Output     : %s", out_dir)

    # ── Load test data ────────────────────────────────────────────────
    logger.info("Loading test dataset ...")
    _, _, test_ds = build_datasets(
        cfg, download=True, normalize=True, save_manifest=False
    )
    logger.info("Test beats: %d", test_ds.n_beats)

    # ── Load model & build predict_fn ────────────────────────────────
    predict_fn = _load_predict_fn(
        model_type=model_type,
        checkpoint=checkpoint,
        cfg=cfg,
        test_waveforms=test_ds.waveforms,
    )

    # ── Evaluate across noise levels ──────────────────────────────────
    y_true   = test_ds.labels
    results: dict[str, Any] = {
        "model_name":    model_type,
        "checkpoint":    str(checkpoint),
        "n_test_beats":  int(test_ds.n_beats),
        "noise_levels":  {},
    }

    for level_name, sigma in NOISE_LEVELS.items():
        logger.info("Noise level: %-8s  sigma=%.2f", level_name, sigma)
        noisy = add_gaussian_noise(test_ds.waveforms, sigma, seed=cfg["seed"])

        y_pred, y_prob = predict_fn(noisy)
        m = compute_metrics(y_true, y_pred, y_prob, split=f"test_{level_name}")
        logger.info("  %s", metrics_summary_line(m))

        results["noise_levels"][level_name] = {
            "sigma_factor": sigma,
            "metrics": {k: v for k, v in m.items() if k != "classification_report"},
        }

        save_confusion_matrix(
            y_true, y_pred,
            title=f"{model_type.upper()} — {level_name} noise",
            save_path=out_dir / f"confusion_{level_name}.png",
        )

    # ── Save + plot ───────────────────────────────────────────────────
    save_robustness_results(results, out_dir)
    logger.info("Robustness results saved → %s", out_dir)

    _log_robustness_table(results)


def _log_robustness_table(results: dict[str, Any]) -> None:
    """Print a compact summary table to the log."""
    logger.info("")
    logger.info("%-10s  %8s  %8s  %8s", "Level", "Acc", "MacroF1", "BalAcc")
    logger.info("-" * 42)
    for level, data in results["noise_levels"].items():
        m = data["metrics"]
        logger.info(
            "%-10s  %8.4f  %8.4f  %8.4f",
            level, m["accuracy"], m["macro_f1"], m["balanced_accuracy"],
        )


# ---------------------------------------------------------------------------
# Model loaders — return a unified predict_fn(waveforms) → (y_pred, y_prob)
# ---------------------------------------------------------------------------

def _load_predict_fn(
    model_type: str,
    checkpoint: Path,
    cfg: dict[str, Any],
    test_waveforms: np.ndarray,
) -> Callable[[np.ndarray], tuple[np.ndarray, Optional[np.ndarray]]]:

    model_type = model_type.lower()

    if model_type in ("cnn", "bilstm"):
        return _keras_predict_fn(checkpoint)

    if model_type == "hybrid":
        return _hybrid_predict_fn(checkpoint, cfg, test_waveforms)

    if model_type in ("lr", "rf", "svm"):
        return _sklearn_predict_fn(checkpoint)

    raise ValueError(
        f"Unknown model_type {model_type!r}. "
        "Choose: cnn, bilstm, hybrid, lr, rf, svm"
    )


def _keras_predict_fn(checkpoint: Path) -> Callable:
    """Predict function for single-input Keras models (CNN / BiLSTM)."""
    import tensorflow as tf  # type: ignore
    model = tf.keras.models.load_model(str(checkpoint))

    def predict(waveforms: np.ndarray):
        X = waveforms[:, :, np.newaxis].astype(np.float32)
        y_prob = model.predict(X, batch_size=256, verbose=0)
        y_pred = np.argmax(y_prob, axis=1).astype(np.int32)
        return y_pred, y_prob.astype(np.float32)

    return predict


def _hybrid_predict_fn(
    checkpoint: Path,
    cfg: dict[str, Any],
    reference_waveforms: np.ndarray,
) -> Callable:
    """Predict function for the two-input Hybrid model.

    NeuroChaos features are extracted from the noisy waveforms at each
    noise level. The NeuroChaos imputer is loaded from the original
    Hybrid training experiment so that no test-set statistics are used
    during robustness evaluation.
    """
    

    model = tf.keras.models.load_model(str(checkpoint))
    extractor = NeuroChaosExtractor(cfg)

    # The imputer must come from the original Hybrid training run.
    imputer_path = checkpoint.parent / "neurochaos_imputer.joblib"

    if not imputer_path.exists():
        raise FileNotFoundError(
            f"Train-fitted NeuroChaos imputer not found: {imputer_path}"
        )

    imputer = joblib.load(imputer_path)

    def predict(waveforms: np.ndarray):
        X_wave = waveforms[:, :, np.newaxis].astype(np.float32)

        # Extract NeuroChaos features from the CURRENT waveforms.
        # Therefore clean/low/medium/high noise levels each receive
        # their own feature representation.
        X_chaos_raw = extractor.extract_dataset(
            waveforms,
            show_progress=False,
        )
        X_chaos = imputer.transform(X_chaos_raw)

        y_prob = model.predict(
            [X_wave, X_chaos],
            batch_size=256,
            verbose=0,
        )
        y_pred = np.argmax(y_prob, axis=1).astype(np.int32)

        return y_pred, y_prob.astype(np.float32)

    return predict

    # Fit imputer on the clean reference waveforms (proxy for training dist)
    imputer = NeuroChaosImputer()
    X_ref_raw = extractor.extract_dataset(reference_waveforms, show_progress=False)
    imputer.fit(X_ref_raw)

    def predict(waveforms: np.ndarray):
        X_wave = waveforms[:, :, np.newaxis].astype(np.float32)
        X_chaos_raw = extractor.extract_dataset(waveforms, show_progress=False)
        X_chaos = imputer.transform(X_chaos_raw)
        y_prob = model.predict([X_wave, X_chaos], batch_size=256, verbose=0)
        y_pred = np.argmax(y_prob, axis=1).astype(np.int32)
        return y_pred, y_prob.astype(np.float32)

    return predict


def _sklearn_predict_fn(checkpoint: Path) -> Callable:
    """Predict function for scikit-learn pipelines (LR / RF / SVM)."""
    import joblib
    model = joblib.load(str(checkpoint))

    def predict(waveforms: np.ndarray):
        X = waveforms.astype(np.float32)
        y_pred = model.predict(X).astype(np.int32)
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X).astype(np.float32)
            except Exception:
                pass
        return y_pred, y_prob

    return predict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Noise robustness experiment for a trained model"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["cnn", "bilstm", "hybrid", "lr", "rf", "svm"],
        help="Model type to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to saved model checkpoint (.keras or .joblib)",
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    run(
        model_type=args.model,
        checkpoint=args.checkpoint,
        cfg_path=args.config,
    )


if __name__ == "__main__":
    main()
