"""
Resume Bi-LSTM training from an existing best model.

This preserves the original Bi-LSTM experiment and creates a new
resume experiment directory.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.utils.logger import setup_logging, get_logger
from src.data.splitting import build_datasets
from src.models.bilstm import prepare_bilstm_inputs
from src.training.keras_utils import (
    make_callbacks,
    compute_class_weights,
    save_history,
    keras_predict,
)
from src.evaluation.metrics import compute_metrics, metrics_summary_line
from src.evaluation.confusion_matrix import save_confusion_matrix
from src.evaluation.plots import (
    save_roc_curves,
    save_pr_curves,
    save_training_history,
)

logger = get_logger(__name__)


def run() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])

    original_model = (
        Path(cfg["output"]["experiments_dir"])
        / "20261003_013425_bilstm"
        / "bilstm_best.keras"
    )

    if not original_model.exists():
        raise FileNotFoundError(
            f"Saved Bi-LSTM model not found: {original_model}"
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(cfg["output"]["experiments_dir"])
        / f"{timestamp}_bilstm_resume"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        level="INFO",
        log_file=Path(cfg["output"]["logs_dir"])
        / f"{timestamp}_bilstm_resume.log",
    )

    logger.info("=== Bi-LSTM Resume Experiment ===")
    logger.info("Original model: %s", original_model)
    logger.info("New output directory: %s", out_dir)

    # ---------------------------------------------------------
    # Load the same datasets
    # ---------------------------------------------------------
    logger.info("Building datasets ...")

    train_ds, val_ds, test_ds = build_datasets(
        cfg,
        download=True,
        normalize=True,
        save_manifest=False,
    )

    logger.info(
        "Train %d  Val %d  Test %d beats",
        train_ds.n_beats,
        val_ds.n_beats,
        test_ds.n_beats,
    )

    X_train = prepare_bilstm_inputs(train_ds.waveforms)
    X_val = prepare_bilstm_inputs(val_ds.waveforms)
    X_test = prepare_bilstm_inputs(test_ds.waveforms)

    y_train = train_ds.labels
    y_val = val_ds.labels
    y_test = test_ds.labels

    # ---------------------------------------------------------
    # Load previously saved best model
    # ---------------------------------------------------------
    logger.info("Loading saved Bi-LSTM model ...")

    model = tf.keras.models.load_model(original_model)

    logger.info("Saved model loaded successfully.")
    model.summary(print_fn=logger.info)

    # ---------------------------------------------------------
    # Class weights
    # ---------------------------------------------------------
    class_weights = compute_class_weights(y_train)
    logger.info("Class weights: %s", class_weights)

    # ---------------------------------------------------------
    # Continue training from saved weights
    # ---------------------------------------------------------
    bl_cfg = cfg["bilstm"]

    callbacks = make_callbacks(
        out_dir,
        model_name="bilstm_resume",
        patience=bl_cfg["early_stopping_patience"],
    )

    logger.info("Continuing training from saved model ...")

    t0 = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=bl_cfg["epochs"],
        batch_size=bl_cfg["batch_size"],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    train_secs = time.time() - t0

    logger.info(
        "Resume training completed in %.1f s",
        train_secs,
    )

    # ---------------------------------------------------------
    # Save history
    # ---------------------------------------------------------
    save_history(
        history.history,
        out_dir / "history.json",
    )

    save_training_history(
        history.history,
        title="Bi-LSTM Resume Training History",
        save_path=out_dir / "training_history.png",
    )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------
    for split_name, X, y in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:

        logger.info("Evaluating %s split ...", split_name)

        y_pred, y_prob = keras_predict(model, X)

        metrics = compute_metrics(
            y,
            y_pred,
            y_prob,
            split=split_name,
        )

        logger.info(
            metrics_summary_line(metrics)
        )

        logger.info(
            "\n%s",
            metrics["classification_report"],
        )

        metrics_path = (
            out_dir / f"metrics_{split_name}.json"
        )

        with metrics_path.open("w") as fh:
            json.dump(
                {
                    k: v
                    for k, v in metrics.items()
                    if k != "classification_report"
                },
                fh,
                indent=2,
            )

        save_confusion_matrix(
            y,
            y_pred,
            title=f"Bi-LSTM Resume — "
                  f"{split_name.capitalize()} Confusion Matrix",
            save_path=(
                out_dir / f"confusion_{split_name}.png"
            ),
        )

        save_roc_curves(
            y,
            y_prob,
            title=f"Bi-LSTM Resume — "
                  f"{split_name.capitalize()} ROC Curves",
            save_path=(
                out_dir / f"roc_{split_name}.png"
            ),
        )

        save_pr_curves(
            y,
            y_prob,
            title=f"Bi-LSTM Resume — "
                  f"{split_name.capitalize()} PR Curves",
            save_path=(
                out_dir / f"pr_{split_name}.png"
            ),
        )

    # Save final model from the continuation run
    final_model_path = out_dir / "bilstm_resume_final.keras"
    model.save(final_model_path)

    logger.info(
        "Final resumed model saved to %s",
        final_model_path,
    )

    logger.info(
        "All outputs saved to %s",
        out_dir,
    )


if __name__ == "__main__":
    run()