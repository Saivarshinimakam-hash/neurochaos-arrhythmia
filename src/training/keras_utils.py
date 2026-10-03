"""
src/training/keras_utils.py
----------------------------
Shared Keras training utilities (callbacks, class weights, history saving).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_callbacks(
    out_dir: Path,
    model_name: str,
    patience: int = 10,
    monitor: str = "val_loss",
) -> list:
    """Return standard Keras callbacks for all DL training runs."""
    import tensorflow as tf  # type: ignore

    checkpoint_path = out_dir / f"{model_name}_best.keras"
    backup_dir = out_dir / f"{model_name}_backup"

    callbacks = [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(backup_dir),
            delete_checkpoint=False,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=max(3, patience // 3),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    return callbacks


def compute_class_weights(y_train: np.ndarray, n_classes: int = 5) -> dict[int, float]:
    """Inverse-frequency class weights to handle class imbalance."""
    from sklearn.utils.class_weight import compute_class_weight  # type: ignore
    classes = np.arange(n_classes)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def save_history(history_dict: dict, path: Path) -> None:
    """Save Keras history dict as JSON (convert numpy floats)."""
    serialisable = {k: [float(v) for v in vals] for k, vals in history_dict.items()}
    with path.open("w") as fh:
        json.dump(serialisable, fh, indent=2)
    logger.info("Training history saved → %s", path)


def keras_predict(
    model,
    X: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model.predict and return (y_pred_labels, y_prob)."""
    y_prob = model.predict(X, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_prob, axis=1).astype(np.int32)
    return y_pred, y_prob.astype(np.float32)
