"""
src/evaluation/plots.py
------------------------
ROC curves, Precision-Recall curves, and feature importance plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

from src.evaluation.metrics import CLASS_NAMES, N_CLASSES, _one_hot
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Colour palette — one per class
_COLOURS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]


def save_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curves",
    save_path: str | Path = "roc_curves.png",
    figsize: tuple[int, int] = (7, 6),
) -> None:
    """One-vs-rest ROC curves for all 5 AAMI classes + macro average."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.shape[1] != N_CLASSES:
        logger.warning("save_roc_curves: expected %d columns, got %d", N_CLASSES, y_prob.shape[1])
        return

    y_bin = _one_hot(y_true, N_CLASSES)

    fig, ax = plt.subplots(figsize=figsize)
    all_fpr = np.linspace(0, 1, 200)
    interp_tprs = []

    for i, (cls, col) in enumerate(zip(CLASS_NAMES, _COLOURS)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=1.5,
                label=f"{cls}  (AUC={roc_auc:.3f})")
        interp_tprs.append(np.interp(all_fpr, fpr, tpr))

    if interp_tprs:
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, "k--", lw=2,
                label=f"Macro avg (AUC={mean_auc:.3f})")

    ax.plot([0, 1], [0, 1], "gray", lw=1, linestyle=":")
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    _save_fig(fig, save_path)


def save_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Precision-Recall Curves",
    save_path: str | Path = "pr_curves.png",
    figsize: tuple[int, int] = (7, 6),
) -> None:
    """One-vs-rest PR curves for all 5 AAMI classes + macro average."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.shape[1] != N_CLASSES:
        return

    y_bin = _one_hot(y_true, N_CLASSES)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, _COLOURS)):
        if y_bin[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax.plot(rec, prec, color=col, lw=1.5,
                label=f"{cls}  (AP={ap:.3f})")

    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    _save_fig(fig, save_path)


def save_neurochaos_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    title: str = "NeuroChaos Feature Importances",
    save_path: str | Path = "feature_importance.png",
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """Horizontal bar chart of feature importances (e.g. from RandomForest)."""
    importances = np.asarray(importances, dtype=float)
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        [feature_names[i] for i in order],
        importances[order],
        color="#2196F3", edgecolor="white", height=0.7,
    )
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.axvline(0, color="black", lw=0.8)
    plt.tight_layout()
    _save_fig(fig, save_path)


def save_training_history(
    history: dict,
    title: str = "Training History",
    save_path: str | Path = "training_history.png",
    figsize: tuple[int, int] = (10, 4),
) -> None:
    """Plot loss and accuracy curves from a Keras history dict."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, metric, ylabel in zip(
        axes,
        [("loss", "val_loss"), ("accuracy", "val_accuracy")],
        ["Loss", "Accuracy"],
    ):
        train_key, val_key = metric
        if train_key in history:
            ax.plot(history[train_key], label="Train", color="#2196F3")
        if val_key in history:
            ax.plot(history[val_key],   label="Val",   color="#F44336")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    _save_fig(fig, save_path)


def save_model_comparison_bar(
    model_names: list[str],
    metrics_list: list[dict],
    metric_key: str = "macro_f1",
    split: str = "test",
    title: str = "Model Comparison",
    save_path: str | Path = "model_comparison.png",
    figsize: tuple[int, int] = (9, 5),
) -> None:
    """Grouped bar chart comparing models across a chosen metric."""
    values = [m.get(metric_key, 0.0) for m in metrics_list]
    colours = plt.cm.Set2(np.linspace(0, 1, len(model_names)))  # type: ignore

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(model_names, values, color=colours, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric_key.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"{title} [{split}]", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", save_path)
