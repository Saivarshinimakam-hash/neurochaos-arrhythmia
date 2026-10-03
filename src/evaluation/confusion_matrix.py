"""
src/evaluation/confusion_matrix.py
------------------------------------
Confusion matrix generation and saving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.evaluation.metrics import CLASS_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str | Path = "confusion_matrix.png",
    normalize: bool = True,
    figsize: tuple[int, int] = (7, 6),
) -> None:
    """Compute and save a confusion matrix figure.

    Parameters
    ----------
    y_true      : ground-truth integer labels
    y_pred      : predicted integer labels
    title       : plot title
    save_path   : output file path (.png)
    normalize   : if True, show row-normalised (recall) values
    figsize     : matplotlib figure size
    """
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1          # avoid div-by-zero for absent classes
        cm_display = cm.astype(float) / row_sums
        fmt = ".2f"
        cbar_label = "Recall (row-normalised)"
    else:
        cm_display = cm
        fmt = "d"
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": cbar_label},
        vmin=0.0,
        vmax=1.0 if normalize else None,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)

    # Annotate raw counts in parentheses when normalised
    if normalize:
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(
                    j + 0.5, i + 0.75,
                    f"({cm[i, j]})",
                    ha="center", va="center",
                    fontsize=7, color="gray",
                )

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", save_path)
