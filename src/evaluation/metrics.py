"""
src/evaluation/metrics.py
--------------------------
Common evaluation protocol shared by ALL models.

Every model experiment calls ``compute_metrics`` with its predictions and
receives the same dictionary structure — ensuring results are directly
comparable across Classical ML, CNN, BiLSTM, NeuroChaos, and Hybrid models.

Metrics computed
----------------
- Accuracy
- Precision (macro, weighted, per-class)
- Recall    (macro, weighted, per-class)
- F1 score  (macro, weighted, per-class)
- Balanced accuracy
- ROC-AUC   (macro OvR, when probabilities are available)
- Average precision (macro, when probabilities are available)
- Matthews Correlation Coefficient

All metrics are returned as a plain dict of Python floats so they
serialise directly to JSON.

Class names follow the AAMI 5-class scheme: N, S, V, F, Q (indices 0–4).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES: list[str] = ["N", "S", "V", "F", "Q"]
N_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    split: str = "test",
) -> dict:
    """Compute the full evaluation metric set for one model / one split.

    Parameters
    ----------
    y_true : shape (n,)   — integer ground-truth labels (0–4)
    y_pred : shape (n,)   — integer predicted labels
    y_prob : shape (n, 5) — predicted class probabilities (optional)
                            Required for ROC-AUC and average precision.
    split  : "train" | "val" | "test"  — stored in the output dict for
             easy identification when combining results across splits.

    Returns
    -------
    dict with keys:
        split, n_samples,
        accuracy, balanced_accuracy, mcc,
        macro_precision, macro_recall, macro_f1,
        weighted_precision, weighted_recall, weighted_f1,
        per_class  (dict N/S/V/F/Q → {precision, recall, f1, support}),
        roc_auc_macro (float or null),
        avg_precision_macro (float or null),
        classification_report (str)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Labels present in y_true (needed for zero_division safety)
    labels = list(range(N_CLASSES))

    result: dict = {"split": split, "n_samples": int(len(y_true))}

    # ── Scalar metrics ────────────────────────────────────────────────
    result["accuracy"]          = float(accuracy_score(y_true, y_pred))
    result["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    result["mcc"]               = float(matthews_corrcoef(y_true, y_pred))

    result["macro_precision"]   = float(precision_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0))
    result["macro_recall"]      = float(recall_score(   y_true, y_pred, average="macro",    labels=labels, zero_division=0))
    result["macro_f1"]          = float(f1_score(       y_true, y_pred, average="macro",    labels=labels, zero_division=0))

    result["weighted_precision"] = float(precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    result["weighted_recall"]    = float(recall_score(   y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    result["weighted_f1"]        = float(f1_score(       y_true, y_pred, average="weighted", labels=labels, zero_division=0))

    # ── Per-class metrics ─────────────────────────────────────────────
    prec_per  = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    rec_per   = recall_score(   y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1_per    = f1_score(       y_true, y_pred, average=None, labels=labels, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred, labels=labels)
    support   = cm.sum(axis=1)   # actual counts per class

    result["per_class"] = {
        cls: {
            "precision": float(prec_per[i]),
            "recall":    float(rec_per[i]),
            "f1":        float(f1_per[i]),
            "support":   int(support[i]),
        }
        for i, cls in enumerate(CLASS_NAMES)
    }

    # ── Probability-based metrics ────────────────────────────────────
    result["roc_auc_macro"]       = None
    result["avg_precision_macro"] = None

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        # Only compute if at least 2 classes are present in y_true
        unique_classes = np.unique(y_true)
        if len(unique_classes) >= 2:
            try:
                # Pad probability matrix to 5 columns if model outputs fewer
                y_prob_full = _ensure_5_cols(y_prob, unique_classes)
                result["roc_auc_macro"] = float(
                    roc_auc_score(y_true, y_prob_full, multi_class="ovr",
                                  average="macro", labels=labels)
                )
            except Exception as exc:
                logger.warning("ROC-AUC computation failed: %s", exc)

            try:
                y_prob_full = _ensure_5_cols(y_prob, unique_classes)
                result["avg_precision_macro"] = float(
                    average_precision_score(
                        _one_hot(y_true, N_CLASSES), y_prob_full,
                        average="macro",
                    )
                )
            except Exception as exc:
                logger.warning("Average precision computation failed: %s", exc)

    # ── Text report ──────────────────────────────────────────────────
    result["classification_report"] = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    return result


def metrics_summary_line(metrics: dict) -> str:
    """One-line summary string for logging."""
    roc = f"  roc={metrics['roc_auc_macro']:.4f}" if metrics.get("roc_auc_macro") else ""
    return (
        f"[{metrics['split']}] n={metrics['n_samples']}  "
        f"acc={metrics['accuracy']:.4f}  "
        f"bal_acc={metrics['balanced_accuracy']:.4f}  "
        f"macro_f1={metrics['macro_f1']:.4f}  "
        f"wtd_f1={metrics['weighted_f1']:.4f}"
        f"{roc}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    oh = np.zeros((len(y), n_classes), dtype=float)
    oh[np.arange(len(y)), y] = 1.0
    return oh


def _ensure_5_cols(y_prob: np.ndarray, present_classes: np.ndarray) -> np.ndarray:
    """If y_prob has fewer than 5 columns, expand to 5 (fill absent classes with 0)."""
    if y_prob.shape[1] == N_CLASSES:
        return y_prob
    out = np.zeros((len(y_prob), N_CLASSES), dtype=float)
    for col_idx, cls in enumerate(present_classes):
        if col_idx < y_prob.shape[1]:
            out[:, cls] = y_prob[:, col_idx]
    return out
