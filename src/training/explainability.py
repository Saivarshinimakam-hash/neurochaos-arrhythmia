"""
src/training/explainability.py
-------------------------------
Phase 10 — Explainability and feature analysis.

Analyses performed
------------------
1. NeuroChaos feature importance via Random Forest
   - Fit an RF on extracted chaos features (train set).
   - Extract and plot ``feature_importances_``.

2. Per-class NeuroChaos feature distributions
   - Box-plots of each feature grouped by AAMI class.
   - Reveals which features separate classes most clearly.

3. Feature correlation heatmap
   - Pearson correlation matrix across the 13 chaos features.
   - Identifies redundant / collinear features.

4. UMAP / t-SNE 2-D projection of chaos features
   - Visual cluster separation of AAMI classes.
   - Tries UMAP first; falls back to t-SNE if umap-learn is absent.

5. Class-conditional mean waveforms
   - Average normalised waveform per AAMI class.
   - Shows the morphological difference the model must discriminate.

All outputs are saved to results/experiments/<timestamp>_explainability/.

Usage
-----
    python -m src.training.explainability

IMPORTANT
---------
This script uses TEST-set beats for visualisation only — it does NOT
fit any model parameters on the test set.  The RF for importance ranking
is fitted on training beats (leakage-safe).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.utils.logger import setup_logging, get_logger
from src.data.splitting import build_datasets
from src.evaluation.metrics import CLASS_NAMES
from src.evaluation.plots import save_neurochaos_feature_importance, _save_fig
from src.neurochaos.extractor import NeuroChaosExtractor, NeuroChaosImputer, FEATURE_NAMES

logger = get_logger(__name__)


def run(cfg_path: Path | None = None) -> None:
    cfg = load_config(cfg_path)
    set_global_seed(cfg["seed"])

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg["output"]["experiments_dir"]) / f"{timestamp}_explainability"
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        level="INFO",
        log_file=Path(cfg["output"]["logs_dir"]) / f"{timestamp}_explainability.log",
    )
    logger.info("=== Phase 10: Explainability & Feature Analysis ===")
    logger.info("Output: %s", out_dir)

    # ── Load data ──────────────────────────────────────────────────────
    logger.info("Building datasets ...")
    train_ds, val_ds, test_ds = build_datasets(
        cfg, download=True, normalize=True, save_manifest=False
    )

    # ── Extract NeuroChaos features ────────────────────────────────────
    logger.info("Extracting NeuroChaos features ...")
    extractor = NeuroChaosExtractor(cfg)
    X_train_raw = extractor.extract_dataset(train_ds.waveforms, show_progress=True)
    X_test_raw  = extractor.extract_dataset(test_ds.waveforms,  show_progress=True)

    imputer = NeuroChaosImputer()
    X_train = imputer.fit_transform(X_train_raw)   # fit on train only
    X_test  = imputer.transform(X_test_raw)

    y_train = train_ds.labels
    y_test  = test_ds.labels

    # ── Analysis 1: RF feature importances ────────────────────────────
    logger.info("[1/5] Random Forest feature importances ...")
    _rf_feature_importance(X_train, y_train, out_dir, cfg)

    # ── Analysis 2: Per-class feature distributions ────────────────────
    logger.info("[2/5] Per-class feature distributions ...")
    _per_class_distributions(X_test, y_test, out_dir)

    # ── Analysis 3: Feature correlation heatmap ────────────────────────
    logger.info("[3/5] Feature correlation heatmap ...")
    _feature_correlation(X_train, out_dir)

    # ── Analysis 4: 2-D projection ────────────────────────────────────
    logger.info("[4/5] 2-D projection (UMAP / t-SNE) ...")
    _dimensionality_reduction(X_test, y_test, out_dir, cfg)

    # ── Analysis 5: Class-conditional mean waveforms ──────────────────
    logger.info("[5/5] Class-conditional mean waveforms ...")
    _mean_waveforms(test_ds.waveforms, y_test, out_dir)

    logger.info("All explainability outputs saved to %s", out_dir)


# ---------------------------------------------------------------------------
# Individual analyses
# ---------------------------------------------------------------------------

def _rf_feature_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    out_dir: Path,
    cfg: dict[str, Any],
) -> None:
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=cfg["seed"],
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_

    # Save raw values
    importance_dict = {name: float(imp)
                       for name, imp in zip(FEATURE_NAMES, importances)}
    with (out_dir / "feature_importances_rf.json").open("w") as fh:
        json.dump(importance_dict, fh, indent=2)

    save_neurochaos_feature_importance(
        feature_names=FEATURE_NAMES,
        importances=importances,
        title="NeuroChaos Feature Importances (Random Forest)",
        save_path=out_dir / "feature_importances_rf.png",
    )
    logger.info("  Top 3: %s",
                sorted(importance_dict, key=importance_dict.get, reverse=True)[:3])


def _per_class_distributions(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
) -> None:
    n_features = len(FEATURE_NAMES)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()
    colours = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        ax = axes[feat_idx]
        data_per_class = []
        labels_present = []
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            mask = y == cls_idx
            if mask.sum() > 0:
                data_per_class.append(X[mask, feat_idx])
                labels_present.append(cls_name)

        bp = ax.boxplot(
            data_per_class,
            patch_artist=True,
            notch=False,
            showfliers=False,
        )
        for patch, col in zip(bp["boxes"], colours[:len(data_per_class)]):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)

        ax.set_xticklabels(labels_present, fontsize=9)
        ax.set_title(feat_name, fontsize=9, pad=4)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for ax in axes[n_features:]:
        ax.set_visible(False)

    fig.suptitle("NeuroChaos Feature Distributions by AAMI Class (Test Set)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _save_fig(fig, out_dir / "feature_distributions_per_class.png")


def _feature_correlation(X: np.ndarray, out_dir: Path) -> None:
    import seaborn as sns  # type: ignore

    corr = np.corrcoef(X.T)  # (13, 13)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        xticklabels=FEATURE_NAMES,
        yticklabels=FEATURE_NAMES,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        annot_kws={"size": 7},
    )
    ax.set_title("NeuroChaos Feature Pearson Correlation (Train Set)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    _save_fig(fig, out_dir / "feature_correlation_heatmap.png")

    # Identify highly correlated pairs (|r| > 0.9)
    high_corr = []
    for i in range(len(FEATURE_NAMES)):
        for j in range(i + 1, len(FEATURE_NAMES)):
            if abs(corr[i, j]) > 0.9:
                high_corr.append({
                    "feature_a": FEATURE_NAMES[i],
                    "feature_b": FEATURE_NAMES[j],
                    "pearson_r": round(float(corr[i, j]), 4),
                })
    if high_corr:
        logger.warning("  Highly correlated feature pairs (|r|>0.9): %s", high_corr)
    else:
        logger.info("  No highly correlated pairs found (|r|>0.9)")

    with (out_dir / "high_correlation_pairs.json").open("w") as fh:
        json.dump(high_corr, fh, indent=2)


def _dimensionality_reduction(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    cfg: dict[str, Any],
) -> None:
    colours = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    # Subsample if test set is large (projection is slow on >5k points)
    MAX_POINTS = 3000
    if len(X) > MAX_POINTS:
        rng = np.random.default_rng(cfg["seed"])
        idx = rng.choice(len(X), MAX_POINTS, replace=False)
        X_sub, y_sub = X[idx], y[idx]
        logger.info("  Subsampled to %d points for 2-D projection", MAX_POINTS)
    else:
        X_sub, y_sub = X, y

    embedding_2d = None
    method_name  = None

    # Try UMAP first
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=2, random_state=cfg["seed"], n_jobs=1)
        embedding_2d = reducer.fit_transform(X_sub)
        method_name  = "UMAP"
        logger.info("  UMAP projection computed")
    except ImportError:
        logger.info("  umap-learn not installed — falling back to t-SNE")

    # Fallback: t-SNE
    if embedding_2d is None:
        try:
            from sklearn.manifold import TSNE  # type: ignore
            tsne = TSNE(
                n_components=2,
                perplexity=min(30, len(X_sub) - 1),
                random_state=cfg["seed"],
                n_iter=1000,
            )
            embedding_2d = tsne.fit_transform(X_sub)
            method_name  = "t-SNE"
            logger.info("  t-SNE projection computed")
        except Exception as exc:
            logger.warning("  2-D projection failed: %s", exc)
            return

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls_idx, (cls_name, col) in enumerate(zip(CLASS_NAMES, colours)):
        mask = y_sub == cls_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=col, s=12, alpha=0.6, label=cls_name, edgecolors="none",
        )
    ax.set_title(f"NeuroChaos Features — {method_name} (Test Set)", fontsize=12)
    ax.legend(title="AAMI Class", fontsize=9, markerscale=2)
    ax.set_xlabel(f"{method_name} dim 1"); ax.set_ylabel(f"{method_name} dim 2")
    plt.tight_layout()
    _save_fig(fig, out_dir / f"projection_{method_name.lower()}.png")


def _mean_waveforms(
    waveforms: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
) -> None:
    n_samples = waveforms.shape[1]
    t = np.arange(n_samples)
    colours = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(16, 3), sharey=True)

    for cls_idx, (cls_name, col, ax) in enumerate(zip(CLASS_NAMES, colours, axes)):
        mask = y == cls_idx
        n = int(mask.sum())
        ax.set_title(f"{cls_name}\n(n={n})", fontsize=10)
        if n == 0:
            ax.text(0.5, 0.5, "no samples", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            continue

        beats = waveforms[mask].astype(np.float64)
        mean  = beats.mean(axis=0)
        std   = beats.std(axis=0)

        ax.plot(t, mean, color=col, lw=1.5)
        ax.fill_between(t, mean - std, mean + std,
                        color=col, alpha=0.2, label="±1 std")
        ax.axvline(108, color="gray", lw=0.8, linestyle="--",
                   label="R-peak (t=108)")
        ax.set_xlabel("Sample")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Normalised amplitude")
    fig.suptitle("Mean ± Std Waveform per AAMI Class (Test Set)", fontsize=12)
    plt.tight_layout()
    _save_fig(fig, out_dir / "mean_waveforms_per_class.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeuroChaos explainability and feature analysis"
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    run(cfg_path=args.config)


if __name__ == "__main__":
    main()
