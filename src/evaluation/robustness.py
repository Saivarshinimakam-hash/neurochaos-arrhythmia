"""
src/evaluation/robustness.py
-----------------------------
Controlled Gaussian noise robustness experiment.

Protocol
--------
The SAME test beats are evaluated at four noise levels:
  clean  — no noise added  (sigma_factor = 0.00)
  low    — sigma_factor = 0.05  (~26 dB SNR)
  medium — sigma_factor = 0.15  (~16 dB SNR)
  high   — sigma_factor = 0.30  (~10 dB SNR)

Noise is added as: x_noisy = x + N(0, sigma_factor × std(x))
per beat — the noise amplitude scales with each beat's own std so the
SNR is consistent regardless of beat amplitude.

This function is model-agnostic: it accepts a callable `predict_fn` that
maps waveforms (n, 288) → labels (n,) and optional probabilities (n, 5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src.data.segmentation import BeatDataset
from src.evaluation.metrics import compute_metrics, metrics_summary_line
from src.utils.logger import get_logger
from src.utils.seed import get_rng

logger = get_logger(__name__)

# Canonical noise level names and sigma factors
NOISE_LEVELS: dict[str, float] = {
    "clean":  0.00,
    "low":    0.05,
    "medium": 0.15,
    "high":   0.30,
}


def add_gaussian_noise(
    waveforms: np.ndarray,
    sigma_factor: float,
    seed: int = 42,
) -> np.ndarray:
    """Add per-beat Gaussian noise scaled by each beat's own std.

    Parameters
    ----------
    waveforms    : shape (n, window_size), already normalised
    sigma_factor : noise std = sigma_factor × per-beat std
    seed         : RNG seed for reproducibility

    Returns
    -------
    np.ndarray  same shape, float32
    """
    if sigma_factor == 0.0:
        return waveforms.copy()

    rng = get_rng(seed)
    waveforms = np.asarray(waveforms, dtype=np.float64)
    beat_stds = waveforms.std(axis=1, keepdims=True)   # (n, 1)
    noise = rng.normal(0.0, 1.0, size=waveforms.shape) * beat_stds * sigma_factor
    return (waveforms + noise).astype(np.float32)


def run_robustness_experiment(
    test_ds: BeatDataset,
    predict_fn: Callable[[np.ndarray], tuple[np.ndarray, Optional[np.ndarray]]],
    noise_levels: Optional[dict[str, float]] = None,
    model_name: str = "model",
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a model across all noise levels on the test set.

    Parameters
    ----------
    test_ds     : BeatDataset (test split — never seen during training)
    predict_fn  : callable(waveforms) → (y_pred, y_prob_or_None)
                  waveforms shape (n, window_size)
    noise_levels: dict mapping level name → sigma_factor
                  defaults to the four canonical levels
    model_name  : stored in output for identification
    seed        : RNG seed for noise generation

    Returns
    -------
    dict mapping noise_level_name → metrics_dict
    Also includes 'model_name' and 'n_test_beats' keys.
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    y_true = test_ds.labels
    results: dict[str, Any] = {
        "model_name": model_name,
        "n_test_beats": int(test_ds.n_beats),
        "noise_levels": {},
    }

    for level_name, sigma in noise_levels.items():
        logger.info("Robustness [%s] sigma=%.2f ...", level_name, sigma)
        noisy_waveforms = add_gaussian_noise(test_ds.waveforms, sigma, seed=seed)
        y_pred, y_prob = predict_fn(noisy_waveforms)
        m = compute_metrics(y_true, y_pred, y_prob, split=f"test_{level_name}")
        results["noise_levels"][level_name] = {
            "sigma_factor": sigma,
            "metrics": m,
        }
        logger.info("  %s", metrics_summary_line(m))

    return results


def save_robustness_results(
    results: dict[str, Any],
    out_dir: Path,
) -> None:
    """Save robustness results as JSON and a summary plot."""
    from src.evaluation.plots import _save_fig
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = results.get("model_name", "model")

    # JSON
    json_path = out_dir / f"robustness_{model_name}.json"
    with json_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Robustness JSON → %s", json_path)

    # Bar chart: macro_f1 at each noise level
    levels = list(results["noise_levels"].keys())
    f1_vals = [results["noise_levels"][l]["metrics"]["macro_f1"] for l in levels]

    fig, ax = plt.subplots(figsize=(7, 4))
    colours = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
    bars = ax.bar(levels, f1_vals, color=colours[:len(levels)], width=0.5, edgecolor="white")
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Noise Level"); ax.set_ylabel("Macro F1")
    ax.set_title(f"Noise Robustness — {model_name}", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, out_dir / f"robustness_{model_name}.png")
