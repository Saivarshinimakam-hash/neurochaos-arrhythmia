"""
src/neurochaos/recurrence.py
-----------------------------
Recurrence Quantification Analysis (RQA) features for ECG beats.

Implemented features
--------------------
- Recurrence Rate  (RR)  — fraction of recurrent states
- Determinism      (DET) — fraction of recurrences in diagonal lines
- Laminarity       (LAM) — fraction of recurrences in vertical lines

RQA characterises the dynamical structure of a time series through its
recurrence matrix R[i,j] = Θ(ε − ‖x_i − x_j‖) where x_i are
phase-space reconstruction vectors.

Implementation note
-------------------
A full RQA library (e.g. pyRQA) is not required as a hard dependency.
We implement the core metrics from scratch using NumPy; this is fast
enough for 288-sample beat windows and keeps the dependency tree clean.

Reference
---------
- Zbilut & Webber (1992). Embeddings and delays as derived from
  quantification of recurrence plots. Physics Letters A.
- Marwan et al. (2007). Recurrence plots for the analysis of complex
  systems. Physics Reports 438(5-6), 237-329.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

_MIN_LEN = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recurrence_matrix(
    signal: np.ndarray,
    dimension: int = 3,
    time_delay: int = 1,
    threshold: float = 0.1,
) -> Optional[np.ndarray]:
    """Build the binary recurrence matrix for a 1-D signal.

    Uses time-delay embedding to reconstruct the phase space, then
    thresholds pairwise Chebyshev distances.

    Parameters
    ----------
    signal     : 1-D array
    dimension  : embedding dimension
    time_delay : time delay τ between components
    threshold  : ε as a fraction of the max pairwise distance
                 (adaptive threshold — scale-invariant)

    Returns
    -------
    np.ndarray, shape (M, M), dtype bool
        Recurrence matrix, or None if computation failed.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if len(signal) < _MIN_LEN:
        return None
    if not np.isfinite(signal).all():
        return None

    # Phase-space reconstruction via time-delay embedding
    M = len(signal) - (dimension - 1) * time_delay
    if M <= 1:
        return None

    vectors = np.array(
        [signal[i : i + dimension * time_delay : time_delay] for i in range(M)]
    )  # shape (M, dimension)

    # Pairwise Chebyshev distances
    # vectorised: |x_i - x_j|_inf
    diff = vectors[:, np.newaxis, :] - vectors[np.newaxis, :, :]  # (M,M,dim)
    dist = np.max(np.abs(diff), axis=2)                           # (M,M)

    max_dist = dist.max()
    if max_dist == 0:
        return None

    eps = threshold * max_dist
    R = dist <= eps     # bool (M,M)
    return R


def recurrence_rate(
    signal: np.ndarray,
    dimension: int = 3,
    time_delay: int = 1,
    threshold: float = 0.1,
) -> float:
    """Recurrence Rate: fraction of recurrent points in the RP (excluding diagonal).

    Returns float or NaN on failure.
    """
    R = recurrence_matrix(signal, dimension, time_delay, threshold)
    if R is None:
        return float("nan")

    M = R.shape[0]
    # Exclude main diagonal (self-recurrences)
    np.fill_diagonal(R, False)
    rr = float(np.sum(R)) / (M * (M - 1)) if M > 1 else float("nan")
    return rr


def determinism(
    signal: np.ndarray,
    dimension: int = 3,
    time_delay: int = 1,
    threshold: float = 0.1,
    min_diagonal_line: int = 2,
) -> float:
    """Determinism: fraction of recurrent points forming diagonal lines.

    Diagonal lines indicate deterministic (predictable) dynamics.

    Parameters
    ----------
    min_diagonal_line : minimum line length counted as a diagonal line (≥2)

    Returns float or NaN on failure.
    """
    R = recurrence_matrix(signal, dimension, time_delay, threshold)
    if R is None:
        return float("nan")

    M = R.shape[0]
    np.fill_diagonal(R, False)

    total_recurrent = int(np.sum(R))
    if total_recurrent == 0:
        return 0.0

    # Count points on diagonal lines of length ≥ min_diagonal_line
    diag_points = _count_diagonal_line_points(R, min_diagonal_line)
    return float(diag_points) / float(total_recurrent)


def laminarity(
    signal: np.ndarray,
    dimension: int = 3,
    time_delay: int = 1,
    threshold: float = 0.1,
    min_vertical_line: int = 2,
) -> float:
    """Laminarity: fraction of recurrent points forming vertical lines.

    Vertical lines indicate intermittent, laminar states.

    Parameters
    ----------
    min_vertical_line : minimum vertical line length (≥2)

    Returns float or NaN on failure.
    """
    R = recurrence_matrix(signal, dimension, time_delay, threshold)
    if R is None:
        return float("nan")

    np.fill_diagonal(R, False)

    total_recurrent = int(np.sum(R))
    if total_recurrent == 0:
        return 0.0

    lam_points = _count_vertical_line_points(R, min_vertical_line)
    return float(lam_points) / float(total_recurrent)


def compute_rqa_features(
    signal: np.ndarray,
    dimension: int = 3,
    time_delay: int = 1,
    threshold: float = 0.1,
) -> dict[str, float]:
    """Compute all three RQA features in one pass (shares one matrix build).

    Returns
    -------
    dict with keys 'rr', 'det', 'lam' — NaN on any failure.
    """
    R = recurrence_matrix(signal, dimension, time_delay, threshold)
    if R is None:
        return {"rr": float("nan"), "det": float("nan"), "lam": float("nan")}

    M = R.shape[0]
    np.fill_diagonal(R, False)
    total_recurrent = int(np.sum(R))

    rr_val = float(total_recurrent) / (M * (M - 1)) if M > 1 else float("nan")

    if total_recurrent == 0:
        return {"rr": rr_val, "det": 0.0, "lam": 0.0}

    diag_pts = _count_diagonal_line_points(R, min_length=2)
    vert_pts  = _count_vertical_line_points(R, min_length=2)

    det_val = float(diag_pts) / float(total_recurrent)
    lam_val = float(vert_pts) / float(total_recurrent)

    return {"rr": rr_val, "det": det_val, "lam": lam_val}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_diagonal_line_points(R: np.ndarray, min_length: int) -> int:
    """Count points that belong to diagonal lines of length ≥ min_length."""
    M = R.shape[0]
    total = 0
    # Iterate over all diagonals (offset k ≠ 0 because main diagonal excluded)
    for k in range(1 - M, M):
        if k == 0:
            continue
        diag = np.diag(R, k)
        total += _line_points(diag, min_length)
    return total


def _count_vertical_line_points(R: np.ndarray, min_length: int) -> int:
    """Count points that belong to vertical lines of length ≥ min_length."""
    total = 0
    for col in range(R.shape[1]):
        total += _line_points(R[:, col], min_length)
    return total


def _line_points(arr: np.ndarray, min_length: int) -> int:
    """Count elements that are part of runs of True of length ≥ min_length."""
    total = 0
    run = 0
    for v in arr:
        if v:
            run += 1
        else:
            if run >= min_length:
                total += run
            run = 0
    if run >= min_length:
        total += run
    return total
