"""
src/neurochaos/phase_space.py
------------------------------
Phase-space reconstruction features for ECG beat windows.

Features
--------
- Mean phase-space velocity    — average Euclidean distance between
                                  consecutive trajectory points
- Phase-space trajectory length — total path length of the attractor
                                   trajectory (normalised by N)
- Convex hull volume estimate  — approximated by the determinant of the
                                  covariance matrix of the embedded vectors
                                  (log-scale for stability)

These capture the geometry of the reconstructed attractor, which
reflects nonlinear dynamics properties not visible in the time domain.

Reference
---------
- Takens (1981). Detecting strange attractors in turbulence.
  Lecture Notes in Mathematics 898.
- Abarbanel (1996). Analysis of Observed Chaotic Data. Springer.
"""

from __future__ import annotations

import logging

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

_MIN_LEN = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def phase_space_embedding(
    signal: np.ndarray,
    emb_dim: int = 3,
    time_delay: int = 1,
) -> np.ndarray | None:
    """Reconstruct phase space via time-delay embedding.

    Parameters
    ----------
    signal     : 1-D float array
    emb_dim    : embedding dimension m
    time_delay : time delay τ

    Returns
    -------
    np.ndarray, shape (M, emb_dim) where M = N - (m-1)*τ
    None if the signal is too short or contains non-finite values.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if not np.isfinite(signal).all():
        return None
    M = len(signal) - (emb_dim - 1) * time_delay
    if M < 2:
        return None

    vectors = np.array(
        [signal[i : i + emb_dim * time_delay : time_delay] for i in range(M)]
    )
    return vectors


def mean_phase_velocity(
    signal: np.ndarray,
    emb_dim: int = 3,
    time_delay: int = 1,
) -> float:
    """Mean Euclidean distance between consecutive phase-space points.

    Captures the average speed of the trajectory — higher for chaotic signals.

    Returns float or NaN on failure.
    """
    vectors = phase_space_embedding(signal, emb_dim, time_delay)
    if vectors is None or len(vectors) < 2:
        return float("nan")

    diffs = np.diff(vectors, axis=0)            # (M-1, emb_dim)
    dists = np.linalg.norm(diffs, axis=1)       # (M-1,)
    return float(np.mean(dists))


def trajectory_length(
    signal: np.ndarray,
    emb_dim: int = 3,
    time_delay: int = 1,
) -> float:
    """Normalised total path length of the phase-space trajectory.

    Total arc length divided by (M-1) so it is comparable across beats
    with different numbers of embedded points.

    Returns float or NaN on failure.
    """
    vectors = phase_space_embedding(signal, emb_dim, time_delay)
    if vectors is None or len(vectors) < 2:
        return float("nan")

    diffs = np.diff(vectors, axis=0)
    total = float(np.sum(np.linalg.norm(diffs, axis=1)))
    return total / (len(vectors) - 1)


def attractor_covariance_logdet(
    signal: np.ndarray,
    emb_dim: int = 3,
    time_delay: int = 1,
) -> float:
    """Log-determinant of the covariance matrix of phase-space vectors.

    A proxy for the 'volume' of the attractor.  More complex attractors
    occupy a larger volume in phase space.

    Returns float or NaN on failure.
    """
    vectors = phase_space_embedding(signal, emb_dim, time_delay)
    if vectors is None or len(vectors) < emb_dim + 1:
        return float("nan")

    try:
        cov = np.cov(vectors.T)                 # (emb_dim, emb_dim)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            # Singular covariance → flat attractor
            return float("nan")
        return float(logdet)
    except np.linalg.LinAlgError:
        return float("nan")


def compute_phase_space_features(
    signal: np.ndarray,
    emb_dim: int = 3,
    time_delay: int = 1,
) -> dict[str, float]:
    """Compute all phase-space features in one call.

    Returns
    -------
    dict with keys 'ps_mean_velocity', 'ps_traj_length', 'ps_logdet_cov'
    """
    return {
        "ps_mean_velocity":  mean_phase_velocity(signal, emb_dim, time_delay),
        "ps_traj_length":    trajectory_length(signal, emb_dim, time_delay),
        "ps_logdet_cov":     attractor_covariance_logdet(signal, emb_dim, time_delay),
    }
