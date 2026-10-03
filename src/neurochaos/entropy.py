"""
src/neurochaos/entropy.py
--------------------------
Entropy-based nonlinear complexity features for ECG beat windows.

Implemented features
--------------------
- Sample Entropy (SampEn)     — regularity / unpredictability
- Permutation Entropy (PE)    — ordinal-pattern complexity
- Approximate Entropy (ApEn)  — similar to SampEn, less bias-free

All functions accept a 1-D numpy array (single beat window) and return
a scalar float.  Failed calculations return NaN with a warning rather
than crashing — callers must handle NaN (see extractor.py).

Sample Entropy implementation note
------------------------------------
The canonical definition (Richman & Moorman 2000) is:

    SampEn(m, r, N) = -ln(A / B)

where
  B = number of template pairs of length m   with max|x_i - x_j| < r
  A = number of template pairs of length m+1 with max|x_i - x_j| < r
  r = r_factor × std(x, ddof=0)    (Chebyshev distance / ∞-norm)

The vectorised implementation uses:
  - ``np.lib.stride_tricks.sliding_window_view`` to build the (N-m, m)
    template matrix in one allocation — no Python loop over pairs.
  - Broadcasting to compute all pairwise Chebyshev distances in one
    NumPy expression.
  - Strict inequality ``< r`` (matches antropy's _numba_sampen exactly).

Numerical equivalence with antropy
------------------------------------
The vectorised code produces results within 1e-10 of antropy's Numba
implementation on all tested signals (verified in tests).  Any residual
difference is pure IEEE-754 floating-point rounding; both implementations
are mathematically identical.

Libraries used
--------------
- antropy  >= 0.1.9  retained as an optional cross-check (no longer in
  the hot path)

References
----------
- Richman & Moorman (2000). Physiological time-series analysis using
  approximate entropy and sample entropy. Am J Physiol Heart Circ Physiol.
- Bandt & Pompe (2002). Permutation entropy. PRL 88(17).
- Pincus (1991). Approximate entropy as a measure of system complexity.
  Proc Natl Acad Sci USA.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum signal length for meaningful entropy calculations
_MIN_LEN = 10


# ---------------------------------------------------------------------------
# Sample Entropy — vectorised NumPy implementation
# ---------------------------------------------------------------------------

def sample_entropy(
    signal: np.ndarray,
    m: int = 2,
    r_factor: float = 0.2,
) -> float:
    """Compute Sample Entropy (SampEn) using a vectorised NumPy implementation.

    Mathematical definition (Richman & Moorman 2000)
    ------------------------------------------------
    SampEn(m, r, N) = -ln(A / B)

    where:
      B = number of template pairs of length m   satisfying max|x_i-x_j| < r
      A = number of template pairs of length m+1 satisfying max|x_i-x_j| < r
      r = r_factor × std(signal, ddof=0)   (Chebyshev / ∞-norm tolerance)

    Self-matches (i == j) are excluded.

    Numerical equivalence with antropy
    ------------------------------------
    This implementation matches antropy.sample_entropy(..., metric="chebyshev")
    within 1e-10 on 288-sample ECG beats.  Differences are pure IEEE-754
    float64 rounding (different operation order; mathematically identical).

    Parameters
    ----------
    signal   : 1-D array
    m        : embedding dimension (default 2)
    r_factor : tolerance as fraction of signal std (default 0.2)

    Returns
    -------
    float — SampEn value, or NaN on failure.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()

    if len(signal) < _MIN_LEN:
        logger.debug("sample_entropy: signal too short (%d)", len(signal))
        return float("nan")
    if not np.isfinite(signal).all():
        logger.debug("sample_entropy: non-finite values in signal")
        return float("nan")

    r = r_factor * float(np.std(signal, ddof=0))
    if r == 0.0:
        logger.debug("sample_entropy: std=0 (flat signal)")
        return float("nan")

    return _sampen_vectorised(signal, m, r)


def _sampen_vectorised(signal: np.ndarray, m: int, r: float) -> float:
    """Vectorised SampEn — O(N²) in distance computations, but NumPy-native.

    Uses sliding_window_view to build all length-m templates in one
    allocation, then computes all pairwise Chebyshev distances via
    broadcasting.  This is mathematically identical to the Numba loop in
    antropy._numba_sampen.

    The implementation counts B (length-m matches) and A (length-(m+1)
    matches) in two passes, each using:

        matches[i, j] = True  iff  max_k |x[i+k] - x[j+k]| < r

    Self-matches (diagonal, i==j) are excluded from both counts.
    """
    N = len(signal)

    # ── Length-m templates ─────────────────────────────────────────────
    # shape: (N-m, m)  — each row is one template of length m
    templates_m  = np.lib.stride_tricks.sliding_window_view(signal, m)   # (N-m+1, m) but we use (N-m, m) below
    # sliding_window_view(signal, m) gives N-m+1 windows of length m.
    # For SampEn we need templates_m: first N-m  windows (indices 0..N-m-1)
    # and                  templates_m1: first N-m windows  with length m+1
    # where templates_m1 = sliding_window_view(signal, m+1)[0:N-m]

    # templates for B (length m): all N-m windows
    T_m  = np.lib.stride_tricks.sliding_window_view(signal, m  )[:-1]   # (N-m, m)
    # templates for A (length m+1): same starting positions, length m+1
    T_m1 = np.lib.stride_tricks.sliding_window_view(signal, m+1)  # (N-m, m+1)
    # Both have N-m rows so indices align: row i starts at position i.

    K = T_m.shape[0]   # = N - m

    # ── Pairwise Chebyshev distances ───────────────────────────────────
    # diff_m[i, j, k] = T_m[i, k] - T_m[j, k]
    # Chebyshev distance = max_k |diff_m[i,j,k]|
    # We use broadcasting:  T_m[:, np.newaxis, :] - T_m[np.newaxis, :, :]
    # For N-m = 286 that's (286, 286, 2) = ~163 k float64 values — fine.

    # Count B: length-m matches
    diff_m  = np.abs(T_m[:, np.newaxis, :]  - T_m[np.newaxis, :, :])    # (K,K,m)
    cheb_m  = diff_m.max(axis=2)                                         # (K,K)
    match_m = cheb_m < r                                                 # strict <
    np.fill_diagonal(match_m, False)                                     # exclude self
    B = int(match_m.sum())

    # Count A: length-(m+1) matches
    diff_m1  = np.abs(T_m1[:, np.newaxis, :] - T_m1[np.newaxis, :, :])  # (K,K,m+1)
    cheb_m1  = diff_m1.max(axis=2)                                       # (K,K)
    match_m1 = cheb_m1 < r                                               # strict <
    np.fill_diagonal(match_m1, False)                                    # exclude self
    A = int(match_m1.sum())

    # ── SampEn = -ln(A/B) ─────────────────────────────────────────────
    if B == 0:
        return 0.0          # antropy convention: 0/0 → 0
    if A == 0:
        return float("inf") # -ln(0) → inf; callers treat inf as failure
    try:
        return float(-math.log(A / B))
    except (ValueError, ZeroDivisionError):
        return float("nan")


# ---------------------------------------------------------------------------
# Permutation Entropy
# ---------------------------------------------------------------------------

def permutation_entropy(
    signal: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """Compute Permutation Entropy (PE) of a 1-D signal.

    Parameters
    ----------
    signal    : 1-D array
    order     : embedding dimension (default 3)
    delay     : time delay between samples (default 1)
    normalize : if True, divide by log(order!) to return value in [0,1]

    Returns
    -------
    float or NaN on failure.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    # Need enough points for at least 3 complete patterns
    min_required = order * delay + order
    if len(signal) < max(min_required, 10):
        logger.debug("permutation_entropy: signal too short (%d)", len(signal))
        return float("nan")
    if not np.isfinite(signal).all():
        logger.debug("permutation_entropy: non-finite values")
        return float("nan")

    try:
        import antropy as ant  # type: ignore
        return float(ant.perm_entropy(signal, order=order, delay=delay,
                                      normalize=normalize))
    except Exception as exc:
        logger.debug("antropy.perm_entropy failed (%s); using fallback", exc)
        return _permutation_entropy_fallback(signal, order, delay, normalize)


def _permutation_entropy_fallback(
    signal: np.ndarray,
    order: int,
    delay: int,
    normalize: bool,
) -> float:
    """Manual permutation entropy — ordinal pattern histogram."""
    N = len(signal)
    n_windows = N - (order - 1) * delay
    if n_windows <= 0:
        return float("nan")

    patterns: dict[tuple, int] = {}
    for i in range(n_windows):
        window = signal[i : i + order * delay : delay]
        pattern = tuple(np.argsort(window, kind="stable"))
        patterns[pattern] = patterns.get(pattern, 0) + 1

    total = sum(patterns.values())
    probs = np.array(list(patterns.values()), dtype=float) / total
    pe = float(-np.sum(probs * np.log(probs + 1e-12)))

    if normalize:
        max_pe = math.log(math.factorial(order))
        pe = pe / max_pe if max_pe > 0 else 0.0

    return pe


# ---------------------------------------------------------------------------
# Approximate Entropy
# ---------------------------------------------------------------------------

def approximate_entropy(
    signal: np.ndarray,
    m: int = 2,
    r_factor: float = 0.2,
) -> float:
    """Compute Approximate Entropy (ApEn) of a 1-D signal.

    Parameters
    ----------
    signal   : 1-D array
    m        : embedding dimension (default 2)
    r_factor : tolerance as fraction of signal std (default 0.2)

    Returns
    -------
    float or NaN on failure.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if len(signal) < _MIN_LEN:
        logger.debug("approximate_entropy: signal too short")
        return float("nan")
    if not np.isfinite(signal).all():
        logger.debug("approximate_entropy: non-finite values")
        return float("nan")

    r = r_factor * float(np.std(signal, ddof=0))
    if r == 0.0:
        logger.debug("approximate_entropy: std=0")
        return float("nan")

    try:
        import antropy as ant  # type: ignore
        return float(ant.app_entropy(signal, order=m))
    except Exception as exc:
        logger.debug("antropy.app_entropy failed (%s); using fallback", exc)
        return _apen_fallback(signal, m, r)


def _apen_fallback(
    signal: np.ndarray,
    m: int,
    r: float,
) -> float:
    """Pure-NumPy ApEn."""
    N = len(signal)

    def _phi(m_len: int) -> float:
        templates = np.array([signal[i : i + m_len] for i in range(N - m_len + 1)])
        counts = []
        for t in templates:
            diffs = np.max(np.abs(templates - t), axis=1)
            counts.append(np.sum(diffs <= r) / (N - m_len + 1))
        log_counts = [math.log(c) if c > 0 else 0.0 for c in counts]
        return sum(log_counts) / (N - m_len + 1)

    try:
        return float(_phi(m) - _phi(m + 1))
    except Exception:
        return float("nan")
