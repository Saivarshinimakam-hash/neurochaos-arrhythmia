"""
src/neurochaos/fractal.py
--------------------------
Fractal dimension and complexity features for ECG beat windows.

Implemented features
--------------------
- Detrended Fluctuation Analysis (DFA) alpha exponent
  α < 0.5 → anti-correlated; α ≈ 0.5 → uncorrelated; α ≈ 1 → 1/f noise;
  α > 1   → non-stationary.
- Hurst exponent (via nolds.hurst_rs or DFA fallback)
- Higuchi Fractal Dimension (HFD) — purely numpy, no external dependency

Libraries
---------
- nolds >= 0.5.2 for DFA and Hurst

References
----------
- Peng et al. (1994). Mosaic organization of DNA nucleotides. PRE 49(2).
- Hurst (1951). Long-term storage capacity of reservoirs.
  Trans ASCE 116, 770-799.
- Higuchi (1988). Approach to an irregular time series on the basis of
  the fractal theory. Physica D 31, 277-283.
"""

from __future__ import annotations

import logging

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

_MIN_LEN_DFA  = 30
_MIN_LEN_HURST = 20
_MIN_LEN_HFD  = 20


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dfa_alpha(
    signal: np.ndarray,
    order: int = 1,
) -> float:
    """Detrended Fluctuation Analysis scaling exponent α.

    Parameters
    ----------
    signal : 1-D float array
    order  : polynomial order for detrending (1 = linear)

    Returns float or NaN on failure.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if len(signal) < _MIN_LEN_DFA:
        logger.debug("dfa_alpha: signal too short (%d)", len(signal))
        return float("nan")
    if not np.isfinite(signal).all():
        return float("nan")
    if np.std(signal) < 1e-10:
        return float("nan")

    try:
        import nolds  # type: ignore
        # fit_exp="poly" uses np.polyfit for the log-log line fit instead of
        # RANSAC.  This is mathematically identical (both estimate the slope of
        # log(F(n)) vs log(n)) but avoids sklearn's RANSACRegressor, which
        # triggers "R^2 score is not well-defined with less than two samples"
        # warnings on short DFA window sequences.
        alpha = float(nolds.dfa(signal, order=order, fit_exp="poly"))
        if not np.isfinite(alpha):
            return float("nan")
        return alpha
    except ImportError:
        logger.warning("nolds not installed — DFA will be NaN")
        return float("nan")
    except Exception as exc:
        logger.debug("dfa_alpha failed: %s", exc)
        return float("nan")


def hurst_exponent(signal: np.ndarray) -> float:
    """Hurst exponent via rescaled range (R/S) analysis.

    H ≈ 0.5 → random walk; H > 0.5 → persistent; H < 0.5 → anti-persistent.

    Returns float or NaN on failure.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if len(signal) < _MIN_LEN_HURST:
        return float("nan")
    if not np.isfinite(signal).all():
        return float("nan")
    if np.std(signal) < 1e-10:
        return float("nan")

    try:
        import nolds  # type: ignore
        h = float(nolds.hurst_rs(signal))
        if not np.isfinite(h):
            return float("nan")
        return h
    except ImportError:
        logger.warning("nolds not installed — Hurst will be NaN")
        return float("nan")
    except Exception as exc:
        logger.debug("hurst_exponent failed: %s", exc)
        return float("nan")


def higuchi_fractal_dimension(
    signal: np.ndarray,
    kmax: int = 8,
) -> float:
    """Higuchi Fractal Dimension (HFD) — pure NumPy, no external dependency.

    HFD estimates the fractal dimension of a time series directly from
    the signal itself without embedding.

    Parameters
    ----------
    signal : 1-D float array
    kmax   : maximum interval length (default 8; too large → noise)

    Returns float or NaN on failure.

    Reference
    ---------
    Higuchi (1988). Physica D 31, 277-283.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    N = len(signal)

    if N < _MIN_LEN_HFD:
        return float("nan")
    if not np.isfinite(signal).all():
        return float("nan")
    if np.std(signal) < 1e-10:
        return float("nan")

    Lk = []
    x = np.log(np.arange(1, kmax + 1))

    for k in range(1, kmax + 1):
        Lmk = []
        for m in range(1, k + 1):
            # Number of terms in the sum
            n_max = int(np.floor((N - m) / k))
            if n_max < 1:
                continue
            idx = np.arange(1, n_max + 1)
            lm = (
                np.sum(np.abs(signal[m + idx * k - 1] - signal[m + (idx - 1) * k - 1]))
                * (N - 1) / (k * k * n_max)
            )
            Lmk.append(lm)
        if Lmk:
            Lk.append(np.mean(Lmk))
        else:
            Lk.append(np.nan)

    Lk = np.array(Lk)
    valid = np.isfinite(Lk) & (Lk > 0)
    if valid.sum() < 2:
        return float("nan")

    # Least-squares fit of log(Lk) vs log(k) → slope = -HFD
    try:
        coeffs = np.polyfit(x[valid], np.log(Lk[valid]), 1)
        hfd = -coeffs[0]
        if not np.isfinite(hfd):
            return float("nan")
        return float(hfd)
    except Exception:
        return float("nan")


def compute_fractal_features(
    signal: np.ndarray,
    dfa_order: int = 1,
) -> dict[str, float]:
    """Compute all fractal/complexity features in one call.

    Returns
    -------
    dict with keys 'dfa_alpha', 'hurst', 'hfd'
    """
    return {
        "dfa_alpha": dfa_alpha(signal, order=dfa_order),
        "hurst":     hurst_exponent(signal),
        "hfd":       higuchi_fractal_dimension(signal),
    }
