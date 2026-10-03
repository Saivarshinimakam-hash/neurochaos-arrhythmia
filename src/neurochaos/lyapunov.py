"""
src/neurochaos/lyapunov.py
---------------------------
Lyapunov-exponent-related features for ECG beat windows.

Implemented features
--------------------
- Largest Lyapunov Exponent (LLE) via nolds.lyap_r (Rosenstein's method)
  A positive LLE indicates chaotic dynamics; near-zero = periodic;
  negative = stable fixed-point.

- LLE divergence rate (alias for the LLE value; included for naming clarity
  in the feature vector).

Libraries
---------
- nolds >= 0.5.2  (Rosenstein's algorithm implementation)

Fallback
--------
If nolds is unavailable or the calculation fails, NaN is returned and
a warning is logged.  The caller (extractor.py) handles NaN.

Reference
---------
- Rosenstein, Collins & De Luca (1993). A practical method for
  calculating largest Lyapunov exponents from small data sets.
  Physica D 65, 117-134.
- Kantz (1994). A robust method to estimate the maximal Lyapunov
  exponent of a time series. Physics Letters A 185, 77-87.
"""

from __future__ import annotations

import logging

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

_MIN_LEN = 30   # nolds needs a reasonable embedding neighbourhood


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def largest_lyapunov_exponent(
    signal: np.ndarray,
    emb_dim: int = 3,
    min_tsep: int = 1,
) -> float:
    """Compute the Largest Lyapunov Exponent using Rosenstein's method.

    Parameters
    ----------
    signal  : 1-D float array (single beat window, already filtered)
    emb_dim : embedding dimension for phase-space reconstruction
    min_tsep: minimum temporal separation between nearest neighbours
              (avoids trivially close temporal neighbours)

    Returns
    -------
    float — LLE value (positive → chaos, ≈0 → periodic, negative → stable)
    NaN on any failure.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()

    if len(signal) < _MIN_LEN:
        logger.debug("LLE: signal too short (%d)", len(signal))
        return float("nan")
    if not np.isfinite(signal).all():
        logger.debug("LLE: non-finite values in signal")
        return float("nan")
    if np.std(signal) < 1e-10:
        logger.debug("LLE: flat signal (std≈0)")
        return float("nan")

    try:
        import nolds  # type: ignore
        lle = float(nolds.lyap_r(signal,emb_dim=emb_dim,min_tsep=min_tsep,
        fit="poly",
        )
)
        if not np.isfinite(lle):
            logger.debug("LLE: nolds returned non-finite value")
            return float("nan")
        return lle
    except ImportError:
        logger.warning("nolds not installed — LLE will be NaN")
        return float("nan")
    except Exception as exc:
        logger.debug("LLE computation failed: %s", exc)
        return float("nan")


def compute_lyapunov_features(
    signal: np.ndarray,
    emb_dim: int = 3,
) -> dict[str, float]:
    """Compute all Lyapunov-based features in one call.

    Currently returns one feature; structured as a dict for easy
    extension (e.g. adding Kantz LLE or LLE spectrum).

    Returns
    -------
    dict with key 'lle'
    """
    lle = largest_lyapunov_exponent(signal, emb_dim=emb_dim)
    return {"lle": lle}
