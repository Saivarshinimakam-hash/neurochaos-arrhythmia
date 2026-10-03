"""
src/utils/seed.py
-----------------
Central reproducibility helper.

Sets seeds for:
  - Python's built-in ``random`` module
  - NumPy
  - TensorFlow / Keras (if installed)

Call ``set_global_seed(cfg["seed"])`` once at the start of every script
before any data loading or model initialisation.

Notes
-----
- TensorFlow GPU operations are still subject to non-determinism in some
  CUDA kernels.  Setting ``TF_DETERMINISTIC_OPS=1`` env var before
  importing TensorFlow helps but may not eliminate all sources.
- scikit-learn uses NumPy's global RNG, so seeding NumPy is sufficient.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds across all relevant libraries.

    Parameters
    ----------
    seed:
        Integer seed value.  Must be in [0, 2**32 - 1].

    Raises
    ------
    TypeError
        If *seed* is not an integer.
    ValueError
        If *seed* is outside the valid range.
    """
    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}")
    if not (0 <= seed <= 2**32 - 1):
        raise ValueError(f"Seed must be in [0, 2**32-1], got {seed}")

    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # TensorFlow / Keras (optional dependency)
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
        # Encourage deterministic GPU ops (best-effort)
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
        logger.debug("TensorFlow seed set to %d", seed)
    except ImportError:
        logger.debug("TensorFlow not installed; skipping TF seed setting")

    # Ensure PYTHONHASHSEED is logged (must be set before interpreter start
    # to take effect, but we at least document it here)
    hashseed = os.environ.get("PYTHONHASHSEED", "not set")
    logger.debug(
        "Global seed set to %d | PYTHONHASHSEED=%s", seed, hashseed
    )


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a seeded NumPy Generator for use in a specific function.

    Preferred over calling ``np.random.seed`` globally inside functions,
    because it does not affect the global RNG state.

    Parameters
    ----------
    seed:
        Optional integer seed.  If *None*, uses the global state.

    Returns
    -------
    numpy.random.Generator
    """
    return np.random.default_rng(seed)
