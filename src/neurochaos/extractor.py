"""
src/neurochaos/extractor.py
----------------------------
Unified NeuroChaos feature extractor.

This is the single entry point for all nonlinear-dynamics features.
It calls each sub-module, collects results into a consistent feature
vector, and handles failures gracefully.

Feature vector (12 features, fixed order)
------------------------------------------
Index  Name                Module
  0    sample_entropy      entropy.py
  1    permutation_entropy entropy.py
  2    approximate_entropy entropy.py
  3    rr                  recurrence.py  (recurrence rate)
  4    det                 recurrence.py  (determinism)
  5    lam                 recurrence.py  (laminarity)
  6    lle                 lyapunov.py    (largest Lyapunov exponent)
  7    ps_mean_velocity    phase_space.py
  8    ps_traj_length      phase_space.py
  9    ps_logdet_cov       phase_space.py
 10    dfa_alpha           fractal.py
 11    hurst               fractal.py
 12    hfd                 fractal.py     (Higuchi fractal dimension)

Total: 13 features

NaN handling
------------
If any individual feature fails, its slot is NaN.  After extraction, a
configurable imputation strategy fills NaN values before model input:
  - 'zero'   : replace NaN with 0 (default, fast)
  - 'median' : replace NaN with the median across the dataset
                (requires a fitted NeuroChaosImputer)

The imputer is fitted on TRAINING data only to avoid leakage.

Usage
-----
    from src.neurochaos.extractor import NeuroChaosExtractor
    extractor = NeuroChaosExtractor(cfg)
    features = extractor.extract_beat(beat_waveform)        # shape (13,)
    X = extractor.extract_dataset(beat_dataset.waveforms)   # shape (N, 13)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from src.neurochaos.entropy import sample_entropy, permutation_entropy, approximate_entropy
from src.neurochaos.recurrence import compute_rqa_features
from src.neurochaos.lyapunov import compute_lyapunov_features
from src.neurochaos.phase_space import compute_phase_space_features
from src.neurochaos.fractal import compute_fractal_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature names (canonical, ordered)
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "sample_entropy",
    "permutation_entropy",
    "approximate_entropy",
    "rr",
    "det",
    "lam",
    "lle",
    "ps_mean_velocity",
    "ps_traj_length",
    "ps_logdet_cov",
    "dfa_alpha",
    "hurst",
    "hfd",
]

N_FEATURES = len(FEATURE_NAMES)   # 13


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class NeuroChaosExtractor:
    """Extract the full NeuroChaos feature vector from ECG beat windows.

    Parameters
    ----------
    cfg : dict
        Project configuration dict (from load_config()).
        Reads neurochaos sub-section for hyperparameters.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        nc = cfg.get("neurochaos", {})

        self._se_m        = nc.get("sample_entropy",      {}).get("m",        2)
        self._se_r        = nc.get("sample_entropy",      {}).get("r_factor",  0.2)
        self._pe_order    = nc.get("permutation_entropy", {}).get("order",     3)
        self._pe_delay    = nc.get("permutation_entropy", {}).get("delay",     1)
        self._ape_m       = nc.get("approximate_entropy", {}).get("m",        2)
        self._ape_r       = nc.get("approximate_entropy", {}).get("r_factor",  0.2)
        self._rqa_dim     = nc.get("rqa",                 {}).get("dimension", 3)
        self._rqa_tau     = nc.get("rqa",                 {}).get("time_delay",1)
        self._rqa_thresh  = nc.get("rqa",                 {}).get("threshold", 0.1)
        self._lyap_dim    = nc.get("lyapunov",            {}).get("emb_dim",   3)
        self._ps_dim      = nc.get("phase_space",         {}).get("emb_dim",   3)
        self._ps_tau      = nc.get("phase_space",         {}).get("time_delay",1)
        self._dfa_order   = nc.get("fractal",             {}).get("dfa_order", 1)

        logger.debug("NeuroChaosExtractor initialised with %d features", N_FEATURES)

    def extract_beat(self, waveform: np.ndarray) -> np.ndarray:
        """Extract all features from a single beat waveform.

        Parameters
        ----------
        waveform : 1-D float array of length beat_window_size (e.g. 288)

        Returns
        -------
        np.ndarray, shape (N_FEATURES,), dtype float32
            NaN where calculation failed.
        """
        w = np.asarray(waveform, dtype=np.float64).ravel()

        # Entropy features
        se  = sample_entropy(w,      m=self._se_m,     r_factor=self._se_r)
        pe  = permutation_entropy(w, order=self._pe_order, delay=self._pe_delay)
        ape = approximate_entropy(w, m=self._ape_m,    r_factor=self._ape_r)

        # RQA features
        rqa = compute_rqa_features(w, dimension=self._rqa_dim,
                                   time_delay=self._rqa_tau,
                                   threshold=self._rqa_thresh)

        # Lyapunov
        lyap = compute_lyapunov_features(w, emb_dim=self._lyap_dim)

        # Phase space
        ps = compute_phase_space_features(w, emb_dim=self._ps_dim,
                                          time_delay=self._ps_tau)

        # Fractal
        frac = compute_fractal_features(w, dfa_order=self._dfa_order)

        vec = np.array([
            se, pe, ape,
            rqa["rr"], rqa["det"], rqa["lam"],
            lyap["lle"],
            ps["ps_mean_velocity"], ps["ps_traj_length"], ps["ps_logdet_cov"],
            frac["dfa_alpha"], frac["hurst"], frac["hfd"],
        ], dtype=np.float32)

        return vec

    def extract_dataset(
        self,
        waveforms: np.ndarray,
        show_progress: bool = True,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """Extract features from a 2-D array of beat waveforms.

        Parameters
        ----------
        waveforms    : shape (n_beats, window_size)
        show_progress: show tqdm progress bar
        n_jobs       : parallel workers (1 = serial; -1 = all CPUs)
                       Parallelism uses joblib.Parallel if n_jobs != 1.

        Returns
        -------
        np.ndarray, shape (n_beats, N_FEATURES), dtype float32
        """
        n = len(waveforms)
        if n == 0:
            return np.empty((0, N_FEATURES), dtype=np.float32)

        if n_jobs == 1:
            result = np.zeros((n, N_FEATURES), dtype=np.float32)
            iterable = tqdm(range(n), desc="NeuroChaos features",
                            disable=not show_progress)
            for i in iterable:
                result[i] = self.extract_beat(waveforms[i])
        else:
            try:
                from joblib import Parallel, delayed  # type: ignore
                rows = Parallel(n_jobs=n_jobs)(
                    delayed(self.extract_beat)(waveforms[i])
                    for i in tqdm(range(n), desc="NeuroChaos features",
                                  disable=not show_progress)
                )
                result = np.stack(rows, axis=0).astype(np.float32)
            except ImportError:
                logger.warning("joblib not available; falling back to serial extraction")
                result = self.extract_dataset(waveforms, show_progress=show_progress,
                                              n_jobs=1)

        nan_count = int(np.sum(np.isnan(result)))
        if nan_count > 0:
            logger.warning(
                "extract_dataset: %d NaN values in feature matrix "
                "(%.1f%% of cells) — impute before modelling",
                nan_count, 100.0 * nan_count / result.size,
            )

        return result


# ---------------------------------------------------------------------------
# NaN imputer (fitted on training data only)
# ---------------------------------------------------------------------------

class NeuroChaosImputer:
    """Median imputer for NeuroChaos feature matrices.

    Fits on training data, transforms all splits.  This prevents leakage
    from val/test statistics.

    Usage
    -----
        imputer = NeuroChaosImputer()
        X_train = imputer.fit_transform(X_train_raw)
        X_val   = imputer.transform(X_val_raw)
        X_test  = imputer.transform(X_test_raw)
    """

    def __init__(self) -> None:
        self._medians: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "NeuroChaosImputer":
        """Compute per-feature medians from training data (ignoring NaN).

        Parameters
        ----------
        X : shape (n_beats, N_FEATURES)
        """
        self._medians = np.nanmedian(X, axis=0)
        n_cols = len(self._medians)
        nan_features = [
            FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"col_{i}"
            for i in range(n_cols)
            if np.isnan(self._medians[i])
        ]
        if nan_features:
            logger.warning(
                "Features entirely NaN in training data (will be 0): %s",
                nan_features,
            )
            self._medians = np.where(np.isnan(self._medians), 0.0, self._medians)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace NaN values with training-data medians.

        Parameters
        ----------
        X : shape (n_beats, N_FEATURES)

        Returns
        -------
        np.ndarray, float32, same shape — no NaN.
        """
        if self._medians is None:
            raise RuntimeError("Call fit() before transform().")
        X = X.copy()
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                X[mask, j] = self._medians[j]
        return X.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def medians(self) -> np.ndarray:
        if self._medians is None:
            raise RuntimeError("Imputer has not been fitted.")
        return self._medians
