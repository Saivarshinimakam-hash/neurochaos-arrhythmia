"""
src/data/preprocessing.py
--------------------------
ECG signal preprocessing for the MIT-BIH Arrhythmia pipeline.

Operations (in order)
---------------------
1. Signal cleaning  — replace NaN/Inf with interpolated or zero values,
                      clip extreme outliers.
2. Band-pass filter — 4th-order zero-phase Butterworth, 0.5–40 Hz.
3. (Beat-level)     — per-beat z-score normalisation applied AFTER
                      segmentation, not here.

Design rules
------------
- The full-record filter is applied BEFORE beat segmentation.
- Normalisation parameters (mean, std) are computed per beat from the
  training beats only, then applied to val/test beats.  This function
  provides the per-beat normaliser; fitting on the dataset is done in
  src/data/splitting.py.
- No global dataset statistics are computed here to avoid leakage.
- ``filtfilt`` (zero-phase) is used for filtering to avoid phase
  distortion on R-peak timing.
- SOS (second-order sections) filter representation is used for
  numerical stability at low cut-off frequencies.

References
----------
- Scipy sosfiltfilt documentation:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html
- Pan-Tompkins, 1985 — original ECG preprocessing recommendations.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt_zi

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_signal(signal: np.ndarray, record_id: str = "") -> np.ndarray:
    """Replace NaN / Inf values and clip extreme outliers.

    Strategy
    --------
    - NaN values are replaced by linear interpolation between the
      nearest finite neighbours.  If an entire segment is NaN the
      fallback is 0.0.
    - Inf values are clipped to ±10 × the inter-quartile range of the
      finite portion, which preserves amplitude scale while removing
      non-physical spikes.

    Parameters
    ----------
    signal:
        1-D float array (raw ECG in mV).
    record_id:
        Used only for log messages.

    Returns
    -------
    np.ndarray
        Cleaned signal of the same shape and float32 dtype.
    """
    signal = np.array(signal, dtype=np.float64)

    n_nan = int(np.sum(np.isnan(signal)))
    n_inf = int(np.sum(np.isinf(signal)))

    if n_nan > 0:
        logger.warning(
            "Record %s: replacing %d NaN sample(s) via linear interpolation",
            record_id, n_nan,
        )
        signal = _interpolate_nan(signal)

    if n_inf > 0:
        logger.warning(
            "Record %s: clipping %d Inf sample(s)", record_id, n_inf
        )
        finite_mask = np.isfinite(signal)
        if finite_mask.any():
            q25, q75 = np.percentile(signal[finite_mask], [25, 75])
            iqr = q75 - q25
            clip_val = 10.0 * iqr if iqr > 0 else 10.0
        else:
            clip_val = 10.0
        signal = np.clip(signal, -clip_val, clip_val)

    return signal.astype(np.float32)


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
    record_id: str = "",
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter.

    Uses SOS (second-order sections) representation for numerical
    stability, especially important at the low 0.5 Hz cut-off.
    ``sosfiltfilt`` applies the filter twice (forward + backward) to
    achieve zero phase shift, which preserves R-peak timing.

    Parameters
    ----------
    signal:
        1-D float array (ECG samples in mV).
    fs:
        Sampling frequency in Hz.
    lowcut:
        Low cut-off frequency in Hz (default 0.5).
    highcut:
        High cut-off frequency in Hz (default 40.0).
    order:
        Butterworth filter order (default 4; higher orders risk
        numerical instability near the Nyquist frequency).
    record_id:
        Used only for log messages.

    Returns
    -------
    np.ndarray
        Filtered signal, float32, same length as input.

    Raises
    ------
    ValueError
        If cut-off frequencies are outside the valid range, or if the
        signal is too short to filter.
    """
    nyquist = fs / 2.0

    if lowcut <= 0:
        raise ValueError(f"lowcut must be > 0 Hz, got {lowcut}")
    if highcut >= nyquist:
        raise ValueError(
            f"highcut ({highcut} Hz) must be < Nyquist ({nyquist} Hz) for fs={fs}"
        )
    if lowcut >= highcut:
        raise ValueError(
            f"lowcut ({lowcut}) must be < highcut ({highcut})"
        )

    # sosfiltfilt needs at least 3× the padlen = 3×(2×order-1) samples
    min_len = 3 * (2 * order - 1) * 3  # conservative
    if len(signal) < min_len:
        raise ValueError(
            f"Signal too short to filter: {len(signal)} samples, "
            f"need at least {min_len} for order={order}"
        )

    # Design filter in SOS form (numerically stable)
    low_norm  = lowcut  / nyquist
    high_norm = highcut / nyquist
    sos = butter(order, [low_norm, high_norm], btype="band", output="sos")

    signal_f64 = signal.astype(np.float64)
    filtered = sosfiltfilt(sos, signal_f64)

    if not np.all(np.isfinite(filtered)):
        n_bad = int(np.sum(~np.isfinite(filtered)))
        logger.warning(
            "Record %s: filter produced %d non-finite sample(s); "
            "falling back to input signal for those samples",
            record_id, n_bad,
        )
        bad_mask = ~np.isfinite(filtered)
        filtered[bad_mask] = signal_f64[bad_mask]

    logger.debug(
        "Record %s: band-pass filtered [%.1f–%.1f Hz], order=%d",
        record_id, lowcut, highcut, order,
    )
    return filtered.astype(np.float32)


def preprocess_signal(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
    record_id: str = "",
) -> np.ndarray:
    """Full preprocessing pipeline for a single ECG record signal.

    Steps
    -----
    1. Clean (NaN/Inf replacement)
    2. Band-pass filter

    Parameters
    ----------
    signal:
        Raw 1-D ECG signal (mV).
    fs:
        Sampling frequency in Hz.
    lowcut, highcut, order:
        Band-pass filter parameters.
    record_id:
        Used in log messages.

    Returns
    -------
    np.ndarray
        Preprocessed float32 signal, same length as input.
    """
    signal = clean_signal(signal, record_id=record_id)
    signal = bandpass_filter(
        signal, fs=fs, lowcut=lowcut, highcut=highcut,
        order=order, record_id=record_id,
    )
    return signal


# ---------------------------------------------------------------------------
# Per-beat normalisation  (applied AFTER segmentation)
# ---------------------------------------------------------------------------

def normalize_beat_zscore(
    beat: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Z-score normalise a single beat window.

    Computes mean and std from the beat itself (per-beat statistics).
    This is intentionally self-contained — no dataset-wide statistics
    are used — which means it is safe to apply identically to train,
    val, and test beats without any leakage concern.

    Parameters
    ----------
    beat:
        1-D float array of length ``beat_window_size`` (e.g. 288).
    epsilon:
        Small constant added to std to avoid division by zero for
        flat segments.

    Returns
    -------
    np.ndarray
        Normalised beat, float32.
    """
    beat = np.asarray(beat, dtype=np.float64)
    mean = beat.mean()
    std  = beat.std()
    normalised = (beat - mean) / (std + epsilon)
    return normalised.astype(np.float32)


def normalize_beats_zscore(
    beats: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Z-score normalise a 2-D array of beats row-wise.

    Parameters
    ----------
    beats:
        Shape ``(n_beats, beat_window_size)``.
    epsilon:
        Stability constant.

    Returns
    -------
    np.ndarray
        Normalised beats, float32, same shape.
    """
    beats = np.asarray(beats, dtype=np.float64)
    if beats.ndim != 2:
        raise ValueError(
            f"beats must be 2-D (n_beats, window_size), got shape {beats.shape}"
        )
    means = beats.mean(axis=1, keepdims=True)
    stds  = beats.std(axis=1,  keepdims=True)
    normalised = (beats - means) / (stds + epsilon)
    return normalised.astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interpolate_nan(signal: np.ndarray) -> np.ndarray:
    """Replace NaN values by linear interpolation of finite neighbours.

    If the signal starts or ends with NaN, the nearest finite value is
    used for extrapolation (edge-fill).  If the entire signal is NaN,
    returns an array of zeros.
    """
    nan_mask = np.isnan(signal)
    if not nan_mask.any():
        return signal

    indices = np.arange(len(signal))
    finite_mask = ~nan_mask

    if not finite_mask.any():
        logger.warning("Signal is entirely NaN; replacing with zeros")
        return np.zeros_like(signal)

    # np.interp clamps to edge values for out-of-range, which handles
    # leading/trailing NaN correctly.
    signal[nan_mask] = np.interp(
        indices[nan_mask],
        indices[finite_mask],
        signal[finite_mask],
    )
    return signal
