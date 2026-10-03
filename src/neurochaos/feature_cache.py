"""
Persistent cache for NeuroChaos feature extraction.

The cache stores features together with enough metadata to verify that
the cached matrix belongs to the expected dataset split/configuration.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.neurochaos.extractor import FEATURE_NAMES, N_FEATURES


CACHE_VERSION = "1.0"


def _config_hash(cfg: dict[str, Any]) -> str:
    """Create a stable hash of the relevant NeuroChaos configuration."""
    neurochaos_cfg = cfg.get("neurochaos", {})

    payload = json.dumps(
        neurochaos_cfg,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_path(
    cache_dir: Path,
    split_name: str,
) -> Path:
    """Return the cache file path for a dataset split."""
    return cache_dir / f"{split_name}_neurochaos.npz"


def save_feature_cache(
    path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    record_ids: list[str],
    beat_indices: np.ndarray,
    r_peak_samples: np.ndarray,
    split_name: str,
    cfg: dict[str, Any],
) -> None:
    """Save NeuroChaos features and provenance metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    beat_indices = np.asarray(beat_indices, dtype=np.int64)
    r_peak_samples = np.asarray(r_peak_samples, dtype=np.int64)

    if features.ndim != 2 or features.shape[1] != N_FEATURES:
        raise ValueError(
            f"Expected features shape (N, {N_FEATURES}), "
            f"got {features.shape}"
        )

    n = len(features)

    if len(labels) != n:
        raise ValueError("features and labels length mismatch")

    if len(record_ids) != n:
        raise ValueError("features and record_ids length mismatch")

    if len(beat_indices) != n:
        raise ValueError("features and beat_indices length mismatch")

    if len(r_peak_samples) != n:
        raise ValueError("features and r_peak_samples length mismatch")

    metadata = {
        "cache_version": CACHE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "split_name": split_name,
        "n_samples": n,
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "config_hash": _config_hash(cfg),
    }

    np.savez_compressed(
        path,
        features=features,
        labels=labels,
        record_ids=np.asarray(record_ids, dtype=str),
        beat_indices=beat_indices,
        r_peak_samples=r_peak_samples,
        metadata=np.asarray(json.dumps(metadata)),
    )


def load_feature_cache(
    path: Path,
    split_name: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Load and validate a NeuroChaos feature cache."""

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        required = {
            "features",
            "labels",
            "record_ids",
            "beat_indices",
            "r_peak_samples",
            "metadata",
        }

        missing = required.difference(data.files)

        if missing:
            raise ValueError(
                f"Cache {path} is missing fields: {sorted(missing)}"
            )

        features = data["features"]
        labels = data["labels"]
        record_ids = data["record_ids"].astype(str).tolist()
        beat_indices = data["beat_indices"]
        r_peak_samples = data["r_peak_samples"]

        metadata = json.loads(str(data["metadata"].item()))

    # Validate cache identity.
    if metadata.get("cache_version") != CACHE_VERSION:
        raise ValueError(
            f"Unsupported cache version: "
            f"{metadata.get('cache_version')!r}"
        )

    if metadata.get("split_name") != split_name:
        raise ValueError(
            f"Cache split mismatch: expected {split_name!r}, "
            f"found {metadata.get('split_name')!r}"
        )

    expected_hash = _config_hash(cfg)

    if metadata.get("config_hash") != expected_hash:
        raise ValueError(
            "NeuroChaos configuration changed since this cache "
            "was created. Recompute the cache."
        )

    if features.ndim != 2 or features.shape[1] != N_FEATURES:
        raise ValueError(
            f"Invalid cached feature shape: {features.shape}"
        )

    n = len(features)

    if len(labels) != n:
        raise ValueError("Cached features/labels length mismatch")

    if len(record_ids) != n:
        raise ValueError("Cached features/record_ids length mismatch")

    if len(beat_indices) != n:
        raise ValueError("Cached features/beat_indices length mismatch")

    if len(r_peak_samples) != n:
        raise ValueError(
            "Cached features/r_peak_samples length mismatch"
        )

    cached_names = metadata.get("feature_names")

    if cached_names != FEATURE_NAMES:
        raise ValueError(
            "Cached feature names do not match the current extractor."
        )

    return {
        "features": features.astype(np.float32),
        "labels": labels.astype(np.int32),
        "record_ids": record_ids,
        "beat_indices": beat_indices.astype(np.int64),
        "r_peak_samples": r_peak_samples.astype(np.int64),
        "metadata": metadata,
    }
def save_checkpoint(
    path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    record_ids: list[str],
    beat_indices: np.ndarray,
    r_peak_samples: np.ndarray,
    split_name: str,
    cfg: dict[str, Any],
    completed: int,
    ) -> None:
    """Save an incremental NeuroChaos extraction checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    beat_indices = np.asarray(beat_indices, dtype=np.int64)
    r_peak_samples = np.asarray(r_peak_samples, dtype=np.int64)

    if features.ndim != 2 or features.shape[1] != N_FEATURES:
        raise ValueError(
            f"Expected features shape (N, {N_FEATURES}), "
            f"got {features.shape}"
        )

    n = len(features)

    if len(labels) != n:
        raise ValueError("features and labels length mismatch")

    if len(record_ids) != n:
        raise ValueError("features and record_ids length mismatch")

    if len(beat_indices) != n:
        raise ValueError("features and beat_indices length mismatch")

    if len(r_peak_samples) != n:
        raise ValueError("features and r_peak_samples length mismatch")

    metadata = {
        "cache_version": CACHE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "split_name": split_name,
        "n_samples": n,
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "config_hash": _config_hash(cfg),
        "checkpoint": True,
        "completed": completed,
    }

    np.savez_compressed(
        path,
        features=features,
        labels=labels,
        record_ids=np.asarray(record_ids, dtype=str),
        beat_indices=beat_indices,
        r_peak_samples=r_peak_samples,
        metadata=np.asarray(json.dumps(metadata)),
    )