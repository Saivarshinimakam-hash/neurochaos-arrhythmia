"""
src/utils/config.py
-------------------
Loads and validates the master config.yaml.

Design decisions:
- Returns a plain Python dict (not a custom class) for simplicity and
  easy serialisation.
- Resolves all relative paths to absolute paths anchored at the project
  root so that scripts work regardless of the working directory.
- Performs lightweight structural validation to catch missing keys early
  rather than letting failures surface inside training loops.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# The project root is two levels above this file:
# src/utils/config.py → src/utils → src → project_root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate config.yaml; resolve relative paths to absolute.

    Parameters
    ----------
    config_path:
        Path to the YAML config file.  Defaults to ``<project_root>/config.yaml``.

    Returns
    -------
    dict
        Fully validated configuration dictionary with all paths absolute.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    KeyError
        If a required top-level section is missing.
    ValueError
        If a required value has an invalid type or range.
    """
    if config_path is None:
        config_path = _PROJECT_ROOT / "config.yaml"

    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Expected location: <project_root>/config.yaml"
        )

    with config_path.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    if cfg is None:
        raise ValueError(f"Config file is empty: {config_path}")

    _validate(cfg)
    _resolve_paths(cfg)
    _create_output_dirs(cfg)

    return cfg


def get_project_root() -> Path:
    """Return the absolute project root directory."""
    return _PROJECT_ROOT


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL_KEYS = {
    "seed",
    "dataset",
    "aami_mapping",
    "class_labels",
    "preprocessing",
    "neurochaos",
    "classical_ml",
    "cnn",
    "bilstm",
    "hybrid",
    "output",
    "robustness",
}

_REQUIRED_DATASET_KEYS = {
    "name",
    "fs",
    "lead",
    "raw_dir",
    "processed_dir",
    "features_dir",
    "all_records",
    "train_records",
    "val_records",
    "test_records",
}

_REQUIRED_PREPROCESSING_KEYS = {
    "bandpass",
    "normalization",
    "beat_before_samples",
    "beat_after_samples",
    "beat_window_size",
}


def _validate(cfg: dict[str, Any]) -> None:
    """Raise informative errors if the config is structurally invalid."""

    # Top-level sections
    missing = _REQUIRED_TOP_LEVEL_KEYS - set(cfg.keys())
    if missing:
        raise KeyError(f"Config is missing required top-level keys: {sorted(missing)}")

    # Dataset section
    ds = cfg["dataset"]
    missing_ds = _REQUIRED_DATASET_KEYS - set(ds.keys())
    if missing_ds:
        raise KeyError(f"config.dataset is missing keys: {sorted(missing_ds)}")

    # Preprocessing section
    pp = cfg["preprocessing"]
    missing_pp = _REQUIRED_PREPROCESSING_KEYS - set(pp.keys())
    if missing_pp:
        raise KeyError(f"config.preprocessing is missing keys: {sorted(missing_pp)}")

    # Numeric sanity checks
    if not isinstance(cfg["seed"], int):
        raise ValueError(f"config.seed must be an integer, got {type(cfg['seed'])}")

    fs = ds["fs"]
    if not isinstance(fs, (int, float)) or fs <= 0:
        raise ValueError(f"config.dataset.fs must be a positive number, got {fs}")

    bp = pp["bandpass"]
    if bp["lowcut"] <= 0:
        raise ValueError(f"bandpass lowcut must be > 0, got {bp['lowcut']}")
    if bp["highcut"] >= fs / 2:
        raise ValueError(
            f"bandpass highcut ({bp['highcut']} Hz) must be < Nyquist "
            f"({fs / 2} Hz) for fs={fs}"
        )
    if bp["lowcut"] >= bp["highcut"]:
        raise ValueError(
            f"bandpass lowcut ({bp['lowcut']}) must be < highcut ({bp['highcut']})"
        )

    beat_window = pp["beat_window_size"]
    before = pp["beat_before_samples"]
    after = pp["beat_after_samples"]
    if before + after != beat_window:
        raise ValueError(
            f"beat_before_samples ({before}) + beat_after_samples ({after}) "
            f"must equal beat_window_size ({beat_window})"
        )

    # Record lists must not overlap
    train_set = set(ds["train_records"])
    val_set = set(ds["val_records"])
    test_set = set(ds["test_records"])

    tv_overlap = train_set & val_set
    tt_overlap = train_set & test_set
    vt_overlap = val_set & test_set
    if tv_overlap or tt_overlap or vt_overlap:
        raise ValueError(
            f"Record split overlap detected — "
            f"train∩val={tv_overlap}, train∩test={tt_overlap}, val∩test={vt_overlap}"
        )

    # AAMI mapping must cover all 5 classes
    required_classes = {"N", "S", "V", "F", "Q"}
    present_classes = set(cfg["aami_mapping"].keys())
    missing_classes = required_classes - present_classes
    if missing_classes:
        raise KeyError(
            f"aami_mapping is missing class definitions: {sorted(missing_classes)}"
        )

    # class_labels must contain the same 5 classes
    label_classes = set(cfg["class_labels"].keys())
    if label_classes != required_classes:
        raise KeyError(
            f"class_labels must define exactly {required_classes}, "
            f"got {label_classes}"
        )


def _resolve_paths(cfg: dict[str, Any]) -> None:
    """Convert all relative directory paths to absolute paths in-place."""
    ds = cfg["dataset"]
    for key in ("raw_dir", "processed_dir", "features_dir"):
        ds[key] = str((_PROJECT_ROOT / ds[key]).resolve())

    out = cfg["output"]
    for key in out:
        out[key] = str((_PROJECT_ROOT / out[key]).resolve())


def _create_output_dirs(cfg: dict[str, Any]) -> None:
    """Create output directories if they do not already exist."""
    dirs_to_create = list(cfg["output"].values())
    # Also ensure the data dirs exist
    ds = cfg["dataset"]
    dirs_to_create += [ds["raw_dir"], ds["processed_dir"], ds["features_dir"]]

    for d in dirs_to_create:
        Path(d).mkdir(parents=True, exist_ok=True)
