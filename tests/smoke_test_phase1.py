"""
tests/smoke_test_phase1.py
--------------------------
Phase 1 smoke test.

Validates:
  1. Config loads and validates without error.
  2. AAMI annotation mapping is internally consistent.
  3. Seed-setting runs without error.
  4. Logger setup runs without error.
  5. A single MIT-BIH record (100) can be downloaded and loaded.
  6. ECGRecord fields are sane (correct types, shapes, no NaN).

Run from the project root:
    python tests/smoke_test_phase1.py

Exit code 0 = all tests passed.
Exit code 1 = one or more tests failed.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Ensure the project root is on the Python path regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

failures: list[str] = []


def check(name: str, fn) -> bool:
    """Run *fn* and print pass/fail.  Returns True on success."""
    try:
        fn()
        print(f"{PASS} {name}")
        return True
    except Exception:
        print(f"{FAIL} {name}")
        traceback.print_exc()
        failures.append(name)
        return False


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def test_config() -> None:
    from src.utils.config import load_config
    cfg = load_config()

    assert isinstance(cfg, dict), "Config must be a dict"
    assert "seed" in cfg
    assert "dataset" in cfg
    assert "aami_mapping" in cfg
    assert "preprocessing" in cfg
    assert cfg["dataset"]["fs"] == 360
    assert cfg["preprocessing"]["bandpass"]["lowcut"] == 0.5
    assert cfg["preprocessing"]["bandpass"]["highcut"] == 40.0

    # Record split must not overlap
    train = set(cfg["dataset"]["train_records"])
    val   = set(cfg["dataset"]["val_records"])
    test  = set(cfg["dataset"]["test_records"])
    assert not (train & val),  f"Train/val overlap: {train & val}"
    assert not (train & test), f"Train/test overlap: {train & test}"
    assert not (val & test),   f"Val/test overlap: {val & test}"

    # Output dirs must have been created
    for key, path in cfg["output"].items():
        assert Path(path).exists(), f"Output dir not created: {path}"


def test_annotations() -> None:
    from src.data.annotations import (
        AAMI_SYMBOL_TO_CLASS,
        AAMI_CLASS_TO_INDEX,
        INDEX_TO_AAMI_CLASS,
        symbol_to_aami,
        symbol_to_index,
        aami_to_index,
        index_to_aami,
        get_class_names,
        summarise_mapping,
    )

    # All 5 classes defined
    assert set(AAMI_CLASS_TO_INDEX.keys()) == {"N", "S", "V", "F", "Q"}

    # Inverse mapping is correct
    for cls, idx in AAMI_CLASS_TO_INDEX.items():
        assert INDEX_TO_AAMI_CLASS[idx] == cls

    # Spot-check symbol mappings
    assert symbol_to_aami("N") == "N"
    assert symbol_to_aami("L") == "N"
    assert symbol_to_aami("A") == "S"
    assert symbol_to_aami("V") == "V"
    assert symbol_to_aami("F") == "F"
    assert symbol_to_aami("/") == "Q"
    assert symbol_to_aami("+") is None   # non-beat symbol
    assert symbol_to_aami("~") is None   # non-beat symbol

    # Index helpers
    assert symbol_to_index("V") == 2
    assert symbol_to_index("+") is None
    assert aami_to_index("N") == 0
    assert index_to_aami(3) == "F"

    # Class names ordered by index
    names = get_class_names()
    assert names == ["N", "S", "V", "F", "Q"], f"Unexpected order: {names}"

    # Summary string is non-empty
    summary = summarise_mapping()
    assert len(summary) > 50

    # No symbol appears in two classes
    seen: dict[str, str] = {}
    for sym, cls in AAMI_SYMBOL_TO_CLASS.items():
        assert sym not in seen, f"Symbol '{sym}' mapped to both {seen[sym]} and {cls}"
        seen[sym] = cls


def test_seed() -> None:
    import numpy as np
    from src.utils.seed import set_global_seed, get_rng

    set_global_seed(42)

    # Two RNG instances with the same seed must produce the same values
    rng1 = get_rng(123)
    rng2 = get_rng(123)
    val1 = rng1.random()
    val2 = rng2.random()
    assert val1 == val2, f"RNG not reproducible: {val1} != {val2}"

    # After set_global_seed, numpy global produces deterministic output
    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    assert (a == b).all(), "NumPy global RNG not reproducible after set_global_seed"


def test_logger() -> None:
    import logging
    from src.utils.logger import setup_logging, get_logger

    # Basic setup without a file
    setup_logging(level="DEBUG")
    log = get_logger("smoke_test")
    log.debug("debug message")
    log.info("info message")
    log.warning("warning message")

    # Calling setup_logging twice must not duplicate handlers
    root = logging.getLogger()
    setup_logging(level="INFO")
    handler_count_1 = len(root.handlers)
    setup_logging(level="INFO")
    handler_count_2 = len(root.handlers)
    assert handler_count_1 == handler_count_2, (
        f"Duplicate handlers after repeated setup_logging calls: "
        f"{handler_count_1} → {handler_count_2}"
    )


def test_record_load() -> None:
    """Download (if needed) and load record 100 from MIT-BIH."""
    import numpy as np
    from src.utils.config import load_config
    from src.data.loader import load_record, ECGRecord

    cfg = load_config()
    data_dir = cfg["dataset"]["raw_dir"]
    fs = cfg["dataset"]["fs"]
    lead = cfg["dataset"]["lead"]

    rec = load_record(
        record_id=100,
        data_dir=data_dir,
        fs=fs,
        lead=lead,
        download=True,
    )

    # Type checks
    assert isinstance(rec, ECGRecord), "load_record must return ECGRecord"
    assert isinstance(rec.signal, np.ndarray)
    assert isinstance(rec.beat_samples, np.ndarray)
    assert isinstance(rec.aami_labels, np.ndarray)

    # Shape / dtype
    assert rec.signal.ndim == 1,        "Signal must be 1D"
    assert rec.signal.dtype == np.float32
    assert rec.beat_samples.ndim == 1
    assert rec.aami_labels.ndim == 1
    assert rec.aami_labels.dtype == np.int32

    # Consistency
    n = rec.n_beats
    assert len(rec.beat_samples) == n
    assert len(rec.beat_symbols) == n
    assert len(rec.aami_classes) == n
    assert len(rec.aami_labels)  == n

    # Signal quality
    assert not any(v != v for v in rec.signal), "Signal contains NaN"  # NaN != NaN
    assert rec.n_samples > 0
    assert rec.n_beats  > 0

    # AAMI labels must all be in [0, 4]
    assert int(rec.aami_labels.min()) >= 0
    assert int(rec.aami_labels.max()) <= 4

    # Class distribution sums to n_beats
    total = sum(rec.class_distribution.values())
    assert total == n, f"Class distribution sum {total} != n_beats {n}"

    print(f"  {INFO} {rec}")
    print(f"  {INFO} Class distribution: {rec.class_distribution}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Phase 1 Smoke Test")
    print("=" * 60 + "\n")

    check("Config loaded and validated", test_config)
    check("Annotation mapping validated", test_annotations)
    check("Seed setting works", test_seed)
    check("Logger works", test_logger)
    check("Record download + load (record 100)", test_record_load)

    print("\n" + "=" * 60)
    if failures:
        print(f"\033[91m  {len(failures)} test(s) FAILED: {failures}\033[0m")
        print("=" * 60 + "\n")
        sys.exit(1)
    else:
        print("\033[92m  All Phase 1 checks passed.\033[0m")
        print("=" * 60 + "\n")
        sys.exit(0)
