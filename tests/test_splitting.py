"""
Test the dataset splitting pipeline.

Verifies:
1. Train/validation/test records do not overlap.
2. All splits contain data.
3. Beat windows have the expected size.
4. Waveforms are finite.
5. Per-beat normalization works.
6. Record-level integrity is preserved.
"""

import numpy as np

from src.utils.config import load_config
from src.data.splitting import (
    build_datasets,
    verify_no_overlap,
    verify_record_level_integrity,
)


def main():
    print("=" * 70)
    print("TESTING DATASET SPLITTING PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------
    cfg = load_config()

    train_ids = [str(x) for x in cfg["dataset"]["train_records"]]
    val_ids = [str(x) for x in cfg["dataset"]["val_records"]]
    test_ids = [str(x) for x in cfg["dataset"]["test_records"]]

    print("\n[1] Checking configured record split...")
    print(f"Train records: {len(train_ids)}")
    print(f"Val records:   {len(val_ids)}")
    print(f"Test records:  {len(test_ids)}")

    verify_no_overlap(train_ids, val_ids, test_ids)
    print("PASS: No record overlap.")

    # ---------------------------------------------------------
    # 2. Build datasets
    # ---------------------------------------------------------
    print("\n[2] Building train/validation/test datasets...")
    print("This may take some time if MIT-BIH records need to be downloaded.")

    train_ds, val_ds, test_ds = build_datasets(
        cfg,
        download=True,
        normalize=True,
        allow_padding=False,
        save_manifest=True,
    )

    # ---------------------------------------------------------
    # 3. Check that all splits contain data
    # ---------------------------------------------------------
    print("\n[3] Checking split sizes...")

    assert not train_ds.is_empty(), "Train dataset is empty."
    assert not val_ds.is_empty(), "Validation dataset is empty."
    assert not test_ds.is_empty(), "Test dataset is empty."

    print(f"Train beats: {train_ds.n_beats}")
    print(f"Val beats:   {val_ds.n_beats}")
    print(f"Test beats:  {test_ds.n_beats}")

    print("PASS: All splits contain data.")

    # ---------------------------------------------------------
    # 4. Check waveform shape
    # ---------------------------------------------------------
    print("\n[4] Checking waveform shape...")

    expected_window = cfg["preprocessing"]["beat_window_size"]

    assert train_ds.waveforms.shape[1] == expected_window
    assert val_ds.waveforms.shape[1] == expected_window
    assert test_ds.waveforms.shape[1] == expected_window

    print(f"Expected window size: {expected_window}")
    print(f"Train shape: {train_ds.waveforms.shape}")
    print(f"Val shape:   {val_ds.waveforms.shape}")
    print(f"Test shape:  {test_ds.waveforms.shape}")

    print("PASS: All waveform windows have correct size.")

    # ---------------------------------------------------------
    # 5. Check finite values
    # ---------------------------------------------------------
    print("\n[5] Checking for NaN / Inf values...")

    assert np.isfinite(train_ds.waveforms).all()
    assert np.isfinite(val_ds.waveforms).all()
    assert np.isfinite(test_ds.waveforms).all()

    print("PASS: All waveform values are finite.")

    # ---------------------------------------------------------
    # 6. Check per-beat normalization
    # ---------------------------------------------------------
    print("\n[6] Checking per-beat normalization...")

    for name, ds in [
        ("train", train_ds),
        ("val", val_ds),
        ("test", test_ds),
    ]:
        means = np.mean(ds.waveforms, axis=1)
        stds = np.std(ds.waveforms, axis=1)

        assert np.allclose(means, 0.0, atol=1e-5), (
            f"{name}: normalization mean check failed."
        )

        assert np.allclose(stds, 1.0, atol=1e-5), (
            f"{name}: normalization std check failed."
        )

        print(
            f"{name}: mean≈{np.mean(means):.6f}, "
            f"std≈{np.mean(stds):.6f}"
        )

    print("PASS: Per-beat z-score normalization verified.")

    # ---------------------------------------------------------
    # 7. Check record-level integrity
    # ---------------------------------------------------------
    print("\n[7] Checking record-level integrity...")

    verify_record_level_integrity(
        train_ds,
        val_ds,
        test_ds,
    )

    print("PASS: No record appears in multiple splits.")

    # ---------------------------------------------------------
    # 8. Print class distributions
    # ---------------------------------------------------------
    print("\n[8] Class distributions")

    print("\nTrain:")
    print(train_ds.class_distribution)

    print("\nValidation:")
    print(val_ds.class_distribution)

    print("\nTest:")
    print(test_ds.class_distribution)

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("ALL SPLITTING PIPELINE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()