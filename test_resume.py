from pathlib import Path
import numpy as np

from src.utils.config import load_config
from src.data.splitting import build_datasets
from src.neurochaos.extractor import NeuroChaosExtractor
from src.neurochaos.feature_cache import (
    save_checkpoint,
    load_feature_cache,
)

cfg = load_config()

print("Loading dataset...")
train_ds, _, _ = build_datasets(
    cfg,
    download=False,
    normalize=True,
    save_manifest=False,
)

extractor = NeuroChaosExtractor(cfg)

checkpoint_path = (
    Path(cfg["output"]["cache_dir"])
    / "neurochaos"
    / "resume_test.npz"
)

# ---------------------------------------------------------
# Phase 1: simulate first extraction chunk
# ---------------------------------------------------------
first_n = 10

print(f"\nPhase 1: extracting first {first_n} beats...")

features_1 = extractor.extract_dataset(
    train_ds.waveforms[:first_n],
    show_progress=True,
)

save_checkpoint(
    path=checkpoint_path,
    features=features_1,
    labels=train_ds.labels[:first_n],
    record_ids=list(train_ds.record_ids[:first_n]),
    beat_indices=train_ds.beat_indices[:first_n],
    r_peak_samples=train_ds.r_peak_samples[:first_n],
    split_name="train",
    cfg=cfg,
    completed=first_n,
)

print("Checkpoint created.")
print("Simulating program interruption...\n")


# ---------------------------------------------------------
# Phase 2: load checkpoint and resume
# ---------------------------------------------------------
checkpoint = load_feature_cache(
    path=checkpoint_path,
    split_name="train",
    cfg=cfg,
)

completed = len(checkpoint["features"])

print(f"Resuming from {completed} beats...")

assert completed == first_n

next_n = 20

features_2 = extractor.extract_dataset(
    train_ds.waveforms[completed:next_n],
    show_progress=True,
)

combined = np.concatenate(
    [
        checkpoint["features"],
        features_2,
    ],
    axis=0,
)

print("\nResume completed.")
print("First checkpoint shape:", checkpoint["features"].shape)
print("New chunk shape:", features_2.shape)
print("Combined shape:", combined.shape)

assert combined.shape == (20, 13)

print("\nRESUME TEST: PASS")

# Cleanup
checkpoint_path.unlink()

print("Temporary checkpoint removed.")