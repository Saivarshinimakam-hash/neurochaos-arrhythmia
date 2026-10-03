import time
import numpy as np

from src.neurochaos.entropy import (
    sample_entropy,
    permutation_entropy,
    approximate_entropy,
)

from src.neurochaos.recurrence import compute_rqa_features
from src.neurochaos.lyapunov import compute_lyapunov_features
from src.neurochaos.phase_space import compute_phase_space_features

from src.neurochaos.fractal import (
    dfa_alpha,
    hurst_exponent,
    higuchi_fractal_dimension,
)


N_BEATS = 100
WINDOW = 288

rng = np.random.default_rng(42)
waveforms = rng.normal(
    size=(N_BEATS, WINDOW)
).astype(np.float64)


def benchmark(name, func):
    # Warm-up: exclude one-time initialization / compilation.
    func(waveforms[0])

    start = time.perf_counter()

    for waveform in waveforms:
        func(waveform)

    elapsed = time.perf_counter() - start

    print(
        f"{name:30s} "
        f"{elapsed:8.3f} sec total | "
        f"{elapsed / N_BEATS:10.6f} sec/beat"
    )


print("=" * 80)
print("NeuroChaos Steady-State Feature Benchmark")
print(f"Number of beats : {N_BEATS}")
print(f"Window size     : {WINDOW}")
print("Warm-up call excluded from timing")
print("=" * 80)

benchmark("Sample Entropy", sample_entropy)

benchmark(
    "Permutation Entropy",
    lambda x: permutation_entropy(
        x, order=3, delay=1, normalize=True
    ),
)

benchmark("Approximate Entropy", approximate_entropy)

benchmark("RQA", compute_rqa_features)

benchmark("Lyapunov", compute_lyapunov_features)

benchmark("Phase Space", compute_phase_space_features)

benchmark("DFA", dfa_alpha)

benchmark("Hurst", hurst_exponent)

benchmark("Higuchi FD", higuchi_fractal_dimension)

print("=" * 80)
print("Benchmark complete.")