@"
# NeuroChaos vs Standard Deep Learning Models for Cardiac Arrhythmia Detection

> **Research / educational project — not a clinical diagnostic system.**
>
> This project experimentally compares classical machine learning, deep learning, nonlinear-dynamics (NeuroChaos), and hybrid approaches for ECG arrhythmia classification using the MIT-BIH Arrhythmia Database.

---

## Overview

This project investigates whether nonlinear-dynamics features extracted from ECG beats provide complementary information to features learned directly from ECG waveforms by neural networks.

The pipeline includes:

- ECG loading and validation
- Band-pass filtering and normalization
- R-peak-based beat segmentation
- Fixed record-wise train/validation/test splitting
- Classical ML baselines
- 1D CNN
- Bi-LSTM
- NeuroChaos feature extraction
- Hybrid CNN + NeuroChaos model
- Ablation study
- Noise robustness evaluation
- Streamlit visualization dashboard

### Model branches

| Branch | Approach |
|---|---|
| Classical ML | Logistic Regression, Random Forest, SVM |
| Deep Learning | 1D CNN, Bi-LSTM |
| NeuroChaos | Nonlinear-dynamics features + classical classifiers |
| Hybrid | CNN embedding + NeuroChaos features → MLP |

---

## Dataset

The project uses the **MIT-BIH Arrhythmia Database**, containing 48 half-hour two-lead ambulatory ECG recordings sampled at 360 Hz.

Dataset source: [PhysioNet — MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)

The raw dataset is **not included in this repository**. It is downloaded locally when required.

### Dataset split

The project uses a fixed **record-wise split**:

| Split | Records | Beats |
|---|---:|---:|
| Train | 33 | 78,419 |
| Validation | 9 | 18,975 |
| Test | 6 | 12,054 |
| **Total** | **48** | **109,448** |

> Beats from the same ECG record are kept within the same split.

### AAMI 5-class mapping

| Class | Description | MIT-BIH symbols |
|---|---|---|
| N | Normal / near-normal | N, L, R, e, j |
| S | Supraventricular ectopic | A, a, J, S |
| V | Ventricular ectopic | V, E |
| F | Fusion beat | F |
| Q | Unknown / paced | /, f, Q |

---

## Preprocessing

Each ECG beat is processed using:

- 0.5–40 Hz Butterworth band-pass filtering
- R-peak-centered segmentation
- 108 samples before the R-peak
- 180 samples after the R-peak
- 288 samples per beat
- Per-beat z-score normalization
- Boundary beats discarded when a complete window is unavailable

---

## NeuroChaos Features

The NeuroChaos branch extracts 13 nonlinear-dynamics features:

1. Sample Entropy
2. Permutation Entropy
3. Approximate Entropy
4. Recurrence Rate
5. Determinism
6. Laminarity
7. Largest Lyapunov Exponent
8. Phase-space mean velocity
9. Phase-space trajectory length
10. Log determinant of covariance
11. DFA alpha
12. Hurst exponent
13. Higuchi fractal dimension

Features are extracted separately from the ECG beat waveform and combined with neural representations in the hybrid model.

---

## Experiments

### Classical ML

The NeuroChaos feature representation was evaluated using:

- Logistic Regression
- Random Forest
- Support Vector Machine

Class balancing was applied during training.

### Deep Learning

Two waveform-based neural models were evaluated:

- 1D CNN
- Bi-LSTM

### Hybrid Model

The hybrid architecture combines:

**ECG waveform → CNN representation**

with

**ECG waveform → NeuroChaos features**

and feeds the combined representation into an MLP classifier.

### Ablation Study

The contribution of different nonlinear feature groups was evaluated using:

- Neural-only
- Neural + entropy
- Neural + recurrence
- Neural + Lyapunov
- Neural + fractal
- Neural + all NeuroChaos features

### Noise Robustness

Models were evaluated on the same test samples under additive Gaussian noise at:

- Clean
- Low noise: σ = 0.05
- Medium noise: σ = 0.15
- High noise: σ = 0.30

---

## Selected Results

Results below are from the fixed test split.

| Model | Accuracy | Macro F1 | Balanced Accuracy |
|---|---:|---:|---:|
| 1D CNN | 0.2739 | 0.1583 | 0.3593 |
| Bi-LSTM | 0.2321 | 0.2032 | 0.3989 |
| Hybrid CNN + NeuroChaos | 0.2116 | 0.1371 | 0.4614 |

Because the test distribution is highly imbalanced, accuracy alone is not sufficient for interpreting performance. Macro F1 and balanced accuracy are therefore also reported.

### NeuroChaos ablation

| Condition | Accuracy | Macro F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Neural only | 0.2739 | 0.1583 | 0.3593 |
| Neural + Entropy | 0.2123 | 0.1250 | 0.4443 |
| Neural + Recurrence | 0.1907 | 0.1216 | 0.3209 |
| Neural + Lyapunov | 0.2984 | 0.1793 | 0.4490 |
| Neural + Fractal | 0.1782 | 0.1078 | 0.4472 |
| Neural + All Chaos | 0.2116 | 0.1371 | 0.4614 |

These results are intended as an experimental comparison rather than a clinical performance claim.

---

## Noise Robustness

### 1D CNN

| Noise | Accuracy | Macro F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Clean | 0.2739 | 0.1583 | 0.3593 |
| Low | 0.3085 | 0.1778 | 0.3480 |
| Medium | 0.3666 | 0.2033 | 0.3121 |
| High | 0.2344 | 0.1036 | 0.2592 |

### Bi-LSTM

| Noise | Accuracy | Macro F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Clean | 0.2321 | 0.2032 | 0.3989 |
| Low | 0.2304 | 0.2017 | 0.3964 |
| Medium | 0.2163 | 0.1904 | 0.3766 |
| High | 0.1917 | 0.1700 | 0.3437 |

### Hybrid CNN + NeuroChaos

| Noise | Accuracy | Macro F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Clean | 0.2119 | 0.1374 | 0.4617 |
| Low | 0.1219 | 0.0599 | 0.1545 |
| Medium | 0.0826 | 0.0543 | 0.1523 |
| High | 0.1444 | 0.1040 | 0.2636 |

---

## Project Structure

```text
neurochaos-arrythmia/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── data/
│   │   ├── annotations.py
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   └── segmentation.py
│   │
│   ├── neurochaos/
│   │   ├── extractor.py
│   │   └── feature_cache.py
│   │
│   ├── models/
│   │
│   ├── training/
│   │   ├── train_classical.py
│   │   ├── train_cnn.py
│   │   ├── train_bilstm.py
│   │   ├── train_hybrid.py
│   │   ├── ablation.py
│   │   ├── run_robustness.py
│   │   └── resume_bilstm.py
│   │
│   └── utils/
│
├── scripts/
├── tests/
├── config.yaml
├── requirements.txt
├── profile_neurochaos.py
├── test_resume.py
├── README.md
└── .gitignore

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd neurochaos-arrythmia