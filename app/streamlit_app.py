"""
app/streamlit_app.py
---------------------
NeuroChaos vs Standard Deep Learning — Research Dashboard

⚠ DISCLAIMER
This application is for RESEARCH AND EDUCATIONAL DEMONSTRATION ONLY.
It is not a medical device, has not been validated for clinical use,
and must not be used to guide any clinical or diagnostic decisions.

Pages
-----
1. Home               — project overview and disclaimer
2. ECG Explorer       — load a record, view signal, beat segmentation
3. NeuroChaos Features — visualise chaos features for individual beats
4. Model Comparison   — side-by-side metric tables from saved experiments
5. Confusion Matrices — view saved confusion matrix images
6. ROC / PR Curves    — view saved ROC and PR curve images
7. Ablation Study     — ablation results table and bar chart
8. Noise Robustness   — robustness bar chart per model
9. About              — dataset, annotation mapping, reproducibility notes

Usage
-----
From project root (venv activated):
    streamlit run app/streamlit_app.py

Requires
--------
    streamlit >= 1.35
    plotly    >= 5.22
    All src/ modules importable (run from project root, or set PYTHONPATH)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

# ── Ensure project root on path ────────────────────────────────────────────
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="NeuroChaos Arrhythmia Research",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading configuration ...")
def _load_cfg():
    from src.utils.config import load_config
    return load_config()


@st.cache_resource(show_spinner="Loading record ...")
def _load_record(record_id: int, data_dir: str, fs: int, lead: int):
    from src.data.loader import load_record
    return load_record(record_id, data_dir=data_dir, fs=fs, lead=lead, download=True)


@st.cache_data(show_spinner="Preprocessing signal ...")
def _preprocess(_signal: np.ndarray, fs: int, lowcut: float, highcut: float, order: int, record_id: str):
    from src.data.preprocessing import preprocess_signal
    return preprocess_signal(_signal, fs=fs, lowcut=lowcut, highcut=highcut,
                             order=order, record_id=record_id)


@st.cache_data(show_spinner="Segmenting beats ...")
def _segment(_signal: np.ndarray, _beat_samples, _beat_symbols, _aami_classes,
             _aami_labels, record_id: str, before: int, after: int):
    """Wrap segment_record with hashable args."""
    import dataclasses
    from src.data.loader import ECGRecord
    from src.data.segmentation import segment_record
    rec = ECGRecord(
        record_id=record_id, signal=_signal, fs=360,
        beat_samples=_beat_samples, beat_symbols=list(_beat_symbols),
        aami_classes=list(_aami_classes), aami_labels=_aami_labels,
        n_samples=len(_signal), n_beats=len(_beat_samples),
        lead_name="MLII",
    )
    return segment_record(rec, before=before, after=after, allow_padding=False)


@st.cache_data(show_spinner="Extracting NeuroChaos features ...")
def _extract_chaos(_waveforms: np.ndarray, _cfg_hash: str):
    from src.utils.config import load_config
    from src.neurochaos.extractor import NeuroChaosExtractor, NeuroChaosImputer
    cfg = load_config()
    extractor = NeuroChaosExtractor(cfg)
    X_raw = extractor.extract_dataset(_waveforms, show_progress=False)
    imputer = NeuroChaosImputer()
    return extractor, imputer.fit_transform(X_raw), X_raw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _experiments_dir() -> Path:
    try:
        cfg = _load_cfg()
        return Path(cfg["output"]["experiments_dir"])
    except Exception:
        return PROJECT_ROOT / "results" / "experiments"


def _find_json_files(pattern: str) -> list[Path]:
    exp_dir = _experiments_dir()
    return sorted(exp_dir.rglob(pattern), reverse=True)


def _load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _metric_table(metrics: dict) -> dict:
    """Extract the key scalar metrics for display."""
    return {
        "Accuracy":          round(metrics.get("accuracy",          0.0), 4),
        "Balanced Accuracy": round(metrics.get("balanced_accuracy", 0.0), 4),
        "Macro F1":          round(metrics.get("macro_f1",          0.0), 4),
        "Weighted F1":       round(metrics.get("weighted_f1",       0.0), 4),
        "MCC":               round(metrics.get("mcc",               0.0), 4),
        "ROC-AUC (macro)":   round(metrics.get("roc_auc_macro") or 0.0,   4),
    }


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def page_home() -> None:
    st.title("💓 NeuroChaos vs Standard Deep Learning for Arrhythmia Detection")

    st.error(
        "⚠ **Research / Educational Demonstration Only.** "
        "This application has not been validated for clinical use and must "
        "not be used to guide medical decisions.",
        icon="⚠",
    )

    st.markdown("""
## Project Overview

This dashboard accompanies an experimental study comparing three approaches to
ECG arrhythmia classification on the
[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/):

| Branch | Models |
|---|---|
| **Classical ML** | Logistic Regression, Random Forest, SVM |
| **Deep Learning** | 1D CNN, Bidirectional LSTM |
| **NeuroChaos** | 13 nonlinear-dynamics features + ML classifier |
| **Hybrid** | Neural embedding ⊕ NeuroChaos features → MLP |

### AAMI 5-Class Scheme (ANSI/AAMI EC57:2012)

| Index | Label | Description |
|---|---|---|
| 0 | **N** | Normal / near-normal |
| 1 | **S** | Supraventricular ectopic |
| 2 | **V** | Ventricular ectopic |
| 3 | **F** | Fusion beat |
| 4 | **Q** | Unknown / paced |

### Dataset Split (record-wise)

| Split | Records | Beats |
|---|---:|---:|
| Train | 33 | 78,419 |
| Val | 9 | 18,975 |
| Test | 6 | 12,054 |
| **Total** | **48** | **109,448** |

> Beats from the same record never appear in both train and test.
""")

    with st.expander("NeuroChaos Feature Set (13 features)"):
        st.markdown("""
| # | Feature | Family |
|---|---|---|
| 0 | Sample Entropy | Entropy |
| 1 | Permutation Entropy | Entropy |
| 2 | Approximate Entropy | Entropy |
| 3 | Recurrence Rate (RR) | RQA |
| 4 | Determinism (DET) | RQA |
| 5 | Laminarity (LAM) | RQA |
| 6 | Largest Lyapunov Exponent | Lyapunov |
| 7 | Phase-Space Mean Velocity | Phase Space |
| 8 | Phase-Space Trajectory Length | Phase Space |
| 9 | Attractor log-det Covariance | Phase Space |
| 10 | DFA Alpha exponent | Fractal |
| 11 | Hurst Exponent | Fractal |
| 12 | Higuchi Fractal Dimension | Fractal |
""")


def page_ecg_explorer() -> None:
    st.title("📈 ECG Signal Explorer")
    st.caption("Explore raw and filtered ECG signals and their beat segmentation.")

    try:
        cfg = _load_cfg()
    except Exception as e:
        st.error(f"Config load failed: {e}")
        return

    import plotly.graph_objects as go  # type: ignore

    ds_cfg = cfg["dataset"]
    pp_cfg = cfg["preprocessing"]
    bp_cfg = pp_cfg["bandpass"]

    col1, col2 = st.columns([1, 3])
    with col1:
        all_records = [int(r) for r in ds_cfg["all_records"]]
        record_id   = st.selectbox("Record", all_records, index=0)
        show_filter = st.checkbox("Show filtered signal", value=True)
        max_secs    = st.slider("Display window (seconds)", 2, 30, 10)

    with col2:
        with st.spinner(f"Loading record {record_id} ..."):
            try:
                rec = _load_record(
                    record_id, ds_cfg["raw_dir"],
                    fs=ds_cfg["fs"], lead=ds_cfg["lead"],
                )
            except Exception as e:
                st.error(f"Could not load record {record_id}: {e}")
                return

        fs      = rec.fs
        n_show  = min(int(max_secs * fs), rec.n_samples)
        t       = np.arange(n_show) / fs
        raw_sig = rec.signal[:n_show]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=raw_sig, mode="lines",
                                 line=dict(color="#90CAF9", width=1),
                                 name="Raw"))

        if show_filter:
            proc_sig = _preprocess(
                rec.signal, fs=fs,
                lowcut=bp_cfg["lowcut"], highcut=bp_cfg["highcut"],
                order=bp_cfg["order"], record_id=str(record_id),
            )
            fig.add_trace(go.Scatter(x=t, y=proc_sig[:n_show], mode="lines",
                                     line=dict(color="#2196F3", width=1.2),
                                     name="Filtered"))

        # Mark R-peaks in the window
        peaks_in_window = rec.beat_samples[rec.beat_samples < n_show]
        peak_times      = peaks_in_window / fs
        peak_vals       = raw_sig[peaks_in_window]
        fig.add_trace(go.Scatter(
            x=peak_times, y=peak_vals,
            mode="markers",
            marker=dict(symbol="x", color="#F44336", size=8),
            name="R-peak",
        ))

        fig.update_layout(
            title=f"Record {record_id} — Lead {rec.lead_name}  "
                  f"(fs={fs} Hz, {n_show} samples shown)",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (mV)",
            height=350,
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
**Record info:** `{rec.n_samples}` samples · `{rec.n_beats}` annotated beats  
**Class distribution:** {rec.class_distribution}
""")


def page_neurochaos_features() -> None:
    st.title("🌀 NeuroChaos Feature Visualisation")
    st.caption(
        "Extract and inspect the 13 nonlinear-dynamics features for individual "
        "beats from any MIT-BIH record."
    )

    try:
        cfg = _load_cfg()
    except Exception as e:
        st.error(f"Config load failed: {e}")
        return

    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    ds_cfg = cfg["dataset"]
    pp_cfg = cfg["preprocessing"]
    bp_cfg = pp_cfg["bandpass"]

    from src.neurochaos.extractor import FEATURE_NAMES
    from src.data.annotations import AAMI_CLASS_DESCRIPTIONS

    col1, col2 = st.columns([1, 3])
    with col1:
        record_id = st.selectbox(
            "Record", [int(r) for r in ds_cfg["all_records"]], index=0,
            key="chaos_record",
        )
        max_beats = st.slider("Beats to analyse", 10, 200, 50)

    with col2:
        with st.spinner(f"Loading record {record_id} ..."):
            try:
                rec = _load_record(
                    record_id, ds_cfg["raw_dir"],
                    fs=ds_cfg["fs"], lead=ds_cfg["lead"],
                )
            except Exception as e:
                st.error(str(e)); return

        proc_signal = _preprocess(
            rec.signal, fs=ds_cfg["fs"],
            lowcut=bp_cfg["lowcut"], highcut=bp_cfg["highcut"],
            order=bp_cfg["order"], record_id=str(record_id),
        )

        ds = _segment(
            proc_signal,
            rec.beat_samples, rec.beat_symbols,
            rec.aami_classes, rec.aami_labels,
            record_id=str(record_id),
            before=pp_cfg["beat_before_samples"],
            after=pp_cfg["beat_after_samples"],
        )

        if ds.is_empty():
            st.warning("No beats could be segmented from this record.")
            return

        n = min(max_beats, ds.n_beats)
        waveforms_sub = ds.waveforms[:n]
        labels_sub    = ds.labels[:n]
        classes_sub   = ds.aami_classes[:n]

        cfg_hash = str(sorted(cfg["neurochaos"].items()))
        _, X_imp, X_raw = _extract_chaos(waveforms_sub, cfg_hash)

        df = pd.DataFrame(X_imp, columns=FEATURE_NAMES)
        df["class"] = classes_sub

        st.success(f"Extracted {n} beats · {int(np.isnan(X_raw).sum())} raw NaN values "
                   f"({100*np.isnan(X_raw).mean():.1f}%)")

        tab1, tab2, tab3 = st.tabs(["Feature Heatmap", "Class Means", "Per-beat waveform"])

        with tab1:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imp)
            fig = px.imshow(
                X_scaled.T,
                x=[f"Beat {i}" for i in range(n)],
                y=FEATURE_NAMES,
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                title=f"Record {record_id} — NeuroChaos Feature Heatmap (z-scored)",
                aspect="auto",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            class_means = df.groupby("class")[FEATURE_NAMES].mean().T
            fig2 = px.bar(
                class_means,
                barmode="group",
                title="Mean Feature Value per AAMI Class",
                labels={"index": "Feature", "value": "Mean value", "variable": "Class"},
                height=400,
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            beat_idx = st.slider("Beat index", 0, n - 1, 0)
            cls_name = classes_sub[beat_idx]
            cls_desc = AAMI_CLASS_DESCRIPTIONS.get(cls_name, cls_name)
            t_beat   = np.arange(pp_cfg["beat_window_size"])
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=t_beat, y=waveforms_sub[beat_idx],
                mode="lines", line=dict(color="#2196F3", width=1.5),
                name=f"Beat {beat_idx}",
            ))
            fig3.add_vline(x=pp_cfg["beat_before_samples"],
                           line_dash="dash", line_color="red",
                           annotation_text="R-peak")
            fig3.update_layout(
                title=f"Beat {beat_idx} — Class {cls_name}: {cls_desc}",
                xaxis_title="Sample", yaxis_title="Amplitude (normalised)",
                height=300,
            )
            st.plotly_chart(fig3, use_container_width=True)

            feat_vals = {name: float(X_imp[beat_idx, i])
                         for i, name in enumerate(FEATURE_NAMES)}
            st.json(feat_vals)


def page_model_comparison() -> None:
    st.title("📊 Model Comparison")
    st.caption("Side-by-side metrics from saved experiment JSON files.")

    metrics_files = _find_json_files("metrics_test.json")
    if not metrics_files:
        st.info(
            "No experiment results found yet.  "
            "Run one or more training scripts first:\n\n"
            "```\npython -m src.training.train_cnn\n"
            "python -m src.training.train_classical --features waveform\n"
            "python -m src.training.train_hybrid\n```"
        )
        return

    import pandas as pd

    rows = []
    for path in metrics_files:
        try:
            data = _load_json(path)
            exp_name = path.parent.name
            row = {"Experiment": exp_name}
            row.update(_metric_table(data))
            rows.append(row)
        except Exception:
            continue

    if not rows:
        st.warning("Could not parse any metrics files.")
        return

    df = pd.DataFrame(rows).set_index("Experiment")
    st.dataframe(df.style.highlight_max(axis=0, color="#C8E6C9"), use_container_width=True)

    import plotly.express as px
    metric_col = st.selectbox(
        "Compare by", ["Macro F1", "Balanced Accuracy", "Accuracy", "Weighted F1"],
    )
    df_reset = df.reset_index()
    fig = px.bar(
        df_reset, x="Experiment", y=metric_col,
        color="Experiment", title=f"Model Comparison — {metric_col}",
        height=400,
    )
    fig.update_layout(xaxis_tickangle=-30, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def page_confusion_matrices() -> None:
    st.title("🧮 Confusion Matrices")

    png_files = sorted(
        _experiments_dir().rglob("confusion_test.png"), reverse=True
    )
    if not png_files:
        st.info("No confusion matrix images found.  Run a training script first.")
        return

    exp_options = {p.parent.name: p for p in png_files}
    selected    = st.selectbox("Experiment", list(exp_options.keys()))
    path        = exp_options[selected]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Test set")
        st.image(str(path), width=700)

    val_path = path.parent / "confusion_val.png"
    with col2:
        st.subheader("Validation set")
        if val_path.exists():
            st.image(str(path), width=700)
        else:
            st.info("Val confusion matrix not found.")


def page_roc_pr_curves() -> None:
    st.title("📈 ROC & Precision-Recall Curves")

    roc_files = sorted(_experiments_dir().rglob("roc_test.png"), reverse=True)
    if not roc_files:
        st.info("No ROC curve images found.  Run a training script first.")
        return

    exp_options = {p.parent.name: p for p in roc_files}
    selected    = st.selectbox("Experiment", list(exp_options.keys()), key="roc_sel")
    roc_path    = exp_options[selected]
    pr_path     = roc_path.parent / "pr_test.png"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curves (test)")
        st.image(str(roc_path), width=700)

    with col2:
        st.subheader("Precision-Recall Curves (test)")
        if pr_path.exists():
            st.image(str(pr_path), width=700)
        else:
            st.info("PR curve image not found.")

def page_ablation() -> None:
    st.title("🔬 Ablation Study")
    st.caption("Which NeuroChaos feature groups contribute most?")

    summary_files = sorted(
        _experiments_dir().rglob("ablation_summary.json"), reverse=True
    )
    if not summary_files:
        st.info(
            "No ablation results found yet.  Run the ablation script:\n\n"
            "```\npython -m src.training.ablation\n```"
        )
        return

    import pandas as pd

    selected_file = st.selectbox(
        "Ablation run", [str(p.parent.name) for p in summary_files],
    )
    data = _load_json(
        next(p for p in summary_files
             if p.parent.name == selected_file)
    )

    rows = []
    for cond_name, cond_data in data.items():
        m = cond_data.get("test_metrics", {})
        rows.append({
            "Condition":          cond_name,
            "n Chaos features":   cond_data.get("n_chaos_features", 0),
            "Features used":      ", ".join(cond_data.get("feature_names", [])) or "—",
            "Accuracy":           round(m.get("accuracy", 0), 4),
            "Balanced Accuracy":  round(m.get("balanced_accuracy", 0), 4),
            "Macro F1":           round(m.get("macro_f1", 0), 4),
            "Weighted F1":        round(m.get("weighted_f1", 0), 4),
            "Train time (s)":     cond_data.get("train_seconds", "—"),
        })

    df = pd.DataFrame(rows).set_index("Condition")
    st.dataframe(
        df.style.highlight_max(subset=["Macro F1", "Balanced Accuracy"], color="#C8E6C9"),
        use_container_width=True,
    )

    # Show bar chart image if it exists
    chart_path = Path(str(summary_files[0].parent)) / "ablation_macro_f1.png"
    if chart_path.exists():
        st.image(str(chart_path),caption="Macro F1 by ablation condition",width=700,)


def page_robustness() -> None:
    st.title("🔊 Noise Robustness")
    st.caption("Model performance vs controlled additive Gaussian noise levels.")

    rob_files = sorted(
        _experiments_dir().rglob("robustness_*.json"), reverse=True
    )
    if not rob_files:
        st.info(
            "No robustness results found yet.  Run the robustness script:\n\n"
            "```\npython -m src.training.run_robustness "
            "--model cnn --checkpoint <path>\n```"
        )
        return

    import pandas as pd
    import plotly.express as px

    file_options = {p.stem: p for p in rob_files}
    selected     = st.selectbox("Result file", list(file_options.keys()))
    data         = _load_json(file_options[selected])

    model_name = data.get("model_name", selected)
    st.markdown(f"**Model:** `{model_name}` — "
                f"**Test beats:** {data.get('n_test_beats', '?')}")

    rows = []
    for level, info in data["noise_levels"].items():
        m = info["metrics"]
        rows.append({
            "Noise Level":        level,
            "σ factor":           info["sigma_factor"],
            "Accuracy":           round(m.get("accuracy", 0), 4),
            "Balanced Accuracy":  round(m.get("balanced_accuracy", 0), 4),
            "Macro F1":           round(m.get("macro_f1", 0), 4),
            "Weighted F1":        round(m.get("weighted_f1", 0), 4),
        })

    df = pd.DataFrame(rows).set_index("Noise Level")
    st.dataframe(df.style.background_gradient(
        subset=["Macro F1"], cmap="RdYlGn"), use_container_width=True)

    fig = px.line(
        df.reset_index(), x="Noise Level", y="Macro F1",
        markers=True,
        title=f"Robustness: {model_name} — Macro F1 vs noise level",
        height=350,
    )
    fig.update_traces(line=dict(color="#2196F3", width=2),
                      marker=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)

    chart_path = file_options[selected].parent / f"robustness_{model_name}.png"
    if chart_path.exists():
        st.image(str(chart_path), width=700)


def page_about() -> None:
    st.title("ℹ About this Project")

    st.markdown("""
## Dataset

**MIT-BIH Arrhythmia Database** — PhysioNet, version 1.0.0  
48 half-hour two-lead ambulatory ECG recordings at 360 Hz.  
URL: https://physionet.org/content/mitdb/1.0.0/

## Annotation Mapping

ANSI/AAMI EC57:2012 five-class scheme applied to wfdb beat annotations.

**Records with unmapped symbols (skipped, NOT remapped):**

| Record | Symbol | Reason |
|---|---|---|
| 214, 215, 219, 228, 231 | `r` | RBBB variant not in standard AAMI EC57 mapping |

These symbols were excluded from the dataset.  The mapping was NOT
changed to accommodate them because doing so requires a scientific
justification and peer review beyond the scope of this study.

## Reproducibility

- All random seeds: `seed: 42` (see `config.yaml`)
- Record-wise split fixed in `config.yaml` (never computed at runtime)
- Per-beat z-score normalisation (no dataset-wide statistics → no leakage)
- Every experiment saves a timestamped manifest and metrics JSON

## Disclaimer

This application is for **research and educational demonstration only**.
It has not been validated for clinical use and must not be used to guide
any medical decisions.  All results are experimental, subject to the
limitations of the MIT-BIH dataset, and have not been peer-reviewed.
""")

    st.subheader("Run order")
    st.code("""
# 1. Download dataset (auto on first run, or explicit)
python scripts/download_mitdb.py

# 2. Verify splitting pipeline
python tests/test_splitting.py

# 3. Verify NeuroChaos modules
python tests/test_neurochaos_phase3.py

# 4. Train models (each independent)
python -m src.training.train_classical --features waveform
python -m src.training.train_classical --features neurochaos
python -m src.training.train_cnn
python -m src.training.train_bilstm
python -m src.training.train_hybrid --backbone cnn

# 5. Ablation study
python -m src.training.ablation

# 6. Robustness (requires a trained checkpoint path)
python -m src.training.run_robustness --model cnn --checkpoint <path>

# 7. Explainability
python -m src.training.explainability

# 8. Launch this dashboard
streamlit run app/streamlit_app.py
""", language="bash")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = {
    "🏠 Home":                    page_home,
    "📈 ECG Explorer":            page_ecg_explorer,
    "🌀 NeuroChaos Features":     page_neurochaos_features,
    "📊 Model Comparison":        page_model_comparison,
    "🧮 Confusion Matrices":      page_confusion_matrices,
    "📈 ROC / PR Curves":         page_roc_pr_curves,
    "🔬 Ablation Study":          page_ablation,
    "🔊 Noise Robustness":        page_robustness,
    "ℹ About":                   page_about,
}


def main() -> None:
    with st.sidebar:
        st.markdown("## 🫀 Navigation")
        selected = st.radio("Navigation",list(PAGES.keys()),label_visibility="collapsed",)
        st.markdown("---")
        st.caption(
            "⚠ Research / Educational Only\n\n"
            "Not a medical device. Not validated for clinical use."
        )

    PAGES[selected]()


if __name__ == "__main__":
    main()
