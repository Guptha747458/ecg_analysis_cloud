"""Cloud ECG Analysis System - Streamlit application."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ecg_analysis import analyze_ecg, read_ecg_csv, read_wfdb_record
from pdf_report import create_pdf_report


st.set_page_config(
    page_title="Cloud ECG Analysis System",
    page_icon="♥",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1200px; padding-top: 2rem; }
        .hero { padding: 1.6rem 1.8rem; border-radius: 18px;
                background: linear-gradient(135deg, #0b5563, #17807d);
                color: white; margin-bottom: 1.5rem; }
        .hero h1 { margin: 0; font-size: 2.2rem; }
        .hero p { margin: .5rem 0 0; opacity: .9; }
        .status-card { padding: 1.2rem; border-radius: 14px; color: white;
                       margin: .7rem 0 1rem; }
        .normal { background: #18866b; }
        .abnormal { background: #c84b4b; }
        .muted { color: #60717a; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def waveform_figure(result: dict[str, object], seconds: float) -> go.Figure:
    signal = np.asarray(result["signal"], dtype=float)
    sample_rate = float(result["sample_rate"])
    limit = min(len(signal), max(1, int(seconds * sample_rate)))
    step = max(1, limit // 5000)
    values = signal[:limit:step]
    time = np.arange(0, limit, step) / sample_rate
    peaks = np.asarray(result["peaks"], dtype=int)
    peaks = peaks[peaks < limit]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=time,
            y=values,
            mode="lines",
            name="ECG signal",
            line={"color": "#087f8c", "width": 1.4},
        )
    )
    if len(peaks):
        fig.add_trace(
            go.Scatter(
                x=peaks / sample_rate,
                y=signal[peaks],
                mode="markers",
                name="Detected beats",
                marker={"color": "#d64b4b", "size": 7},
            )
        )
    fig.update_layout(
        height=430,
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def show_home() -> None:
    st.markdown(
        '<div class="hero"><h1>Cloud ECG Analysis System</h1>'
        "<p>A lightweight, privacy-conscious ECG screening workspace.</p></div>",
        unsafe_allow_html=True,
    )
    st.subheader("Analyze an ECG in three steps")
    columns = st.columns(3)
    for column, title, text in zip(
        columns,
        ("1. Upload", "2. Analyze", "3. Report"),
        (
            "Upload a CSV containing one or more numeric ECG signal columns.",
            "Explore an interactive waveform and run transparent signal checks.",
            "Review the result and download a PDF summary.",
        ),
    ):
        column.markdown(f"### {title}")
        column.write(text)
    st.info(
        "This educational screening tool is not a medical device and does not "
        "provide a diagnosis. Consult a qualified clinician for medical advice."
    )
    st.subheader("Why cloud processing?")
    st.write(
        "Streamlit Cloud provides centralized hosting, remote access, and "
        "on-demand processing without requiring local software installation. "
        "Uploaded bytes are processed in memory and are not written to a "
        "database or retained as personal information."
    )
    st.subheader("CSV instructions")
    st.write(
        "Use one numeric sample per row. You can also upload a complete "
        "MIT-BIH/WFDB record (.hea + .dat, with optional .atr annotations)."
    )


def show_upload() -> None:
    st.header("Upload ECG data")
    format_choice = st.radio("Input format", ["CSV", "MIT-BIH / WFDB"], horizontal=True)
    if format_choice == "CSV":
        show_csv_upload()
    else:
        show_wfdb_upload()


def show_csv_upload() -> None:
    uploaded = st.file_uploader(
        "Choose an ECG CSV file", type=["csv"],
        help="CSV files must contain at least 100 numeric ECG samples.",
        key="csv_upload",
    )
    sample_rate = st.number_input(
        "Sampling rate (Hz)", min_value=50, max_value=5000, value=360, step=1
    )
    if uploaded is None:
        st.caption("No file is stored until you start an analysis.")
        return
    try:
        frame = read_ecg_csv(uploaded.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return
    columns = list(frame.columns)
    signal_column = st.selectbox(
        "Signal column", columns, format_func=lambda column: str(column)
    )
    st.session_state["pending_upload"] = {
        "name": uploaded.name,
        "bytes": uploaded.getvalue(),
        "frame": frame,
        "column": signal_column,
        "sample_rate": int(sample_rate),
    }
    st.success(f"Validated {uploaded.name}: {len(frame):,} rows.")
    st.dataframe(frame.head(8), use_container_width=True, hide_index=True)
    if st.button("Analyze ECG", type="primary", use_container_width=True):
        with st.spinner("Processing ECG signal..."):
            pending = st.session_state["pending_upload"]
            values = pending["frame"][pending["column"]].to_numpy(dtype=float)
            st.session_state["analysis"] = analyze_ecg(
                values, int(pending["sample_rate"]), pending["name"]
            )
        st.session_state["pending_upload"] = None
        st.session_state["page"] = "Results"
        st.rerun()


def show_wfdb_upload() -> None:
    uploaded_files = st.file_uploader(
        "Choose MIT-BIH record files",
        type=["hea", "dat", "atr"],
        accept_multiple_files=True,
        help="Upload one matching .hea and .dat pair. An .atr file is optional.",
        key="wfdb_upload",
    )
    if not uploaded_files:
        st.caption("WFDB files are written only to a temporary directory while loading.")
        return
    file_map = {Path(file.name).name: file.getvalue() for file in uploaded_files}
    try:
        signal, sample_rate, record_name, annotation_count = read_wfdb_record(file_map)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.success(f"Validated MIT-BIH record {record_name} at {sample_rate} Hz.")
    columns = st.columns(3)
    columns[0].metric("Samples", f"{len(signal):,}")
    columns[1].metric("Channels", "1st ECG channel")
    columns[2].metric("Annotations", f"{annotation_count:,}")
    st.caption("The first physical ECG channel is analyzed. Reference annotations are informational.")
    if st.button("Analyze MIT-BIH record", type="primary", use_container_width=True):
        with st.spinner("Processing MIT-BIH ECG signal..."):
            st.session_state["analysis"] = analyze_ecg(
                signal, sample_rate, f"{record_name}.dat"
            )
        st.session_state["pending_upload"] = None
        st.session_state["page"] = "Results"
        st.rerun()


def show_analysis(result: dict[str, object]) -> None:
    st.header("ECG visualization and analysis")
    seconds = st.slider("Waveform window (seconds)", 2, 60, 10)
    st.plotly_chart(waveform_figure(result, seconds), use_container_width=True)
    statistics = result["statistics"]
    cols = st.columns(5)
    for column, label, value in zip(
        cols,
        ("Samples", "Duration", "Heart rate", "Detected beats", "Std. deviation"),
        (
            f"{statistics['samples']:,}",
            f"{statistics['duration_seconds']:.2f} s",
            f"{statistics['heart_rate_bpm']:.1f} BPM"
            if statistics["heart_rate_bpm"] is not None
            else "Unavailable",
            f"{statistics['detected_beats']:,}",
            f"{statistics['std']:.4f}",
        ),
    ):
        column.metric(label, value)
    status = str(result["prediction"])
    style = "normal" if status == "Normal Heartbeat" else "abnormal"
    st.markdown(
        f'<div class="status-card {style}"><h2>{status}</h2>'
        f"<p>Confidence: {float(result['confidence']):.1%}</p></div>",
        unsafe_allow_html=True,
    )
    st.caption(str(result["explanation"]))


def show_results(result: dict[str, object]) -> None:
    st.header("Analysis results")
    st.write(f"**File:** {result['file_name']}")
    show_analysis(result)
    pdf = create_pdf_report(result)
    st.download_button(
        "Download PDF report",
        data=pdf,
        file_name="ecg_analysis_report.pdf",
        mime="application/pdf",
        type="primary",
    )
    if st.button("Clear Results", use_container_width=True):
        st.session_state.pop("analysis", None)
        st.session_state.pop("pending_upload", None)
        st.rerun()


def main() -> None:
    apply_styles()
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"
    with st.sidebar:
        st.markdown("## ♥ Cloud ECG")
        page = st.radio(
            "Navigation", ["Home", "Upload ECG", "Analysis", "Results"],
            index=["Home", "Upload ECG", "Analysis", "Results"].index(
                st.session_state["page"]
            ),
        )
        st.session_state["page"] = page
        st.divider()
        st.caption("Temporary processing • No accounts • No personal data storage")

    if page == "Home":
        show_home()
    elif page == "Upload ECG":
        show_upload()
    elif page == "Analysis":
        if "analysis" not in st.session_state:
            st.info("Upload and analyze an ECG first.")
        else:
            show_analysis(st.session_state["analysis"])
    else:
        if "analysis" not in st.session_state:
            st.info("Upload and analyze an ECG first.")
        else:
            show_results(st.session_state["analysis"])


if __name__ == "__main__":
    main()
