
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import wfdb

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from ecg_utils import filter_ecg, prepare_beat, detect_r_peaks

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ecg_model.joblib"

LABEL_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V", "F": "F"
}
CLASSES = ["N", "S", "V", "F"]

st.set_page_config(
    page_title="ECG Analysis in the Cloud",
    page_icon="❤️",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def match_peaks(reference, detected, tolerance):
    """One-to-one matching of reference and detected peak samples."""
    reference = np.asarray(reference, dtype=int)
    detected = np.asarray(detected, dtype=int)

    if len(reference) == 0 or len(detected) == 0:
        return [], len(reference), len(detected)

    # Only create candidate matches within the tolerance.
    distances = np.abs(reference[:, None] - detected[None, :])
    ref_idx, det_idx = linear_sum_assignment(distances)

    matches = [
        (int(reference[i]), int(detected[j]))
        for i, j in zip(ref_idx, det_idx)
        if distances[i, j] <= tolerance
    ]

    tp = len(matches)
    return matches, len(reference) - tp, len(detected) - tp


st.title("❤️ Healthcare: ECG Analysis in the Cloud")
st.write(
    "Upload ECG recordings to visualize signals, estimate heart rate, "
    "classify candidate beats and compare predictions with annotations."
)
st.info(
    "Educational research prototype only. Not intended for medical "
    "diagnosis or treatment decisions."
)

with st.sidebar:
    st.header("Analysis Settings")

    fs_input = st.number_input(
        "Sampling rate for CSV/TXT/NumPy (Hz)",
        min_value=100, max_value=2000, value=360, step=1
    )

    polarity = st.selectbox(
        "Expected R-peak polarity",
        ["Positive", "Negative", "Absolute"]
    )

    display_seconds = st.slider(
        "Waveform display duration (seconds)", 2, 20, 10
    )

    csv_has_header = st.checkbox(
        "CSV/TXT contains a header row", value=False
    )

    st.caption(
        "WFDB recordings use their header sampling rate. "
        "For other formats, provide the correct sampling rate."
    )

uploaded_files = st.file_uploader(
    "Upload ECG data",
    type=["csv", "txt", "hea", "dat", "atr", "npy", "npz"],
    accept_multiple_files=True,
    help=(
        "Upload one CSV/TXT/NPY/NPZ signal, or matching WFDB "
        "files such as 219.hea, 219.dat and optional 219.atr."
    )
)

if not uploaded_files:
    st.markdown("### Supported formats")
    st.write(
        "CSV, TXT, NPY, NPZ and WFDB recordings (.hea + .dat; "
        "optional .atr reference annotations)."
    )
    st.stop()

try:
    file_map = {Path(f.name).name: f for f in uploaded_files}
    extensions = {Path(n).suffix.lower() for n in file_map}

    signal = None
    annotations = None
    fs = int(fs_input)
    signal_name = "ECG"
    channel_names = []
    record_name = None

    # ---------------- WFDB ----------------
    if ".hea" in extensions or ".dat" in extensions:
        headers = [n for n in file_map if n.lower().endswith(".hea")]
        if len(headers) != 1:
            st.error("Upload exactly one .hea file and its matching .dat file.")
            st.stop()

        record_name = Path(headers[0]).stem
        dat_name = record_name + ".dat"

        if dat_name not in file_map:
            st.error(f"Matching file {dat_name} was not uploaded.")
            st.stop()

        with TemporaryDirectory() as temp:
            temp_path = Path(temp)

            for name, uploaded in file_map.items():
                if (
                    Path(name).stem == record_name
                    and Path(name).suffix.lower() in {".hea", ".dat", ".atr"}
                ):
                    (temp_path / name).write_bytes(uploaded.getvalue())

            base = str(temp_path / record_name)
            record = wfdb.rdrecord(base)
            data = record.p_signal if record.p_signal is not None else record.d_signal
            data = np.asarray(data)

            if data.ndim == 1:
                data = data.reshape(-1, 1)

            fs = float(record.fs)
            channel_names = list(record.sig_name or [
                f"Channel {i + 1}" for i in range(data.shape[1])
            ])

            atr_name = record_name + ".atr"
            if atr_name in file_map:
                ann = wfdb.rdann(base, "atr")
                annotations = {
                    "samples": np.asarray(ann.sample, dtype=int),
                    "symbols": list(ann.symbol)
                }

        channel = st.selectbox(
            "Select ECG channel",
            range(data.shape[1]),
            format_func=lambda i: channel_names[i]
        )
        signal = data[:, channel].astype(float)
        signal_name = channel_names[channel]

        st.success(f"Loaded WFDB record {record_name} at {fs:g} Hz.")

    # ---------------- CSV / TXT ----------------
    elif extensions.intersection({".csv", ".txt"}):
        candidates = [
            f for f in uploaded_files
            if Path(f.name).suffix.lower() in {".csv", ".txt"}
        ]
        if len(candidates) != 1:
            st.error("Upload only one CSV/TXT signal at a time.")
            st.stop()

        uploaded = candidates[0]
        ext = Path(uploaded.name).suffix.lower()

        if ext == ".csv":
            df = pd.read_csv(
                uploaded, header=0 if csv_has_header else None
            )
        else:
            df = pd.read_csv(
                uploaded,
                header=0 if csv_has_header else None,
                sep=None,
                engine="python"
            )

        numeric = [
            col for col in df.columns
            if pd.to_numeric(df[col], errors="coerce").notna().any()
        ]
        if not numeric:
            st.error("No numeric ECG signal column found.")
            st.stop()

        col = st.selectbox(
            "Select ECG signal column",
            numeric,
            format_func=lambda c: str(c) if csv_has_header else f"Column {int(c)+1}"
        )
        signal = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        signal_name = str(col)

    # ---------------- NumPy ----------------
    elif extensions.intersection({".npy", ".npz"}):
        candidates = [
            f for f in uploaded_files
            if Path(f.name).suffix.lower() in {".npy", ".npz"}
        ]
        if len(candidates) != 1:
            st.error("Upload one NPY/NPZ file at a time.")
            st.stop()

        uploaded = candidates[0]
        loaded = np.load(uploaded, allow_pickle=False)

        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded as archive:
                keys = list(archive.files)
                if not keys:
                    st.error("NPZ file contains no arrays.")
                    st.stop()
                key = st.selectbox("Select array", keys)
                arr = np.asarray(archive[key])
        else:
            arr = np.asarray(loaded)

        if arr.ndim == 1:
            signal = arr.astype(float)
        elif arr.ndim == 2:
            channel = st.selectbox(
                "Select signal channel", range(arr.shape[1])
            )
            signal = arr[:, channel].astype(float)
        else:
            st.error("Only 1D and 2D numeric arrays are supported.")
            st.stop()

        signal_name = uploaded.name

    else:
        st.error("Unsupported input format.")
        st.stop()

    signal = np.asarray(signal, dtype=float).squeeze()

    if signal.ndim != 1 or len(signal) < fs:
        st.error("Signal must be a 1D array with at least one second of samples.")
        st.stop()

    if not np.all(np.isfinite(signal)):
        st.error("Signal contains invalid numeric values.")
        st.stop()

    with st.spinner("Filtering ECG and detecting candidate peaks..."):
        filtered = filter_ecg(signal, fs)
        peaks = detect_r_peaks(filtered, fs, polarity)

    duration = len(filtered) / fs

    st.subheader("ECG Recording Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{duration:.2f} s")
    c2.metric("Samples", f"{len(signal):,}")
    c3.metric("Sampling rate", f"{fs:g} Hz")
    c4.metric("Candidate peaks", f"{len(peaks):,}")

    # ---------------- ECG plot ----------------
    st.subheader("Original vs Filtered ECG")

    shown = min(len(signal), int(display_seconds * fs))
    visible = peaks[peaks < shown]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(np.arange(shown) / fs, signal[:shown],
            linewidth=0.7, label="Original")
    ax.plot(np.arange(shown) / fs, filtered[:shown],
            linewidth=0.9, label="Filtered")

    if len(visible):
        ax.scatter(visible / fs, filtered[visible],
                   s=18, label="Detected peaks")

    if annotations is not None:
        ref_visible = annotations["samples"]
        ref_visible = ref_visible[ref_visible < shown]
        ax.scatter(
            ref_visible / fs, filtered[ref_visible],
            marker="x", s=35, label="Reference annotations"
        )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"ECG Signal: {signal_name}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ---------------- Heart rate ----------------
    st.subheader("Heart Rate Estimation")
    rr = np.diff(peaks) / fs
    valid_rr = rr[(rr >= 0.30) & (rr <= 2.0)]

    if len(valid_rr):
        bpm = 60 / np.median(valid_rr)
        c1, c2 = st.columns(2)
        c1.metric("Estimated heart rate", f"{bpm:.1f} BPM")
        c2.metric("Usable intervals", f"{len(valid_rr):,}")
    else:
        st.warning("No usable peak-to-peak intervals found.")

    # ---------------- Reference peak comparison ----------------
    if annotations is not None:
        st.subheader("Reference Annotation Comparison")

        ref_samples = annotations["samples"]
        symbols = annotations["symbols"]

        # Standard MIT-BIH beat symbols; excludes non-beat annotations.
        beat_indices = [
            i for i, symbol in enumerate(symbols)
            if symbol in LABEL_MAP and 0 <= ref_samples[i] < len(signal)
        ]
        ref_beats = ref_samples[beat_indices]
        ref_symbols = [symbols[i] for i in beat_indices]
        tolerance = int(round(0.10 * fs))

        matches, missed, extra = match_peaks(ref_beats, peaks, tolerance)

        tp = len(matches)
        precision = tp / (tp + extra) if tp + extra else 0.0
        recall = tp / (tp + missed) if tp + missed else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        st.write(f"Reference beat annotations: {len(ref_beats):,}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Matched peaks", f"{tp:,}")
        m2.metric("Missed reference beats", f"{missed:,}")
        m3.metric("Extra detected peaks", f"{extra:,}")
        m4.metric("Peak F1-score", f"{f1:.3f}")

        st.caption(
            "One-to-one matching with a ±100 ms tolerance. "
            "These are peak-detection metrics, not ML classification accuracy."
        )

    # ---------------- ML classification ----------------
    st.subheader("Machine-Learning Beat Classification")

    if not MODEL_PATH.exists():
        st.error("Model missing. Run python train_model.py first.")
        st.stop()

    bundle = load_model()
    model = bundle["model"]

    # The model must have been retrained using the same prepare_beat version.
    rows = []
    progress = st.progress(0)

    for i, peak in enumerate(peaks):
        beat = prepare_beat(
            filtered, int(peak), fs, bundle["beat_length"]
        )

        if beat is not None:
            x = beat.reshape(1, -1)
            prediction = str(model.predict(x)[0])
            confidence = np.nan

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(x)[0]
                confidence = float(np.max(probabilities))

            rows.append({
                "Sample": int(peak),
                "Time (seconds)": round(peak / fs, 3),
                "Predicted category": prediction,
                "Model confidence": confidence
            })

        if i % 25 == 0 or i == len(peaks) - 1:
            progress.progress((i + 1) / max(1, len(peaks)))

    progress.empty()

    if not rows:
        st.warning("No complete heartbeat segments were available.")
        st.stop()

    results = pd.DataFrame(rows)

    st.write("Predicted category counts")
    st.bar_chart(results["Predicted category"].value_counts())

    results["Model confidence"] = results["Model confidence"].map(
        lambda x: round(x, 3) if pd.notna(x) else None
    )
    st.dataframe(
        results.drop(columns=["Sample"]),
        use_container_width=True
    )

    st.download_button(
        "Download analysis results",
        data=results.drop(columns=["Sample"]).to_csv(index=False).encode("utf-8"),
        file_name="ecg_analysis_results.csv",
        mime="text/csv"
    )

    # ---------------- Actual vs predicted classification ----------------
    if annotations is not None:
        st.subheader("Reference vs ML Classification")

        # Match each reference beat to its nearest detected peak one-to-one.
        # Match the original sample indices to prediction rows.
        peak_to_prediction = {
            int(row["Sample"]): row["Predicted category"] for row in rows
        }
        ref_sample_to_label = {
            int(sample): LABEL_MAP[symbol]
            for sample, symbol in zip(ref_beats, ref_symbols)
        }

        if len(matches):
            ref_label_list = []
            pred_label_list = []

            # Only evaluate matched beats for which a model prediction exists.
            for ref_sample, detected_sample in matches:
                if (
                    ref_sample in ref_sample_to_label
                    and detected_sample in peak_to_prediction
                ):
                    ref_label_list.append(ref_sample_to_label[ref_sample])
                    pred_label_list.append(peak_to_prediction[detected_sample])

            if ref_label_list:
                st.caption(
                    "Classification scores are calculated only on matched "
                    "reference beats with available model predictions."
                )
                st.metric(
                    "Matched-beat classification accuracy",
                    f"{accuracy_score(ref_label_list, pred_label_list):.3f}"
                )

                st.text("Classification report")
                report = classification_report(
                    ref_label_list,
                    pred_label_list,
                    labels=CLASSES,
                    output_dict=True,
                    zero_division=0
                )
                st.dataframe(pd.DataFrame(report).transpose())

                cm = confusion_matrix(
                    ref_label_list,
                    pred_label_list,
                    labels=CLASSES
                )
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=CLASSES
                ).plot(ax=ax_cm, colorbar=False)
                ax_cm.set_title("Reference vs Predicted Classes")
                fig_cm.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)
            else:
                st.warning(
                    "No matched reference beats had usable model predictions."
                )
        else:
            st.warning(
                "No reference beats matched detected peaks within ±100 ms."
            )

except Exception as exc:
    st.error(f"Analysis failed: {exc}")
    st.info(
        "Check file formats, sampling rate, annotation files, and model compatibility."
    )

st.divider()
st.caption(
    "Healthcare: ECG Analysis in the Cloud | "
    "Educational prototype, not for clinical use."
)
