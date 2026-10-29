"""CSV parsing and lightweight, explainable ECG analysis."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from sklearn.preprocessing import StandardScaler


def read_ecg_csv(data: bytes) -> pd.DataFrame:
    """Read a CSV while preserving the first row whether a header exists."""
    if not data:
        raise ValueError("The uploaded CSV is empty.")
    try:
        raw = pd.read_csv(BytesIO(data), header=None, low_memory=False)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError("The file is not a readable CSV.") from exc
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if numeric.empty:
        raise ValueError("The CSV must contain at least one numeric column.")
    numeric.columns = [f"Signal {index + 1}" for index in range(numeric.shape[1])]
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    numeric = numeric.dropna(axis=0, how="any")
    if numeric.empty:
        raise ValueError("The CSV does not contain usable numeric samples.")
    return numeric


def _filtered_signal(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    nyquist = sample_rate / 2
    high_cutoff = min(40.0, nyquist * 0.8)
    if high_cutoff <= 0.5:
        return signal - np.median(signal)
    sos = butter(3, [0.5, high_cutoff], btype="bandpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, signal) - np.median(signal)


def analyze_ecg(signal: np.ndarray, sample_rate: int, file_name: str) -> dict[str, Any]:
    """Analyze one ECG channel and return serializable display/report data."""
    values = np.asarray(signal, dtype=float).reshape(-1)
    if len(values) < 100:
        raise ValueError("The ECG signal must contain at least 100 samples.")
    if sample_rate < 50 or not np.all(np.isfinite(values)):
        raise ValueError("The ECG contains invalid values or sampling rate.")
    if np.ptp(values) == 0:
        raise ValueError("The ECG signal is constant and cannot be analyzed.")

    filtered = _filtered_signal(values, sample_rate)
    prominence = max(float(np.std(filtered) * 0.45), 1e-8)
    peaks, _ = find_peaks(
        filtered,
        distance=max(1, int(sample_rate * 0.3)),
        prominence=prominence,
    )
    intervals = np.diff(peaks) / sample_rate
    usable_intervals = intervals[(intervals >= 0.3) & (intervals <= 2.0)]
    heart_rate = (
        float(60 / np.median(usable_intervals))
        if len(usable_intervals)
        else None
    )
    variability = (
        float(np.std(usable_intervals) / np.mean(usable_intervals))
        if len(usable_intervals) > 1 and np.mean(usable_intervals) > 0
        else 0.0
    )
    feature_values = np.array(
        [heart_rate or 0.0, variability, len(peaks) / (len(values) / sample_rate)]
    )
    scaled = StandardScaler().fit_transform(
        np.vstack([np.zeros(3), feature_values])
    )[1]
    abnormal_score = 0.0
    if heart_rate is None or heart_rate < 50 or heart_rate > 110:
        abnormal_score += 0.45
    if variability > 0.20:
        abnormal_score += min(0.35, variability)
    expected_beats = len(values) / sample_rate * 1.2
    if len(peaks) < max(1, expected_beats * 0.35):
        abnormal_score += 0.25
    abnormal_score = min(1.0, abnormal_score)
    abnormal = abnormal_score >= 0.45
    confidence = min(0.99, max(0.55, 0.55 + abs(abnormal_score - 0.45) * 0.9))
    prediction = "Abnormal Heartbeat" if abnormal else "Normal Heartbeat"
    explanation = (
        "The signal shows rate or beat-interval characteristics outside the "
        "screening thresholds."
        if abnormal
        else "The detected rate and beat intervals are within the screening thresholds."
    )
    return {
        "file_name": file_name,
        "signal": values.tolist(),
        "sample_rate": sample_rate,
        "peaks": peaks.tolist(),
        "prediction": prediction,
        "confidence": float(confidence),
        "explanation": explanation,
        "statistics": {
            "samples": len(values),
            "duration_seconds": len(values) / sample_rate,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "heart_rate_bpm": heart_rate,
            "detected_beats": len(peaks),
            "interval_variability": variability,
            "feature_scale": scaled.tolist(),
        },
    }
