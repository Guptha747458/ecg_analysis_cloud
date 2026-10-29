
import numpy as np

from scipy.signal import butter, sosfiltfilt, find_peaks, resample


def filter_ecg(signal, fs):
    """Bandpass filter for ECG signals."""
    signal = np.asarray(signal, dtype=float).squeeze()

    if signal.ndim != 1 or len(signal) < 30:
        raise ValueError("ECG signal must be a valid 1D array.")

    if not np.isfinite(fs) or fs < 100:
        raise ValueError("Sampling rate must be at least 100 Hz.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG contains invalid values.")

    if np.std(signal) < 1e-10:
        raise ValueError("ECG signal is constant.")

    sos = butter(
        3, [0.5, 40],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    filtered = sosfiltfilt(sos, signal)
    return filtered - np.median(filtered)


def prepare_beat(signal, peak, fs, target_length=180):
    """Extract and normalize an ECG beat around its R peak."""
    signal = np.asarray(signal, dtype=float).squeeze()

    if signal.ndim != 1 or fs <= 0 or target_length <= 0:
        raise ValueError("Invalid signal or beat parameters.")

    # 200 ms before and 400 ms after the R peak
    start = int(peak) - int(round(0.20 * fs))
    end = int(peak) + int(round(0.40 * fs))

    if start < 0 or end > len(signal):
        return None

    beat = signal[start:end]

    if not np.all(np.isfinite(beat)):
        return None

    beat = resample(beat, target_length)
    beat = beat - np.mean(beat)

    std = np.std(beat)
    if std < 1e-8:
        return None

    return (beat / std).astype(np.float32)


def detect_r_peaks(signal, fs, polarity="Positive"):
    """Detect candidate R peaks in a filtered ECG signal."""
    signal = np.asarray(signal, dtype=float).squeeze()

    if signal.ndim != 1 or len(signal) < 3:
        raise ValueError("ECG signal must be a valid 1D array.")

    if not np.isfinite(fs) or fs < 100:
        raise ValueError("Invalid sampling rate.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG contains invalid values.")

    if np.std(signal) < 1e-10:
        return np.array([], dtype=int)

    if polarity == "Negative":
        working_signal = -signal
    elif polarity == "Absolute":
        working_signal = np.abs(signal)
    elif polarity == "Positive":
        working_signal = signal
    else:
        raise ValueError("Choose Positive, Negative, or Absolute polarity.")

    prominence = max(0.5 * np.std(working_signal), 1e-8)

    peaks, _ = find_peaks(
        working_signal,
        distance=int(0.30 * fs),
        prominence=prominence
    )

    return peaks
