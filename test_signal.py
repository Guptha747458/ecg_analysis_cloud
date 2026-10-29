
import numpy as np
import matplotlib.pyplot as plt

from ecg_utils import filter_ecg, detect_r_peaks

fs = 360
duration = 10

time = np.arange(fs * duration) / fs

# Synthetic test signal only; not a real ECG recording
signal = (
    np.sin(2 * np.pi * 1.2 * time)
    + 0.2 * np.sin(2 * np.pi * 50 * time)
)

filtered = filter_ecg(signal, fs)
peaks = detect_r_peaks(filtered, fs)

plt.figure(figsize=(12, 4))
plt.plot(time, signal, label="Original")
plt.plot(time, filtered, label="Filtered")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("ECG Signal Processing Test")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Input samples:", len(signal))
print("Filtered samples:", len(filtered))
print("Candidate peaks:", len(peaks))
