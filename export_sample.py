
from pathlib import Path

import pandas as pd
import wfdb

root = Path(
    "data/mit-bih-arrhythmia-database-1.0.0"
)

record = wfdb.rdrecord(str(root / "100"))

df = pd.DataFrame({
    "ecg_signal": record.p_signal[:, 0]
})

df.to_csv("sample_ecg.csv", index=False)

print("Exported sample_ecg.csv")
print("Sampling rate:", record.fs, "Hz")
print("Number of samples:", len(df))
