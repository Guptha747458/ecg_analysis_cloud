# Cloud ECG Analysis System

A lightweight Streamlit healthcare dashboard for uploading ECG CSV files,
visualizing waveforms, running an explainable screening analysis, and
downloading a PDF report.

> **Medical notice:** This is an educational prototype, not a medical device.
> Results do not provide a diagnosis or treatment recommendation.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload `sample_ecg.csv`, use the default 360 Hz sampling rate, and select
**Analyze ECG**. CSV files can have an optional header and one or more numeric
columns. The sample rate must match the recording.

The **MIT-BIH / WFDB** input mode accepts a matching `.hea` and `.dat` pair,
plus an optional `.atr` annotation file. The first physical ECG channel is
analyzed and annotation counts are shown for reference. WFDB files are
created only inside a temporary directory while loading and are removed
automatically afterward.

## Deploy on Streamlit Cloud

1. Push this folder to a GitHub repository.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), choose **New
   app** and select the repository and `app.py`.
3. Deploy. Streamlit Cloud installs the packages from `requirements.txt`.

The app has no authentication, login, registration, or user accounts. Uploads
are read from Streamlit's in-memory upload object, analyzed, and discarded from
session state after processing. Results can be cleared with **Clear Results**.
No personal information or analysis history is written to SQLite or another
persistent store.

## Project files

- `app.py` - dashboard, navigation, Plotly visualization, and cleanup controls
- `ecg_analysis.py` - CSV validation, filtering, peak detection, and screening
- `pdf_report.py` - ReportLab PDF generation
- `sample_ecg.csv` - synthetic ECG-like sample for testing
