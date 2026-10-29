"""PDF report generation for ECG analysis results."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def create_pdf_report(result: dict[str, Any]) -> bytes:
    """Create a report without including personal or account information."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output, pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch
    )
    styles = getSampleStyleSheet()
    statistics = result["statistics"]
    heart_rate = statistics["heart_rate_bpm"]
    rows = [
        ["File", str(result["file_name"])],
        ["Prediction", str(result["prediction"])],
        ["Confidence", f"{float(result['confidence']):.1%}"],
        ["Sampling rate", f"{result['sample_rate']} Hz"],
        ["Samples", f"{statistics['samples']:,}"],
        ["Duration", f"{statistics['duration_seconds']:.2f} seconds"],
        ["Heart rate", f"{heart_rate:.1f} BPM" if heart_rate else "Unavailable"],
        ["Mean amplitude", f"{statistics['mean']:.6f}"],
        ["Standard deviation", f"{statistics['std']:.6f}"],
        ["Detected beats", f"{statistics['detected_beats']:,}"],
    ]
    story = [
        Paragraph("Cloud ECG Analysis System", styles["Title"]),
        Paragraph("Automated ECG screening report", styles["Heading2"]),
        Spacer(1, 0.2 * inch),
        Table(rows, colWidths=[1.65 * inch, 4.7 * inch]),
        Spacer(1, 0.2 * inch),
        Paragraph(str(result["explanation"]), styles["BodyText"]),
        Spacer(1, 0.2 * inch),
        Paragraph(
            "Educational screening output only. This report is not a diagnosis "
            "and should not replace professional medical advice.",
            styles["Italic"],
        ),
    ]
    table = story[3]
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e7f3f3")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b7c8ca")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    document.build(story)
    return output.getvalue()
