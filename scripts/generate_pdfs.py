"""Generate sample PDF sustainability summaries for each demo organization.

Uses reportlab to create real PDFs — not stubs — so that the PDF loader
path in the pipeline gets exercised end-to-end.

Usage:
    python scripts/generate_pdfs.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _build_pdf(path: Path, org_name: str, data: dict) -> None:
    """Build a single-page sustainability summary PDF."""
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = styles["BodyText"]

    elements: list = []

    # Title
    elements.append(Paragraph(f"{org_name} — Sustainability Summary 2025", title_style))
    elements.append(Spacer(1, 6 * mm))

    # Overview
    elements.append(Paragraph("Overview", heading_style))
    elements.append(Paragraph(data["overview"], body_style))
    elements.append(Spacer(1, 3 * mm))

    # Emissions table
    elements.append(Paragraph("GHG Emissions (tCO2e)", heading_style))
    table_data = [["Scope", "2023", "2024", "2025"]]
    for row in data["emissions"]:
        table_data.append(row)

    table = Table(table_data, colWidths=[50 * mm, 35 * mm, 35 * mm, 35 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 3 * mm))

    # Key initiatives
    elements.append(Paragraph("Key Sustainability Initiatives", heading_style))
    for initiative in data["initiatives"]:
        elements.append(Paragraph(f"• {initiative}", body_style))
    elements.append(Spacer(1, 3 * mm))

    # Targets
    elements.append(Paragraph("Targets", heading_style))
    for target in data["targets"]:
        elements.append(Paragraph(f"• {target}", body_style))

    doc.build(elements)
    print(f"Generated: {path}")


# ── Org-specific data ─────────────────────────────────────────────

_ORG_001_DATA = {
    "overview": (
        "Aurora Textiles Pvt Ltd is a leading Indian textile manufacturer "
        "operating four facilities across Gujarat and Tamil Nadu. In FY 2025, "
        "total GHG emissions were 13,570 tCO2e, a 7.8% reduction from the "
        "2023 baseline. The company's sustainability strategy focuses on energy "
        "transition, water stewardship, and circular economy principles."
    ),
    "emissions": [
        ["Scope 1", "1,420", "1,310", "1,250"],
        ["Scope 2", "3,800", "3,600", "3,420"],
        ["Scope 3", "9,500", "9,200", "8,900"],
        ["Total", "14,720", "14,110", "13,570"],
    ],
    "initiatives": [
        "Replaced coal-fired boilers with natural gas at Surat facility.",
        "Commissioned 4.2 MWp rooftop solar across two plants.",
        "Achieved 42.4% water recycling rate through ZLD systems.",
        "Fabric scrap recycling: 180 tonnes diverted annually.",
        "Electrified 40% of the company vehicle fleet.",
    ],
    "targets": [
        "30% Scope 1+2 reduction by 2030 (base year 2023).",
        "50% renewable energy by 2028.",
        "Zero liquid discharge at all facilities by 2027.",
        "90% waste diversion rate by 2027.",
    ],
}

_ORG_002_DATA = {
    "overview": (
        "Meridian Logistics Ltd operates a fleet of 320 trucks and 12 "
        "warehouses, providing road freight and 3PL services across India. "
        "In FY 2025, total GHG emissions were 22,140 tCO2e, a 9.8% reduction "
        "from the 2023 baseline. The company is focused on fleet modernisation, "
        "warehouse energy efficiency, and driver welfare."
    ),
    "emissions": [
        ["Scope 1", "3,150", "2,980", "2,810"],
        ["Scope 2", "5,600", "5,350", "5,130"],
        ["Scope 3", "15,800", "15,100", "14,200"],
        ["Total", "24,550", "23,430", "22,140"],
    ],
    "initiatives": [
        "Inducted 45 CNG-powered trucks into the fleet.",
        "Route optimisation reduced empty-run kilometres by 12%.",
        "3 MWp rooftop solar at Pune distribution centre.",
        "Reusable pallet program eliminated 120 tonnes of packaging waste.",
        "Driver eco-training certified 280 drivers in FY 2025.",
    ],
    "targets": [
        "25% Scope 1+2 reduction by 2030 (base year 2023).",
        "30% of fleet to electric/CNG by 2028.",
        "30% renewable electricity by 2028.",
        "80% waste diversion rate by 2027.",
    ],
}


def main() -> None:
    """Generate PDFs for both demo organizations."""
    data_dir = Path(__file__).resolve().parent.parent / "data"

    _build_pdf(
        data_dir / "org_001" / "sustainability_summary_2025.pdf",
        "Aurora Textiles Pvt Ltd",
        _ORG_001_DATA,
    )
    _build_pdf(
        data_dir / "org_002" / "sustainability_summary_2025.pdf",
        "Meridian Logistics Ltd",
        _ORG_002_DATA,
    )


if __name__ == "__main__":
    main()
