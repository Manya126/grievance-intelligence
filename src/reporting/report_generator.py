"""
report_generator.py
-------------------
Week 6 — GenAI Automated Report Generation

Uses Groq (LLaMA-3) to generate a formal government-style weekly
grievance intelligence report, then converts it to a styled PDF.

Pipeline:
  1. Aggregate stats from DB (grievances, anomalies, SLA)
  2. Feed stats to Groq with a formal government prompt
  3. Groq writes the report in structured sections
  4. ReportLab renders it as a professional PDF

Output: reports_output/grievance_report_YYYYMMDD.pdf

Run:
  python report_generator.py              # generate this week's report
  python report_generator.py --preview    # print report text only (no PDF)
  python report_generator.py --period "May 2026"
"""

import os, sys, sqlite3, logging, argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parents[2]
DB_PATH      = os.getenv("DB_PATH", str(ROOT / "data" / "grievances.db"))
REPORTS_DIR  = ROOT / "reports_output"
GROQ_KEY     = os.getenv("GROQ_API_KEY", "")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: AGGREGATE STATS FROM DB
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def gather_stats(period_days: int = 30) -> dict:
    """Pull all stats needed for the report from DB."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM grievances", conn)
    conn.close()

    df["date_filed_dt"] = pd.to_datetime(df["date_filed"], errors="coerce", utc=False)
    if df["date_filed_dt"].dt.tz is not None:
        df["date_filed_dt"] = df["date_filed_dt"].dt.tz_localize(None)

    cutoff = pd.Timestamp.now() - timedelta(days=period_days)
    recent = df[df["date_filed_dt"] >= cutoff].copy()

    # Basic counts
    stats = {
        "period_days":       period_days,
        "total_all_time":    len(df),
        "total_recent":      len(recent),
        "pending":           (df["status"] == "pending").sum(),
        "resolved":          df["status"].isin(["resolved","closed"]).sum(),
        "escalated":         (df["status"] == "escalated").sum(),
        "resolution_rate":   round(df["status"].isin(["resolved","closed"]).mean() * 100, 1),
    }

    # Top departments
    stats["top_departments"] = (
        df["department"].value_counts().head(5).to_dict()
    )

    # Top states
    stats["top_states"] = (
        df[df["state"] != "national"]["state"]
        .value_counts().head(5).to_dict()
    )

    # Urgency breakdown
    stats["urgency_dist"] = df["urgency"].value_counts().to_dict()

    # Root cause breakdown
    stats["root_causes"] = df["root_cause"].value_counts().head(5).to_dict()

    # Anomalies
    stats["total_anomalies"]  = int(df["is_anomaly"].fillna(0).astype(int).sum())
    stats["anomaly_rate_pct"] = round(df["is_anomaly"].fillna(0).mean() * 100, 1)

    # SLA stats
    pred_days = pd.to_numeric(df["predicted_resolution_days"], errors="coerce")
    stats["avg_predicted_resolution"] = round(pred_days.mean(), 1)
    stats["sla_breach_count"]         = int((pred_days > 21).sum())
    stats["sla_breach_rate_pct"]      = round((pred_days > 21).mean() * 100, 1)

    # Top SLA breaching departments
    sla_dept = (df.assign(pred=pred_days)
                 .groupby("department")["pred"]
                 .mean()
                 .sort_values(ascending=False)
                 .head(5)
                 .round(1)
                 .to_dict())
    stats["worst_sla_departments"] = sla_dept

    # High escalation risk
    esc_risk = pd.to_numeric(df["escalation_risk"], errors="coerce")
    stats["high_escalation_count"] = int((esc_risk > 0.5).sum())

    # Sentiment breakdown
    stats["sentiment_dist"] = df["sentiment"].value_counts().to_dict()

    logger.info("Stats gathered from DB")
    logger.info("  Total records:      %d", stats["total_all_time"])
    logger.info("  Pending:            %d", stats["pending"])
    logger.info("  Resolution rate:    %.1f%%", stats["resolution_rate"])
    logger.info("  Anomalies:          %d", stats["total_anomalies"])
    logger.info("  SLA breaches:       %d (%.1f%%)",
                stats["sla_breach_count"], stats["sla_breach_rate_pct"])

    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: GROQ REPORT GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REPORT_SYSTEM_PROMPT = """You are a senior policy analyst writing official reports for the
Government of India's Department of Administrative Reforms & Public Grievances (DARPG).

Write formal, precise, data-driven government English.
Use specific numbers from the data provided.
Be actionable — every finding should have a recommendation.
Keep total length under 800 words.
Do NOT use markdown symbols like ** or ##. Use plain text with CAPS for headers."""

def build_report_prompt(stats: dict, period: str) -> str:
    top_dept_str = "\n".join(
        f"  - {dept}: {cnt} grievances"
        for dept, cnt in stats["top_departments"].items()
    )
    sla_str = "\n".join(
        f"  - {dept}: {days} days avg"
        for dept, days in stats["worst_sla_departments"].items()
    )
    top_rc = list(stats["root_causes"].keys())[:3]

    return f"""Generate a formal Weekly Grievance Intelligence Report for the period: {period}

DATA SUMMARY:
- Total grievances in system: {stats['total_all_time']:,}
- Pending resolution: {stats['pending']:,}
- Resolved/Closed: {stats['resolved']:,}
- Resolution rate: {stats['resolution_rate']}%
- Escalated cases: {stats['escalated']}
- Anomalous records detected: {stats['total_anomalies']} ({stats['anomaly_rate_pct']}%)
- High escalation risk cases: {stats['high_escalation_count']}
- Average predicted resolution: {stats['avg_predicted_resolution']} days
- SLA breaches (>21 days): {stats['sla_breach_count']} ({stats['sla_breach_rate_pct']}%)

TOP DEPARTMENTS BY VOLUME:
{top_dept_str}

DEPARTMENTS WITH WORST SLA PERFORMANCE:
{sla_str}

TOP ROOT CAUSES: {', '.join(top_rc)}

URGENCY BREAKDOWN: {stats['urgency_dist']}

TOP STATES: {list(stats['top_states'].keys())[:4]}

Write the report with these exact sections:
1. EXECUTIVE SUMMARY
2. CRITICAL ALERTS
3. TOP GRIEVANCE AREAS
4. DEPARTMENT PERFORMANCE SCORECARD
5. GEOGRAPHIC HOTSPOTS
6. RECOMMENDED INTERVENTIONS
7. MONITORING FOCUS FOR NEXT PERIOD

Use plain text only. No markdown. Be specific with numbers."""


def generate_report_text(stats: dict, period: str) -> str:
    """Call Groq to generate the report text."""
    if not GROQ_KEY:
        logger.warning("GROQ_API_KEY not set — using template report")
        return _template_report(stats, period)

    try:
        from groq import Groq
        try:
            client = Groq(api_key=GROQ_KEY)
        except TypeError:
            import httpx
            client = Groq(api_key=GROQ_KEY, http_client=httpx.Client())

        logger.info("Generating report with Groq LLaMA-3...")
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                {"role": "user",   "content": build_report_prompt(stats, period)},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        report_text = completion.choices[0].message.content.strip()
        logger.info("Report generated successfully (%d chars)", len(report_text))
        return report_text

    except Exception as e:
        logger.error("Groq report generation failed: %s", str(e)[:100])
        logger.info("Falling back to template report")
        return _template_report(stats, period)


def _template_report(stats: dict, period: str) -> str:
    """Fallback template report when Groq is unavailable."""
    top_dept = list(stats["top_departments"].keys())[0] if stats["top_departments"] else "Unknown"
    top_state = list(stats["top_states"].keys())[0] if stats["top_states"] else "Unknown"
    worst_sla_dept = list(stats["worst_sla_departments"].keys())[0] if stats["worst_sla_departments"] else "Unknown"

    return f"""GOVERNMENT OF INDIA
DEPARTMENT OF ADMINISTRATIVE REFORMS & PUBLIC GRIEVANCES
WEEKLY GRIEVANCE INTELLIGENCE REPORT

Reporting Period: {period}
Generated: {datetime.now().strftime('%d %B %Y, %H:%M IST')}

EXECUTIVE SUMMARY

The CPGRAMS Grievance Intelligence System has processed {stats['total_all_time']:,} grievances
as of this reporting period. The current resolution rate stands at {stats['resolution_rate']}%,
with {stats['pending']:,} cases pending resolution. A total of {stats['escalated']} cases have
been escalated, requiring immediate departmental attention. The AI-powered anomaly detection
system has flagged {stats['total_anomalies']} records ({stats['anomaly_rate_pct']}%) for
further review.

CRITICAL ALERTS

{stats['sla_breach_count']} grievances ({stats['sla_breach_rate_pct']}%) are projected to
breach the 21-day resolution SLA. {stats['high_escalation_count']} cases have been assigned
a high escalation risk score (>0.5) by the ML classifier. Immediate intervention is recommended
for these cases.

TOP GRIEVANCE AREAS

The highest grievance volume was recorded in {top_dept}, followed by other central ministries.
Top root causes include: {', '.join(list(stats['root_causes'].keys())[:3])}.
Sentiment analysis indicates the majority of grievances carry negative sentiment,
reflecting citizen frustration with service delivery timelines.

DEPARTMENT PERFORMANCE SCORECARD

Average predicted resolution time across all departments is {stats['avg_predicted_resolution']} days,
significantly above the 21-day SLA target. Departments with the highest average predicted
resolution times require focused intervention. The AI system has classified urgency levels
to assist in prioritisation: {stats['urgency_dist']}.

GEOGRAPHIC HOTSPOTS

{top_state} and neighbouring states account for the highest grievance concentration.
State-wise analysis indicates that urban districts generate higher complaint volumes,
while rural areas show higher proportions of agriculture and infrastructure grievances.

RECOMMENDED INTERVENTIONS

1. Establish a dedicated grievance cell in {top_dept} to address the backlog of
   {list(stats['top_departments'].values())[0]} pending cases.
2. Implement automated SLA breach alerts for {worst_sla_dept} where predicted
   resolution exceeds 45 days.
3. Deploy district-level grievance camps in top-volume states to address complaints
   at source and reduce portal dependency.
4. Mandate weekly review meetings for all departments with escalation rates above 5%.
5. Integrate real-time CPGRAMS data feeds to improve ML model accuracy for
   resolution time prediction.

MONITORING FOCUS FOR NEXT PERIOD

Priority monitoring should focus on escalated cases, SLA-breaching departments,
and the {stats['total_anomalies']} anomalous records flagged by the Isolation Forest
detector. A follow-up review of anomaly patterns is recommended within 72 hours.

---
Report generated by AI-Powered Grievance Intelligence System
Classification: INTERNAL | For Official Use Only
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: PDF GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_pdf(report_text: str, stats: dict, period: str, output_path: str) -> str:
    """Render the report as a styled PDF using ReportLab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    except ImportError:
        logger.error("ReportLab not installed. Run: pip install reportlab")
        # Save as plain text instead
        txt_path = output_path.replace(".pdf", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info("Saved as text: %s", txt_path)
        return txt_path

    doc  = SimpleDocTemplate(
        output_path,
        pagesize       = A4,
        rightMargin    = 2*cm,
        leftMargin     = 2*cm,
        topMargin      = 2*cm,
        bottomMargin   = 2*cm,
    )

    # Styles
    styles = getSampleStyleSheet()
    NAVY   = colors.HexColor("#1B2A5E")
    GOLD   = colors.HexColor("#C9993A")
    LIGHT  = colors.HexColor("#F5F5F5")

    title_style = ParagraphStyle("title",
        fontSize=14, fontName="Helvetica-Bold",
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=4)
    subtitle_style = ParagraphStyle("subtitle",
        fontSize=10, fontName="Helvetica",
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=12)
    section_style = ParagraphStyle("section",
        fontSize=11, fontName="Helvetica-Bold",
        textColor=NAVY, spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("body",
        fontSize=9, fontName="Helvetica",
        leading=14, alignment=TA_JUSTIFY, spaceAfter=8)
    footer_style = ParagraphStyle("footer",
        fontSize=7, fontName="Helvetica",
        textColor=colors.grey, alignment=TA_CENTER)

    story = []

    # Header
    story.append(Paragraph("GOVERNMENT OF INDIA", title_style))
    story.append(Paragraph("Department of Administrative Reforms &amp; Public Grievances", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=GOLD))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"WEEKLY GRIEVANCE INTELLIGENCE REPORT &mdash; {period}",
        ParagraphStyle("rptitle", fontSize=12, fontName="Helvetica-Bold",
                       textColor=NAVY, alignment=TA_CENTER, spaceAfter=4)))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M IST')} &nbsp;|&nbsp; AI-Powered CPGRAMS Analytics",
        subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY))
    story.append(Spacer(1, 12))

    # KPI table
    kpi_data = [
        ["METRIC", "VALUE", "STATUS"],
        ["Total Grievances",          f"{stats['total_all_time']:,}",              ""],
        ["Pending Resolution",         f"{stats['pending']:,}",                    "⚠"],
        ["Resolution Rate",            f"{stats['resolution_rate']}%",             "✓" if stats['resolution_rate'] > 50 else "✗"],
        ["Escalated Cases",            f"{stats['escalated']}",                    "🔴" if stats['escalated'] > 10 else "🟡"],
        ["SLA Breaches (>21d)",        f"{stats['sla_breach_count']} ({stats['sla_breach_rate_pct']}%)", "🔴"],
        ["Anomalies Detected",         f"{stats['total_anomalies']}",              "⚠"],
        ["Avg Predicted Resolution",   f"{stats['avg_predicted_resolution']} days", "🔴" if stats['avg_predicted_resolution'] > 21 else "🟢"],
        ["High Escalation Risk Cases", f"{stats['high_escalation_count']}",        "⚠"],
    ]
    tbl = Table(kpi_data, colWidths=[8*cm, 4*cm, 2*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("BACKGROUND",    (0,1), (-1,-1), LIGHT),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("ALIGN",         (1,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 16))

    # Report body — parse sections
    lines = report_text.split("\n")
    current_section = None
    buffer = []

    SECTION_KEYWORDS = [
        "EXECUTIVE SUMMARY", "CRITICAL ALERTS", "TOP GRIEVANCE",
        "DEPARTMENT PERFORMANCE", "GEOGRAPHIC HOTSPOT",
        "RECOMMENDED INTERVENTION", "MONITORING FOCUS",
    ]

    def flush_buffer():
        if buffer:
            text = " ".join(buffer).strip()
            if text:
                story.append(Paragraph(text, body_style))
            buffer.clear()

    for line in lines:
        line = line.strip()
        if not line:
            flush_buffer()
            continue

        # Detect section headers
        is_header = any(kw in line.upper() for kw in SECTION_KEYWORDS)
        if is_header and len(line) < 80:
            flush_buffer()
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
            story.append(Paragraph(line, section_style))
        elif line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            flush_buffer()
            story.append(Paragraph("&bull; " + line[3:].strip(), body_style))
        elif line.startswith("-"):
            flush_buffer()
            story.append(Paragraph("&bull; " + line[1:].strip(), body_style))
        elif line.startswith("---") or line.startswith("==="):
            flush_buffer()
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        elif line.upper() == line and len(line) > 5 and len(line) < 60:
            # All-caps lines = section title
            flush_buffer()
            story.append(Paragraph(line, section_style))
        else:
            buffer.append(line)

    flush_buffer()

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GOLD))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Classification: INTERNAL | For Official Use Only | "
        "Generated by AI-Powered Grievance Intelligence System",
        footer_style))

    doc.build(story)
    logger.info("PDF saved -> %s", output_path)
    return output_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MASTER PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(period: str = None, preview: bool = False) -> str:
    if not period:
        period = datetime.now().strftime("%B %Y")

    logger.info("=" * 55)
    logger.info("WEEK 6 — GENAI REPORT GENERATOR")
    logger.info("=" * 55)
    logger.info("Period: %s", period)

    # Step 1: Gather stats
    stats = gather_stats(period_days=30)

    # Step 2: Generate report text with Groq
    report_text = generate_report_text(stats, period)

    if preview:
        print("\n" + "="*60)
        print(report_text)
        print("="*60)
        return ""

    # Step 3: Build PDF
    ts          = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = str(REPORTS_DIR / f"grievance_report_{ts}.pdf")
    final_path  = build_pdf(report_text, stats, period, output_path)

    logger.info("=" * 55)
    logger.info("REPORT COMPLETE")
    logger.info("  File: %s", final_path)
    logger.info("=" * 55)
    logger.info("Next: python app/main.py  (Week 7 — Streamlit Dashboard)")

    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--period",  type=str, default=None,
                        help="Reporting period label (default: current month)")
    parser.add_argument("--preview", action="store_true",
                        help="Print report text only, skip PDF generation")
    args = parser.parse_args()

    run(period=args.period, preview=args.preview)