"""app/pages/reports.py — Page 5: Reports"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

ROOT        = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports_output"


def render(df: pd.DataFrame):
    st.markdown('<div class="main-header">📄 Automated Reports</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-generated weekly grievance intelligence reports in PDF format</div>',
                unsafe_allow_html=True)

    # ── Quick stats for the report ────────────────────────────────────────────
    st.subheader("Current Period Snapshot")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Grievances",    f"{len(df):,}")
    with k2:
        st.metric("Resolution Rate",
                  f"{df['status'].isin(['resolved','closed']).mean()*100:.1f}%")
    with k3:
        sla_breach = (pd.to_numeric(df["predicted_resolution_days"],
                                    errors="coerce") > 21).sum()
        st.metric("SLA Breaches",        f"{int(sla_breach):,}")
    with k4:
        st.metric("Anomalies Detected",
                  f"{int(df['is_anomaly'].fillna(0).sum()):,}")

    st.markdown("---")

    # ── Generate new report ───────────────────────────────────────────────────
    st.subheader("Generate New Report")
    col1, col2 = st.columns([2, 1])
    with col1:
        period = st.text_input(
            "Reporting period",
            value=datetime.now().strftime("%B %Y"),
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("📝 Generate Report", use_container_width=True)

    if generate_btn:
        with st.spinner("Groq is writing your report... (~15 seconds)"):
            try:
                import sys
                sys.path.insert(0, str(ROOT / "src" / "reporting"))
                from report_generator import run
                pdf_path = run(period=period)
                if pdf_path and Path(pdf_path).exists():
                    st.success(f"Report generated: {Path(pdf_path).name}")
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label    = "⬇️ Download PDF",
                            data     = f,
                            file_name= Path(pdf_path).name,
                            mime     = "application/pdf",
                        )
                else:
                    st.warning("Report generated as text (PDF library issue). Check reports_output/")
            except Exception as e:
                st.error(f"Report generation failed: {e}")

    st.markdown("---")

    # ── List existing reports ─────────────────────────────────────────────────
    st.subheader("Previous Reports")
    REPORTS_DIR.mkdir(exist_ok=True)
    pdfs = sorted(REPORTS_DIR.glob("*.pdf"), reverse=True)
    txts = sorted(REPORTS_DIR.glob("*.txt"), reverse=True)
    all_reports = list(pdfs) + list(txts)

    if not all_reports:
        st.info("No reports yet. Click 'Generate Report' above to create your first one.")
    else:
        for report_path in all_reports[:10]:
            col_r1, col_r2, col_r3 = st.columns([3, 1, 1])
            stat = report_path.stat()
            with col_r1:
                st.write(f"📄 {report_path.name}")
            with col_r2:
                st.caption(
                    datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %Y %H:%M")
                )
            with col_r3:
                with open(report_path, "rb") as f:
                    mime = "application/pdf" if report_path.suffix == ".pdf" else "text/plain"
                    st.download_button(
                        label    = "⬇️",
                        data     = f,
                        file_name= report_path.name,
                        mime     = mime,
                        key      = f"dl_{report_path.name}",
                    )

    # ── Report schedule info ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📅 Automated Schedule")
    st.info("""
Reports are generated automatically by the Airflow DAG in `pipeline/weekly_report_dag.py`.

**Schedule:** Every Monday at 8:00 AM IST
**Delivery:** Saved to `reports_output/` folder
**Format:** PDF with KPI table, 7 sections, government styling

To run manually at any time:
```bash
python src/reporting/report_generator.py
```
    """)

    with st.expander("📋 Report sections"):
        st.markdown("""
1. **Executive Summary** — Key numbers, resolution rate, critical alerts
2. **Critical Alerts** — High-risk escalations, SLA breaches
3. **Top Grievance Areas** — Department and root cause analysis
4. **Department Performance Scorecard** — Average resolution vs SLA
5. **Geographic Hotspots** — State-wise concentration
6. **Recommended Interventions** — Specific, actionable steps
7. **Monitoring Focus** — Next period priorities
        """)