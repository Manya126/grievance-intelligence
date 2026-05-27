"""
main.py — Streamlit Dashboard Entry Point
AI-Powered Public Grievance Intelligence System

Run:
  streamlit run app/main.py

Pages:
  1. Overview        — KPIs, trends, maps
  2. Predictions     — enter text, get resolution + escalation risk + SHAP
  3. Anomalies       — surge alerts, SLA heatmap, outlier table
  4. Ask AI          — natural language query (Groq)
  5. Reports         — download auto-generated PDF
"""

import streamlit as st
import sqlite3
import pandas as pd
import os
import sys
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "ingestion"))
sys.path.insert(0, str(ROOT / "src" / "classification"))
sys.path.insert(0, str(ROOT / "src" / "reporting"))

os.chdir(ROOT)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title  = "Grievance Intelligence System",
    page_icon   = "🏛️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1B2A5E;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #F8F9FA;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1B2A5E;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1B2A5E;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .alert-box {
        background: #FFF3CD;
        border: 1px solid #FFC107;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }
    .critical-box {
        background: #F8D7DA;
        border: 1px solid #DC3545;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }
    [data-testid="stSidebar"] {
        background: #1B2A5E;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #CCC !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Shared data loader (cached) ───────────────────────────────────────────────
@st.cache_data(ttl=300)  # refresh every 5 minutes
def load_data():
    db_path = ROOT / "data" / "grievances.db"
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("SELECT * FROM grievances", conn)
    conn.close()

    df["date_filed_dt"] = pd.to_datetime(df["date_filed"], errors="coerce", utc=False)
    if df["date_filed_dt"].dt.tz is not None:
        df["date_filed_dt"] = df["date_filed_dt"].dt.tz_localize(None)
    df["date_filed_dt"] = df["date_filed_dt"].fillna(pd.Timestamp.now())

    df["escalation_risk"]            = pd.to_numeric(df["escalation_risk"], errors="coerce").fillna(0)
    df["predicted_resolution_days"]  = pd.to_numeric(df["predicted_resolution_days"], errors="coerce").fillna(0)
    df["anomaly_score"]              = pd.to_numeric(df["anomaly_score"], errors="coerce").fillna(0)
    df["is_anomaly"]                 = df["is_anomaly"].fillna(0).astype(int)
    return df


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ Grievance Intelligence")
    st.markdown("*AI-Powered CPGRAMS Analytics*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Overview",
         "🔮 Predictions",
         "🚨 Anomalies",
         "💬 Ask AI",
         "📄 Reports"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Filters (shared across pages)
    df_full = load_data()
    all_depts  = ["All"] + sorted(df_full["department"].dropna().unique().tolist())
    all_states = ["All"] + sorted(df_full["state"].dropna().unique().tolist())

    dept_filter  = st.selectbox("Department", all_depts)
    state_filter = st.selectbox("State", all_states)

    st.markdown("---")
    st.caption(f"DB records: {len(df_full):,}")
    st.caption(f"Last updated: {pd.Timestamp.now().strftime('%H:%M')}")

# ── Apply filters ─────────────────────────────────────────────────────────────
df = load_data()
if dept_filter  != "All": df = df[df["department"] == dept_filter]
if state_filter != "All": df = df[df["state"]      == state_filter]

# ── Route to pages ────────────────────────────────────────────────────────────
if page == "📊 Overview":
    from pages.overview import render
    render(df)

elif page == "🔮 Predictions":
    from pages.predictions import render
    render(df)

elif page == "🚨 Anomalies":
    from pages.anomalies import render
    render(df)

elif page == "💬 Ask AI":
    from pages.ask_ai import render
    render(df)

elif page == "📄 Reports":
    from pages.reports import render
    render(df)