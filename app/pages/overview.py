"""app/pages/overview.py — Page 1: Overview Dashboard"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render(df: pd.DataFrame):
    st.markdown('<div class="main-header">📊 Grievance Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time summary of citizen grievances across India</div>',
                unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)

    total      = len(df)
    pending    = (df["status"] == "pending").sum()
    resolved   = df["status"].isin(["resolved","closed"]).sum()
    escalated  = (df["status"] == "escalated").sum()
    high_risk  = (df["escalation_risk"] > 0.5).sum()

    with k1:
        st.metric("Total Grievances", f"{total:,}")
    with k2:
        st.metric("Pending", f"{pending:,}",
                  delta=f"{pending/total*100:.1f}% of total",
                  delta_color="inverse")
    with k3:
        st.metric("Resolved", f"{resolved:,}",
                  delta=f"{resolved/total*100:.1f}% rate")
    with k4:
        st.metric("Escalated", f"{escalated:,}",
                  delta_color="inverse")
    with k5:
        st.metric("High Escalation Risk", f"{high_risk:,}",
                  delta="⚠ needs attention",
                  delta_color="inverse")

    st.markdown("---")

    # ── Row 1: Status pie + Department bar ────────────────────────────────────
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.subheader("Status Distribution")
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        color_map = {
            "resolved":    "#2ecc71", "closed":      "#27ae60",
            "pending":     "#e74c3c", "in_progress":  "#f39c12",
            "escalated":   "#9b59b6",
        }
        fig = px.pie(status_counts, names="status", values="count",
                     color="status", color_discrete_map=color_map,
                     hole=0.45)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=320, showlegend=False, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Departments by Volume")
        dept = df["department"].value_counts().head(10).reset_index()
        dept.columns = ["department", "count"]
        fig = px.bar(dept.sort_values("count"), x="count", y="department",
                     orientation="h", color="count",
                     color_continuous_scale="Blues",
                     text="count",
                     labels={"count": "Grievances", "department": ""})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=370, coloraxis_showscale=False,
                          margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Monthly trend + Urgency ───────────────────────────────────────
    col3, col4 = st.columns([1.6, 1])

    with col3:
        st.subheader("Monthly Grievance Trend")
        monthly = (df.groupby(df["date_filed_dt"].dt.to_period("M"))
                    .size().reset_index())
        monthly.columns = ["month", "count"]
        monthly["month_str"] = monthly["month"].astype(str)
        fig = px.area(monthly, x="month_str", y="count",
                      color_discrete_sequence=["#1B2A5E"],
                      labels={"month_str": "Month", "count": "Grievances"})
        fig.update_traces(fillcolor="rgba(27,42,94,0.15)")
        fig.update_layout(height=310, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Urgency Breakdown")
        urg = df["urgency"].value_counts().reset_index()
        urg.columns = ["urgency", "count"]
        urg_colors = {
            "low": "#2ecc71", "medium": "#f39c12",
            "high": "#e67e22", "critical": "#e74c3c",
        }
        fig = px.bar(urg, x="urgency", y="count",
                     color="urgency", color_discrete_map=urg_colors,
                     text="count",
                     labels={"urgency": "Urgency", "count": "Count"})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=310, showlegend=False,
                          margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: State map + Root cause ────────────────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Top States by Volume")
        states = (df[df["state"] != "national"]["state"]
                   .value_counts().head(12).reset_index())
        states.columns = ["state", "count"]
        fig = px.bar(states.sort_values("count"),
                     x="count", y="state", orientation="h",
                     color="count", color_continuous_scale="Oranges",
                     text="count",
                     labels={"count": "Grievances", "state": ""})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, coloraxis_showscale=False,
                          margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.subheader("Root Cause Analysis")
        rc = df["root_cause"].value_counts().reset_index()
        rc.columns = ["root_cause", "count"]
        fig = px.pie(rc.head(8), names="root_cause", values="count",
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     hole=0.3)
        fig.update_layout(height=380, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Day of week + Sentiment ───────────────────────────────────────
    col7, col8 = st.columns(2)

    with col7:
        st.subheader("Grievances by Day of Week")
        df["weekday_name"] = df["date_filed_dt"].dt.strftime("%a")
        dow = (df["weekday_name"].value_counts()
                .reindex(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
                .reset_index())
        dow.columns = ["day", "count"]
        fig = px.bar(dow, x="day", y="count",
                     color="count", color_continuous_scale="Teal",
                     text="count",
                     labels={"day": "Day", "count": "Count"})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, coloraxis_showscale=False,
                          margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col8:
        st.subheader("Sentiment Distribution")
        sent = df["sentiment"].value_counts().reset_index()
        sent.columns = ["sentiment", "count"]
        sent_colors = {
            "negative": "#e74c3c",
            "neutral":  "#95a5a6",
            "positive": "#2ecc71",
        }
        fig = px.bar(sent, x="sentiment", y="count",
                     color="sentiment", color_discrete_map=sent_colors,
                     text="count",
                     labels={"sentiment": "Sentiment", "count": "Count"})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, showlegend=False,
                          margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Raw data table ────────────────────────────────────────────────────────
    with st.expander("📋 View Raw Data"):
        display_cols = ["grievance_id", "text", "department", "state",
                        "status", "urgency", "sentiment", "root_cause",
                        "date_filed", "source"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[display_cols].head(100),
            use_container_width=True,
            height=300,
        )