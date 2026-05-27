"""app/pages/anomalies.py — Page 3: Anomaly Detection"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def render(df: pd.DataFrame):
    st.markdown('<div class="main-header">🚨 Anomaly Detection</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Volume surges, outlier grievances, and SLA breach alerts</div>',
                unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_anomalies = df["is_anomaly"].sum()
    anomaly_rate    = df["is_anomaly"].mean() * 100
    high_esc        = (df["escalation_risk"] > 0.5).sum()
    avg_days        = df["predicted_resolution_days"].mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Anomalous Records", f"{int(total_anomalies):,}",
                  delta=f"{anomaly_rate:.1f}% of total",
                  delta_color="inverse")
    with k2:
        st.metric("High Escalation Risk", f"{int(high_esc):,}",
                  delta_color="inverse")
    with k3:
        st.metric("Avg Predicted Resolution", f"{avg_days:.1f} days",
                  delta="SLA: 21 days", delta_color="inverse")
    with k4:
        sla_breaches = (df["predicted_resolution_days"] > 21).sum()
        st.metric("SLA Breaches", f"{int(sla_breaches):,}",
                  delta_color="inverse")

    st.markdown("---")

    # ── Volume time series ────────────────────────────────────────────────────
    st.subheader("📈 Daily Volume with Surge Detection")

    vol_path = ROOT / "data" / "processed" / "prophet_anomalies.csv"
    if vol_path.exists():
        vol_df = pd.read_csv(vol_path, parse_dates=["ds"])

        fig = go.Figure()
        normal = vol_df[~vol_df["is_volume_anomaly"]]
        surges = vol_df[vol_df["is_volume_anomaly"]]

        fig.add_trace(go.Scatter(
            x=vol_df["ds"], y=vol_df["yhat_upper"],
            mode="lines", name="Upper bound (2σ)",
            line=dict(color="#BDC3C7", dash="dash", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=vol_df["ds"], y=vol_df["yhat_lower"],
            mode="lines", name="Lower bound",
            line=dict(color="#BDC3C7", dash="dash", width=1),
            fill="tonexty", fillcolor="rgba(189,195,199,0.2)",
        ))
        fig.add_trace(go.Scatter(
            x=normal["ds"], y=normal["actual"],
            mode="lines+markers", name="Daily Volume",
            line=dict(color="#1B2A5E", width=1.5),
            marker=dict(size=3),
        ))
        if len(surges) > 0:
            fig.add_trace(go.Scatter(
                x=surges["ds"], y=surges["actual"],
                mode="markers", name="🔴 Surge",
                marker=dict(color="#e74c3c", size=14, symbol="star"),
            ))

        fig.update_layout(
            height=380,
            xaxis_title="Date",
            yaxis_title="Daily Grievance Count",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(surges) > 0:
            st.markdown("**Surge days detected:**")
            for _, row in surges.iterrows():
                direction = "↑ SURGE" if row["anomaly_score"] > 0 else "↓ DROP"
                st.markdown(
                    f'<div class="alert-box">📅 <b>{row["ds"].date()}</b> — '
                    f'Actual: <b>{row["actual"]:.0f}</b> vs Expected: {row["yhat"]:.1f} '
                    f'| Z-score: {row["anomaly_score"]:.2f} | {direction}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("Volume anomaly file not found. Run `python src/models/anomaly_detector.py`")

    st.markdown("---")

    # ── Isolation Forest outliers ─────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Anomaly Score Distribution")
        scores = df["anomaly_score"]
        fig = px.histogram(
            x=scores, nbins=30,
            color_discrete_sequence=["#e74c3c"],
            labels={"x": "Anomaly Score"},
        )
        threshold = scores.quantile(0.95)
        fig.add_vline(x=threshold, line_dash="dash", line_color="darkred",
                      annotation_text=f"95th pct: {threshold:.3f}")
        fig.update_layout(height=320, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏛️ Anomalies by Department")
        anom_dept = (df[df["is_anomaly"] == 1]["department"]
                     .value_counts().head(10).reset_index())
        anom_dept.columns = ["department", "count"]
        fig = px.bar(
            anom_dept.sort_values("count"),
            x="count", y="department", orientation="h",
            color="count", color_continuous_scale="Reds",
            text="count",
            labels={"count": "Anomalies", "department": ""},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=320, coloraxis_showscale=False,
                          margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── SLA Scorecard ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 SLA Breach Scorecard")

    sla_path = ROOT / "data" / "processed" / "sla_scorecard.csv"
    if sla_path.exists():
        sla_df = pd.read_csv(sla_path).sort_values("avg_pred_days", ascending=False)

        color_map_status = {
            "🔴 CRITICAL": "#e74c3c",
            "🟡 WARNING":  "#f39c12",
            "🟢 OK":       "#2ecc71",
        }
        sla_df["bar_color"] = sla_df["sla_status"].map(color_map_status).fillna("#95a5a6")

        fig = go.Figure(go.Bar(
            x=sla_df["avg_pred_days"],
            y=sla_df["department"],
            orientation="h",
            marker_color=sla_df["bar_color"],
            text=sla_df["avg_pred_days"].round(1),
            textposition="outside",
        ))
        fig.add_vline(x=21, line_dash="dash", line_color="black",
                      annotation_text="21-day SLA")
        fig.update_layout(
            height=500,
            xaxis_title="Avg Predicted Resolution (days)",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("🔴 Critical (>45d)",
                      f"{(sla_df['avg_pred_days']>45).sum()} depts")
        with col_s2:
            st.metric("🟡 Warning (21-45d)",
                      f"{((sla_df['avg_pred_days']>21)&(sla_df['avg_pred_days']<=45)).sum()} depts")
        with col_s3:
            st.metric("🟢 OK (≤21d)",
                      f"{(sla_df['avg_pred_days']<=21).sum()} depts")

        with st.expander("View full SLA table"):
            st.dataframe(sla_df[["department","avg_pred_days","breach_rate","sla_status"]],
                         use_container_width=True)
    else:
        st.info("SLA scorecard not found. Run `python src/models/anomaly_detector.py`")

    # ── Top anomalous records table ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚠️ Top Anomalous Records")
    top_anom = (df[df["is_anomaly"] == 1]
                .sort_values("anomaly_score", ascending=False)
                .head(20)[["text","department","state","urgency",
                           "anomaly_score","escalation_risk"]])
    top_anom["text"] = top_anom["text"].str[:80] + "..."
    top_anom["anomaly_score"]   = top_anom["anomaly_score"].round(3)
    top_anom["escalation_risk"] = top_anom["escalation_risk"].round(3)
    st.dataframe(top_anom, use_container_width=True, height=300)