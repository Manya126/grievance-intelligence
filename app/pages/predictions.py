"""app/pages/predictions.py — Page 2: ML Predictions + SHAP"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@st.cache_resource
def load_models():
    try:
        import joblib
        res_model    = joblib.load(ROOT / "models_saved" / "resolution_xgb.pkl")
        esc_model    = joblib.load(ROOT / "models_saved" / "escalation_xgb.pkl")
        feature_cols = joblib.load(ROOT / "models_saved" / "feature_columns.pkl")
        return res_model, esc_model, feature_cols
    except Exception as e:
        return None, None, None


def predict_grievance(text: str, department: str, urgency: str,
                      state: str, root_cause: str,
                      res_model, esc_model, feature_cols: list) -> dict:
    """Build feature vector and run both models."""
    import re
    from datetime import datetime

    URGENCY_MAP = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    row = {col: 0 for col in feature_cols}
    row["urgency_enc"]    = URGENCY_MAP.get(urgency.lower(), 1)
    row["sentiment_enc"]  = -1
    row["word_count"]     = len(text.split())
    row["char_count"]     = len(text)
    row["has_amount"]     = int(bool(re.search(r"rs\.?\s*\d+|₹", text, re.I)))
    row["has_date_ref"]   = int(bool(re.search(r"\d+\s*(?:day|month|year|week)", text, re.I)))
    row["mentions_portal"]= int(bool(re.search(r"portal|cpgrams|online|app|website", text, re.I)))
    row["month"]          = datetime.now().month
    row["weekday"]        = datetime.now().weekday()
    row["quarter"]        = (datetime.now().month - 1) // 3 + 1
    row["is_live_data"]   = 0

    dept_col  = f"department_{department}"
    state_col = f"state_{state}"
    rc_col    = f"root_cause_{root_cause}"
    for col in [dept_col, state_col, rc_col]:
        if col in row:
            row[col] = 1

    X = pd.DataFrame([row])[feature_cols].fillna(0)
    res_days = float(res_model.predict(X)[0])
    esc_risk = float(esc_model.predict_proba(X)[0][1])

    return {"resolution_days": max(0, res_days), "escalation_risk": esc_risk, "X": X}


def render(df: pd.DataFrame):
    st.markdown('<div class="main-header">🔮 Grievance Predictor</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter a grievance to predict resolution time, escalation risk, and get SHAP explanation</div>',
                unsafe_allow_html=True)

    res_model, esc_model, feature_cols = load_models()

    if res_model is None:
        st.error("Models not found. Run `python src/models/resolution_predictor.py` first.")
        return

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.subheader("Enter Grievance Details")

        text = st.text_area(
            "Grievance text",
            placeholder="e.g. My income tax refund of Rs 45,000 has not been received after 90 days of filing ITR...",
            height=120,
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            department = st.selectbox("Department", [
                "Ministry of Railways", "Ministry of Finance",
                "Ministry of Health", "Ministry of Agriculture",
                "Ministry of Education", "Ministry of Home Affairs",
                "Ministry of Labour", "Ministry of Power",
                "Ministry of Telecommunications", "Ministry of Housing",
                "Department of Posts", "Ministry of Water Resources",
                "Ministry of Transport", "Ministry of Petroleum",
                "Ministry of Social Justice", "Other",
            ])
        with col2:
            urgency = st.selectbox("Urgency", ["medium", "low", "high", "critical"])
        with col3:
            state = st.selectbox("State", [
                "national", "Uttar Pradesh", "Maharashtra", "Bihar",
                "West Bengal", "Madhya Pradesh", "Rajasthan", "Tamil Nadu",
                "Karnataka", "Gujarat", "Andhra Pradesh", "Delhi",
                "Odisha", "Telangana", "Punjab", "Haryana", "Kerala",
            ])
        with col4:
            root_cause = st.selectbox("Root Cause", [
                "service_delay", "payment_not_received", "document_issue",
                "portal_technical_error", "staff_misconduct",
                "scheme_not_implemented", "infrastructure_gap",
                "policy_confusion", "corruption", "other",
            ])

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted and text.strip():
        with st.spinner("Running prediction..."):
            result = predict_grievance(
                text, department, urgency, state, root_cause,
                res_model, esc_model, feature_cols
            )

        res_days = result["resolution_days"]
        esc_risk = result["escalation_risk"]
        X        = result["X"]

        # ── Results ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Prediction Results")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Predicted Resolution", f"{res_days:.1f} days")
        with r2:
            st.metric("Escalation Risk", f"{esc_risk:.1%}")
        with r3:
            sla = "🔴 SLA BREACH" if res_days > 21 else "🟢 Within SLA"
            st.metric("SLA Status", sla)
        with r4:
            risk_label = (
                "🔴 CRITICAL" if esc_risk > 0.7 else
                "🟡 HIGH"     if esc_risk > 0.4 else
                "🟢 LOW"
            )
            st.metric("Risk Level", risk_label)

        # Gauges
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = res_days,
                title = {"text": "Resolution Days"},
                delta = {"reference": 21, "valueformat": ".1f"},
                gauge = {
                    "axis":  {"range": [0, 90]},
                    "bar":   {"color": "#1B2A5E"},
                    "steps": [
                        {"range": [0,  21], "color": "#2ecc71"},
                        {"range": [21, 45], "color": "#f39c12"},
                        {"range": [45, 90], "color": "#e74c3c"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 21,
                    },
                },
            ))
            fig.update_layout(height=280, margin=dict(t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_g2:
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = esc_risk * 100,
                title = {"text": "Escalation Risk %"},
                gauge = {
                    "axis":  {"range": [0, 100]},
                    "bar":   {"color": "#9b59b6"},
                    "steps": [
                        {"range": [0,  40], "color": "#2ecc71"},
                        {"range": [40, 70], "color": "#f39c12"},
                        {"range": [70,100], "color": "#e74c3c"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            fig.update_layout(height=280, margin=dict(t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)

        # ── SHAP Explanation ──────────────────────────────────────────────────
        st.subheader("🔍 SHAP Explanation — Why this prediction?")
        try:
            import shap
            explainer   = shap.TreeExplainer(res_model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_df = pd.DataFrame({
                "feature":    X.columns,
                "shap_value": shap_values[0],
            }).sort_values("shap_value", key=abs, ascending=False).head(12)

            colors = ["#e74c3c" if v > 0 else "#2ecc71"
                      for v in shap_df["shap_value"]]

            fig = go.Figure(go.Bar(
                x    = shap_df["shap_value"],
                y    = shap_df["feature"],
                orientation = "h",
                marker_color = colors,
                text = shap_df["shap_value"].round(2),
                textposition = "outside",
            ))
            fig.update_layout(
                title  = "Feature Impact on Resolution Time (SHAP Values)",
                xaxis_title = "SHAP value (positive = increases days, negative = decreases)",
                height = 420,
                margin = dict(t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🔴 Red = increases resolution time | 🟢 Green = decreases resolution time")

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

    elif submitted:
        st.warning("Please enter some grievance text to predict.")

    # ── Historical predictions from DB ────────────────────────────────────────
    st.markdown("---")
    st.subheader("Historical Predictions in Database")

    has_preds = df["predicted_resolution_days"].notna() & (df["predicted_resolution_days"] > 0)
    if has_preds.any():
        pred_df = df[has_preds][
            ["department", "state", "urgency", "status",
             "predicted_resolution_days", "escalation_risk"]
        ].copy()
        pred_df["predicted_resolution_days"] = pred_df["predicted_resolution_days"].round(1)
        pred_df["escalation_risk"]           = pred_df["escalation_risk"].round(3)
        pred_df["sla_status"] = pred_df["predicted_resolution_days"].apply(
            lambda d: "🔴 Breach" if d > 21 else "🟢 OK"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Avg Predicted Resolution",
                      f"{pred_df['predicted_resolution_days'].mean():.1f} days")
        with col_b:
            st.metric("Records Above SLA (>21d)",
                      f"{(pred_df['predicted_resolution_days'] > 21).sum():,}")

        st.dataframe(pred_df.head(50), use_container_width=True, height=300)
    else:
        st.info("No predictions in DB yet. Run `python src/models/resolution_predictor.py`")