"""app/pages/ask_ai.py — Page 4: Natural Language Query Interface"""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT     = Path(__file__).resolve().parents[2]
GROQ_KEY = os.getenv("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are an AI analyst for India's CPGRAMS grievance system.
You answer questions about grievance data by analysing the stats provided.
Always give specific numbers. Be concise (under 150 words).
Format key numbers in bold using **number** syntax.
If asked to show data, provide a summary table in plain text."""

EXAMPLE_QUESTIONS = [
    "Which department has the most grievances?",
    "What is the average escalation risk?",
    "How many grievances are pending resolution?",
    "Which state has the highest complaint volume?",
    "What are the top 3 root causes of grievances?",
    "How many critical urgency cases are there?",
    "What percentage of grievances are resolved?",
    "Which department has the worst SLA performance?",
]


def build_context(df: pd.DataFrame) -> str:
    """Build a concise stats summary to feed to the LLM."""
    top_dept  = df["department"].value_counts().head(5).to_dict()
    top_state = df[df["state"] != "national"]["state"].value_counts().head(5).to_dict()
    status    = df["status"].value_counts().to_dict()
    urgency   = df["urgency"].value_counts().to_dict()
    root_c    = df["root_cause"].value_counts().head(5).to_dict()

    pred = pd.to_numeric(df["predicted_resolution_days"], errors="coerce")
    esc  = pd.to_numeric(df["escalation_risk"], errors="coerce")

    return f"""
GRIEVANCE DATABASE STATS (total: {len(df):,} records):

Status: {json.dumps(status)}
Urgency: {json.dumps(urgency)}
Sentiment: {json.dumps(df['sentiment'].value_counts().to_dict())}
Top departments: {json.dumps(top_dept)}
Top states: {json.dumps(top_state)}
Top root causes: {json.dumps(root_c)}

ML Predictions:
- Avg predicted resolution: {pred.mean():.1f} days
- SLA breaches (>21d): {(pred > 21).sum():,} ({(pred > 21).mean()*100:.1f}%)
- High escalation risk (>0.5): {(esc > 0.5).sum():,}
- Avg escalation risk: {esc.mean():.3f}

Anomalies:
- Flagged records: {df['is_anomaly'].fillna(0).astype(int).sum():,}
- Anomaly rate: {df['is_anomaly'].fillna(0).mean()*100:.1f}%
"""


def ask_groq(question: str, context: str) -> str:
    if not GROQ_KEY:
        return "GROQ_API_KEY not set. Add it to your .env file."
    try:
        from groq import Groq
        try:
            client = Groq(api_key=GROQ_KEY)
        except TypeError:
            import httpx
            client = Groq(api_key=GROQ_KEY, http_client=httpx.Client())

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system",
                 "content": SYSTEM_PROMPT + "\n\nDATA CONTEXT:\n" + context},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)[:100]}"


def render(df: pd.DataFrame):
    st.markdown('<div class="main-header">💬 Ask AI</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask any question about the grievance data in plain English</div>',
                unsafe_allow_html=True)

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    context = build_context(df)

    # ── Example questions ─────────────────────────────────────────────────────
    st.subheader("💡 Example Questions")
    cols = st.columns(4)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 4]:
            if st.button(q, key=f"ex_{i}", use_container_width=True):
                with st.spinner("Thinking..."):
                    answer = ask_groq(q, context)
                st.session_state.chat_history.append({"q": q, "a": answer})

    st.markdown("---")

    # ── Chat input ────────────────────────────────────────────────────────────
    st.subheader("Ask your own question")
    with st.form("ask_form", clear_on_submit=True):
        user_q = st.text_input(
            "Your question",
            placeholder="e.g. Which state has the most high-urgency grievances?",
        )
        submitted = st.form_submit_button("Ask 🤖", use_container_width=True)

    if submitted and user_q.strip():
        with st.spinner("Groq is thinking..."):
            answer = ask_groq(user_q, context)
        st.session_state.chat_history.append({"q": user_q, "a": answer})

    # ── Chat history display ──────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("Conversation")
        for item in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(item["q"])
            with st.chat_message("assistant"):
                st.write(item["a"])

        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

    # ── Data context panel ────────────────────────────────────────────────────
    with st.expander("📊 Data context being sent to AI"):
        st.code(context, language="text")