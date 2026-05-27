
"""
prompts.py
----------
All prompt templates for the grievance classification pipeline.
Keeping prompts in one file makes them easy to iterate on
without touching pipeline logic.
"""

# ── Classification prompt ─────────────────────────────────────────────────────
CLASSIFICATION_SYSTEM = """You are an expert analyst for India's public grievance redressal system (CPGRAMS).
Your job is to classify citizen grievances into structured categories.
Always respond with valid JSON only. No explanation, no markdown, no extra text."""

CLASSIFICATION_USER = """Classify this Indian citizen grievance text into the following fields.
Return ONLY a JSON object with exactly these keys:

{{
  "department": "<one of the departments listed below>",
  "urgency": "<low | medium | high | critical>",
  "sentiment": "<negative | neutral | positive>",
  "root_cause": "<one of the root causes listed below>",
  "confidence": <0.0 to 1.0>
}}

VALID DEPARTMENTS (pick the single best match):
- Ministry of Railways
- Ministry of Finance
- Ministry of Health
- Ministry of Agriculture
- Ministry of Education
- Ministry of Home Affairs
- Ministry of Labour
- Ministry of Power
- Ministry of Telecommunications
- Ministry of Housing
- Department of Posts
- Ministry of Water Resources
- Ministry of Transport
- Ministry of Petroleum
- Ministry of Panchayati Raj
- Ministry of Social Justice
- Ministry of Commerce
- Ministry of External Affairs
- DARPG / Grievances
- Other

URGENCY DEFINITIONS:
- critical: life-threatening, emergency, involves elderly/disabled/child, illegal detention
- high: financial loss > Rs 10,000, service denied > 60 days, corruption allegation
- medium: service delayed 15-60 days, refund pending, incorrect billing
- low: general information, minor delay < 15 days, feedback

ROOT CAUSES (pick the single best match):
- service_delay          (took longer than promised)
- payment_not_received   (benefit/refund/salary not credited)
- document_issue         (certificate, ID, form problem)
- portal_technical_error (website/app not working)
- staff_misconduct       (bribery, rude behaviour, negligence)
- scheme_not_implemented (government scheme benefit not reaching citizen)
- infrastructure_gap     (physical facility missing or broken)
- policy_confusion       (unclear rules or wrong information given)
- corruption             (bribery, misappropriation of funds)
- other

EXAMPLES:

Input:
"Income tax refund of Rs 45000 pending for 90 days."

Output:
{
  "department": "Ministry of Finance",
  "urgency": "high",
  "sentiment": "negative",
  "root_cause": "payment_not_received",
  "confidence": 0.93
}

Input:
"Train delayed by 6 hours and no announcement at station."

Output:
{
  "department": "Ministry of Railways",
  "urgency": "medium",
  "sentiment": "negative",
  "root_cause": "service_delay",
  "confidence": 0.88
}

Input:
"Doctor demanded unofficial payment at district hospital."

Output:
{
  "department": "Ministry of Health",
  "urgency": "high",
  "sentiment": "negative",
  "root_cause": "corruption",
  "confidence": 0.95
}

GRIEVANCE TEXT:
\"\"\"{text}\"\"\"

IMPORTANT:
- Return ONLY valid JSON.
- No markdown.
- No explanation.
- No code fences.
- No extra text.

Return JSON now:"""


# ── Batch classification prompt (multiple grievances at once) ─────────────────
BATCH_CLASSIFICATION_USER = """Classify each of these {n} Indian citizen grievances.
Return ONLY a JSON array with {n} objects, one per grievance, in the same order.
Each object must have exactly these keys: department, urgency, sentiment, root_cause, confidence.

Use these valid values:
- department: [Ministry of Railways, Ministry of Finance, Ministry of Health, Ministry of Agriculture,
  Ministry of Education, Ministry of Home Affairs, Ministry of Labour, Ministry of Power,
  Ministry of Telecommunications, Ministry of Housing, Department of Posts, Ministry of Water Resources,
  Ministry of Transport, Ministry of Petroleum, Ministry of Panchayati Raj, Ministry of Social Justice,
  Ministry of Commerce, Ministry of External Affairs, DARPG / Grievances, Other]

- urgency: [low, medium, high, critical]

- sentiment: [negative, neutral, positive]

- root_cause: [
    service_delay,
    payment_not_received,
    document_issue,
    portal_technical_error,
    staff_misconduct,
    scheme_not_implemented,
    infrastructure_gap,
    policy_confusion,
    corruption,
    other
]

- confidence: float 0.0-1.0

GRIEVANCES:
{grievances_json}

IMPORTANT:
- Return ONLY valid JSON array.
- No markdown.
- No explanation.
- No extra text.
- Output length MUST equal input length.

Return JSON array now:"""


# ── Report generation prompt ───────────────────────────────────────────────────
WEEKLY_REPORT_SYSTEM = """You are a senior public policy analyst writing official weekly reports
for the Government of India's Department of Administrative Reforms & Public Grievances (DARPG).
Write in formal, clear government English. Be specific with numbers. Be actionable."""

WEEKLY_REPORT_USER = """Write a formal Weekly Grievance Intelligence Report based on this data.

REPORTING PERIOD: {period}
TOTAL GRIEVANCES ANALYSED: {total}

STATISTICS:
{stats_json}

TOP ISSUES THIS WEEK:
{top_issues}

SLA BREACHES (departments exceeding 21-day resolution target):
{sla_breaches}

ANOMALIES DETECTED:
{anomalies}

Write the report with these sections:

1. Executive Summary
(3-4 sentences, key numbers)

2. Critical Alerts
(if any urgency=critical grievances)

3. Top 5 Issue Areas
(with counts and trend vs last week)

4. Department Performance Scorecard
(table format: dept | received | resolved | avg_days | SLA_status)

5. Geographic Hotspots
(states with highest volume)

6. Recommended Interventions
(3-5 specific, actionable recommendations)

7. Next Week Monitoring Focus

Keep total length under 800 words.
Use formal government language."""

