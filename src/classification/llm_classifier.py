# =============================================================================
# llm_classifier.py
# =============================================================================

import os
import re
import json
import time
import sqlite3
import logging
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DB_PATH    = ROOT / "data" / "grievances.db"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY   = os.getenv("GROQ_API_KEY")

VALID_DEPARTMENTS = [
    "Ministry of Railways", "Ministry of Finance", "Ministry of Health",
    "Ministry of Agriculture", "Ministry of Education", "Ministry of Home Affairs",
    "Ministry of Labour", "Ministry of Power", "Ministry of Telecommunications",
    "Ministry of Housing", "Department of Posts", "Ministry of Water Resources",
    "Ministry of Transport", "Ministry of Petroleum", "Ministry of Panchayati Raj",
    "Ministry of Social Justice", "Ministry of Commerce", "Ministry of External Affairs",
    "DARPG / Grievances", "Other",
]
VALID_URGENCY     = ["low", "medium", "high", "critical"]
VALID_SENTIMENT   = ["negative", "neutral", "positive"]
VALID_ROOT_CAUSES = [
    "service_delay", "payment_not_received", "document_issue",
    "portal_technical_error", "staff_misconduct", "scheme_not_implemented",
    "infrastructure_gap", "policy_confusion", "corruption", "other",
]

RULE_KEYWORDS = {
    "Ministry of Railways":            ["train", "railway", "station", "pnr", "ticket", "coach", "platform"],
    "Ministry of Finance":             ["bank", "income tax", "refund", "epf", "loan", "gst", "atm"],
    "Ministry of Health":              ["hospital", "doctor", "medicine", "ambulance", "health"],
    "Ministry of Education":           ["school", "college", "teacher", "scholarship", "exam"],
    "Ministry of Power":               ["electricity", "power", "transformer", "meter", "voltage"],
    "Department of Posts":             ["post office", "parcel", "speed post", "courier"],
    "Ministry of Telecommunications":  ["network", "internet", "sim", "tower", "broadband", "mobile"],
    "Ministry of Water Resources":     ["water", "pipeline", "drainage", "sewage"],
    "Ministry of Transport":           ["bus", "road", "traffic", "license"],
    "Ministry of Labour":              ["salary", "labour", "worker", "wages"],
}

def rule_based_classify(text: str) -> dict:
    text_lower = text.lower()
    if any(x in text_lower for x in ["launched", "announced", "initiative", "campaign",
        "programme", "program", "inaugurated", "press release",
        "minister said", "government announced"]):
        return {"is_grievance": False, "department": "Other", "urgency": "low",
                "sentiment": "neutral", "root_cause": "other", "confidence": 0.82,
                "method": "rule_news_filter"}

    best_score, dept = 0, "Other"
    for department, keywords in RULE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score, dept = score, department

    urgency = "medium"
    if any(x in text_lower for x in ["death", "suicide", "critical", "fraud", "bribe"]):
        urgency = "critical"
    elif any(x in text_lower for x in ["pending", "not received", "refund", "delay", "harassment"]):
        urgency = "high"

    sentiment = "negative"
    if any(x in text_lower for x in ["thank you", "resolved", "appreciate"]):
        sentiment = "positive"

    root_cause = "other"
    for rc, keywords in {
        "service_delay":          ["delay", "late", "pending", "waiting"],
        "payment_not_received":   ["refund", "salary", "scholarship", "payment"],
        "document_issue":         ["certificate", "aadhaar", "pan", "document"],
        "portal_technical_error": ["website", "portal", "server", "login"],
        "staff_misconduct":       ["rude", "harassment", "misbehave"],
        "infrastructure_gap":     ["road", "network", "pipeline", "transformer"],
        "corruption":             ["corruption", "bribe", "fraud"],
    }.items():
        if any(k in text_lower for k in keywords):
            root_cause = rc
            break

    return {"is_grievance": True, "department": dept, "urgency": urgency,
            "sentiment": sentiment, "root_cause": root_cause,
            "confidence": round(min(0.45 + best_score * 0.12, 0.88), 2),
            "method": "rule_based"}


def _validate(result: dict) -> dict:
    if not isinstance(result, dict):
        return rule_based_classify("")
    result["is_grievance"] = bool(result.get("is_grievance", True))
    result["department"]   = result.get("department", "Other")
    if result["department"] not in VALID_DEPARTMENTS:
        result["department"] = "Other"
    result["urgency"] = result.get("urgency", "medium").lower()
    if result["urgency"] not in VALID_URGENCY:
        result["urgency"] = "medium"
    result["sentiment"] = result.get("sentiment", "neutral").lower()
    if result["sentiment"] not in VALID_SENTIMENT:
        result["sentiment"] = "neutral"
    result["root_cause"] = result.get("root_cause", "other")
    if result["root_cause"] not in VALID_ROOT_CAUSES:
        result["root_cause"] = "other"
    # FIX 1: deterministic confidence — no random noise
    try:
        result["confidence"] = float(result.get("confidence", 0.5))
    except Exception:
        result["confidence"] = 0.5
    result["confidence"] = round(min(1.0, max(0.0, result["confidence"])), 2)
    return result


def _call_groq(text: str) -> dict | None:
    try:
        from groq import Groq
        try:
            client = Groq(api_key=GROQ_KEY)
        except TypeError:
            import httpx
            client = Groq(api_key=GROQ_KEY, http_client=httpx.Client())

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[
                {"role": "system", "content": """
You are an expert analyst for India's public grievance redressal system (CPGRAMS).
Identify department, urgency, sentiment, and root cause.

DEPARTMENT MAPPING:
Railway/train/ticket/PNR → Ministry of Railways
Bank/refund/EPF/tax/GST → Ministry of Finance
Electricity/power/meter → Ministry of Power
Hospital/doctor/medicine → Ministry of Health
Scholarship/school/exam → Ministry of Education
Internet/network/SIM → Ministry of Telecommunications
Road/bus/traffic/license → Ministry of Transport
Water/pipeline/drainage → Ministry of Water Resources
Post office/parcel → Department of Posts

URGENCY: low=minor issue | medium=normal complaint | high=financial loss/delay>60d | critical=life-threatening
ROOT CAUSE: service_delay | payment_not_received | document_issue | portal_technical_error | staff_misconduct | infrastructure_gap | corruption | other
Set is_grievance=false for government news/announcements.
Return ONLY valid JSON. No markdown.
"""},
                {"role": "user", "content": f"""
Classify this Indian public grievance. Return ONLY JSON:

{{"is_grievance": true, "department": "...", "urgency": "...", "sentiment": "...", "root_cause": "...", "confidence": 0.0}}

VALID departments: {", ".join(VALID_DEPARTMENTS)}

EXAMPLES:
"Railway ticket refund pending 3 months" → {{"is_grievance":true,"department":"Ministry of Railways","urgency":"high","sentiment":"negative","root_cause":"payment_not_received","confidence":0.91}}
"Electricity bill overcharged 2 months" → {{"is_grievance":true,"department":"Ministry of Power","urgency":"high","sentiment":"negative","root_cause":"payment_not_received","confidence":0.88}}
"Scholarship not credited 5 months" → {{"is_grievance":true,"department":"Ministry of Education","urgency":"high","sentiment":"negative","root_cause":"payment_not_received","confidence":0.93}}
"Government launched railway modernization" → {{"is_grievance":false,"department":"Other","urgency":"low","sentiment":"neutral","root_cause":"other","confidence":0.84}}

TEXT: {text}
"""},
            ],
        )
        raw = re.sub(r"```json|```", "", completion.choices[0].message.content.strip()).strip()
        parsed = json.loads(raw)
        parsed["method"] = "groq"
        return _validate(parsed)
    except json.JSONDecodeError as e:
        logger.warning(f"Groq JSON error: {e}")
        return None
    except Exception as e:
        err = str(e)
        if "429" in err or "rate" in err.lower():
            logger.warning("Groq rate limit — waiting 30s...")
            time.sleep(30)
        else:
            logger.warning(f"Groq error: {type(e).__name__}: {err[:100]}")
        return None


def _classify_batch_groq(texts: list) -> list | None:
    try:
        from groq import Groq
        try:
            client = Groq(api_key=GROQ_KEY)
        except TypeError:
            import httpx
            client = Groq(api_key=GROQ_KEY, http_client=httpx.Client())

        n = len(texts)
        numbered = "\n".join(f"{i+1}. {t[:300]}" for i, t in enumerate(texts))
        prompt = (
            f"Classify these {n} Indian grievances. "
            f"Return a JSON array of exactly {n} objects with keys: "
            "is_grievance, department, urgency, sentiment, root_cause, confidence.\n\n"
            f"Valid departments: {', '.join(VALID_DEPARTMENTS)}\n"
            "urgency: low|medium|high|critical  |  sentiment: negative|neutral|positive\n"
            f"root_cause: {', '.join(VALID_ROOT_CAUSES)}\n"
            "Set is_grievance=false for news/announcements.\n\n"
            f"GRIEVANCES:\n{numbered}\n\nReturn ONLY JSON array:"
        )
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=n * 100,
        )
        raw = re.sub(r"```json|```", "", completion.choices[0].message.content.strip()).strip()
        results = json.loads(raw)
        if isinstance(results, dict):
            for key in ["results", "classifications", "data"]:
                if key in results and isinstance(results[key], list):
                    results = results[key]
                    break
        if isinstance(results, list) and len(results) == n:
            validated = [_validate(r) for r in results]
            for r in validated:
                r["method"] = "groq_batch"
            return validated
        logger.warning(f"Batch count mismatch: got {len(results) if isinstance(results, list) else type(results)}, need {n}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Groq batch JSON error: {e}")
        return None
    except Exception as e:
        err = str(e)
        if "429" in err or "rate" in err.lower():
            logger.warning("Groq rate limit (batch) — waiting 60s...")
            time.sleep(60)
        else:
            logger.warning(f"Groq batch error: {type(e).__name__}: {err[:100]}")
        return None


def classify_one(text: str) -> dict:
    if GROQ_KEY:
        result = _call_groq(text)
        if result:
            return result
    return rule_based_classify(text)


def classify_db(limit: int = None, batch_size: int = 10, force: bool = False) -> pd.DataFrame:
    conn  = sqlite3.connect(DB_PATH)
    where = "" if force else "WHERE category IS NULL"
    query = f"SELECT grievance_id, text, source FROM grievances {where}"
    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql(query, conn)
    logger.info(f"Records to classify: {len(df)}")
    if df.empty:
        logger.info("Nothing to classify. Use --force to reclassify.")
        conn.close()
        return pd.DataFrame()

    if GROQ_KEY:
        logger.info(f"Groq batch mode — {batch_size} records per API call")
    else:
        logger.info("No API key — rule-based only")

    results  = []
    groq_ok  = 0
    rule_ok  = 0
    total    = len(df)
    n_batches = (total + batch_size - 1) // batch_size

    for batch_num, start in enumerate(range(0, total, batch_size)):
        batch         = df.iloc[start:start + batch_size]
        texts         = batch["text"].tolist()
        batch_results = None

        if GROQ_KEY:
            batch_results = _classify_batch_groq(texts)
            if batch_results:
                groq_ok += len(batch_results)

        if batch_results is None:
            batch_results = [rule_based_classify(t) for t in texts]
            rule_ok += len(batch_results)

        for (_, row), result in zip(batch.iterrows(), batch_results):
            result["grievance_id"] = row["grievance_id"]
            result["source"]       = row["source"]
            results.append(result)

        logger.info(f"  Batch {batch_num+1}/{n_batches} | groq={groq_ok} rules={rule_ok}")

        if GROQ_KEY and batch_results and batch_results[0].get("method", "").startswith("groq"):
            time.sleep(4.0)

    result_df = pd.DataFrame(results)

    # FIX 2: Write back to DB
    cursor  = conn.cursor()
    updated = 0
    for _, row in result_df.iterrows():
        cursor.execute(
            "UPDATE grievances SET category=?, urgency=?, sentiment=?, root_cause=? WHERE grievance_id=?",
            (row.get("department","Other"), row.get("urgency","medium"),
             row.get("sentiment","negative"), row.get("root_cause","other"),
             row["grievance_id"]),
        )
        updated += cursor.rowcount
    conn.commit()
    conn.close()
    logger.info(f"DB updated: {updated} records")

    save_path = ROOT / "data" / "processed" / "classified_output.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(save_path, index=False)
    logger.info(f"CSV saved → {save_path}")

    logger.info(f"\nSummary: groq_batch={groq_ok} | rule_based={rule_ok}")
    if "department" in result_df.columns:
        logger.info(f"Urgency:   {result_df['urgency'].value_counts().to_dict()}")
        logger.info(f"Sentiment: {result_df['sentiment'].value_counts().to_dict()}")

    return result_df


def test_text(text: str):
    print("\n" + "=" * 60)
    print("INPUT:")
    print(text)
    print("\nOUTPUT:")
    print(json.dumps(classify_one(text), indent=2, ensure_ascii=False))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--test",  type=str, default=None)
    args = parser.parse_args()

    if args.test:
        test_text(args.test)
    else:
        classify_db(limit=args.limit, batch_size=args.batch, force=args.force)