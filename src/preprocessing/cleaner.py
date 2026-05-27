"""
cleaner.py
----------
Full data cleaning pipeline for grievance records.

Steps:
  1. Load raw data from SQLite
  2. Parse & standardise dates
  3. Clean text
  4. Remove realistic duplicates
  5. Detect language
  6. Engineer features
  7. Validate values
  8. Save cleaned dataset

Run:
  python cleaner.py
  python cleaner.py --sample 10
"""

import os
import re
import sqlite3
import logging
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "data/grievances.db")
PROCESSED_PATH = "data/processed"

# ─────────────────────────────────────────────────────────────
# VALID VALUES
# ─────────────────────────────────────────────────────────────

VALID_STATUSES = {
    "pending",
    "in_progress",
    "resolved",
    "closed",
    "escalated"
}

VALID_STATES = {
    "Uttar Pradesh", "Maharashtra", "Bihar", "West Bengal",
    "Madhya Pradesh", "Rajasthan", "Tamil Nadu", "Karnataka",
    "Gujarat", "Andhra Pradesh", "Odisha", "Telangana",
    "Punjab", "Haryana", "Delhi", "Assam",
    "Jharkhand", "Chhattisgarh", "Kerala",
    "Uttarakhand", "Himachal Pradesh", "Goa",
    "Manipur", "Meghalaya", "Nagaland",
    "Tripura", "Sikkim", "Arunachal Pradesh",
    "Mizoram", "national"
}

VALID_FEEDBACK = {"Excellent", "Good", "Average", "Poor", ""}

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────

def load_from_db():

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql(
        "SELECT * FROM grievances",
        conn
    )

    conn.close()

    logger.info(f"Loaded {len(df)} records from {DB_PATH}")

    return df

# ─────────────────────────────────────────────────────────────
# STEP 2 — DATE PARSING
# ─────────────────────────────────────────────────────────────

DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%d %b %Y",
    "%d/%m/%Y",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S GMT",
]

def _parse_date(val):

    if pd.isna(val) or str(val).strip() == "":
        return None

    val = str(val).strip()

    for fmt in DATE_FORMATS:
        try:
            ts = pd.Timestamp(datetime.strptime(val, fmt))

            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)

            return ts

        except:
            continue

    try:
        ts = pd.Timestamp(val)

        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)

        return ts

    except:
        return None

def parse_dates(df):

    logger.info("Parsing dates...")

    df["date_filed_dt"] = df["date_filed"].apply(_parse_date)

    df["date_resolved_dt"] = df["date_resolved"].apply(_parse_date)

    df["date_filed_dt"] = pd.to_datetime(
        df["date_filed_dt"],
        errors="coerce"
    )

    df["date_resolved_dt"] = pd.to_datetime(
        df["date_resolved_dt"],
        errors="coerce"
    )

    unparsed = df["date_filed_dt"].isna().sum()

    logger.info(
        f"  Parsed: {len(df)-unparsed}/{len(df)} "
        f"({unparsed} invalid dates)"
    )

    df["date_filed_dt"] = df["date_filed_dt"].fillna(
        pd.Timestamp.now()
    )

    return df

# ─────────────────────────────────────────────────────────────
# STEP 3 — TEXT CLEANING
# ─────────────────────────────────────────────────────────────

_HTML_TAGS = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_PHONE = re.compile(r"\b\d{10,12}\b")
_MULTI_SPACE = re.compile(r"\s+")
_SPECIAL = re.compile(r"[^\w\s\.\,\!\?\-\:\;\'\"\(\)\/₹%]")

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.strip()

    if text == "":
        return ""

    t = text

    t = _HTML_TAGS.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _EMAIL.sub(" ", t)
    t = _PHONE.sub(" ", t)
    t = _SPECIAL.sub(" ", t)
    t = _MULTI_SPACE.sub(" ", t)

    return t.strip()

def clean_texts(df):

    logger.info("Cleaning text...")

    df["text_clean"] = df["text"].apply(clean_text)

    df["text_len"] = df["text_clean"].str.len()

    df["word_count"] = (
        df["text_clean"]
        .str.split()
        .str.len()
    )

    short_count = (df["text_len"] < 15).sum()

    logger.info(f"  Very short texts: {short_count}")

    return df

# ─────────────────────────────────────────────────────────────
# STEP 4 — REMOVE DUPLICATES
# ─────────────────────────────────────────────────────────────

def remove_duplicates(df):

    logger.info("Removing duplicates...")

    before = len(df)

    # Remove duplicate grievance IDs
    df = df.drop_duplicates(
        subset=["grievance_id"],
        keep="first"
    )

    # Realistic duplicate removal
    df = df.drop_duplicates(
        subset=[
            "text_clean",
            "department",
            "state",
            "date_filed"
        ],
        keep="first"
    )

    removed = before - len(df)

    logger.info(
        f"  Removed {removed} duplicates "
        f"({before} → {len(df)})"
    )

    return df

# ─────────────────────────────────────────────────────────────
# STEP 5 — LANGUAGE DETECTION
# ─────────────────────────────────────────────────────────────

def detect_language(df):

    logger.info("Detecting language...")

    try:

        from langdetect import detect

        def _detect(text):

            try:
                return detect(text) if len(text) > 20 else "en"

            except:
                return "unknown"

        df["lang_detected"] = (
            df["text_clean"]
            .apply(_detect)
        )

        non_english = (
            df["lang_detected"] != "en"
        ).sum()

        logger.info(
            f"  Non-English: {non_english} "
            f"({non_english/len(df)*100:.1f}%)"
        )

    except ImportError:

        logger.warning(
            "langdetect not installed"
        )

        df["lang_detected"] = "en"

    return df

# ─────────────────────────────────────────────────────────────
# STEP 6 — VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_categories(df):

    logger.info("Validating categories...")

    df["status"] = (
        df["status"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    invalid_status = ~df["status"].isin(
        VALID_STATUSES
    )

    df.loc[invalid_status, "status"] = "pending"

    invalid_state = ~df["state"].isin(
        VALID_STATES
    )

    df.loc[invalid_state, "state"] = "national"

    df["feedback_rating"] = (
        df["feedback_rating"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    invalid_feedback = ~df["feedback_rating"].isin(
        VALID_FEEDBACK
    )

    df.loc[invalid_feedback, "feedback_rating"] = ""

    logger.info(
        f"  Invalid statuses fixed: {invalid_status.sum()}"
    )

    logger.info(
        f"  Invalid states fixed: {invalid_state.sum()}"
    )

    return df

# ─────────────────────────────────────────────────────────────
# STEP 7 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_features(df):

    logger.info("Engineering features...")

    df["year"] = df["date_filed_dt"].dt.year
    df["month"] = df["date_filed_dt"].dt.month
    df["weekday"] = df["date_filed_dt"].dt.dayofweek
    df["quarter"] = df["date_filed_dt"].dt.quarter

    df["month_name"] = (
        df["date_filed_dt"]
        .dt.strftime("%b")
    )

    df["weekday_name"] = (
        df["date_filed_dt"]
        .dt.strftime("%a")
    )

    df["resolution_days"] = (
        df["date_resolved_dt"] -
        df["date_filed_dt"]
    ).dt.total_seconds() / 86400

    df["resolution_days"] = (
        df["resolution_days"]
        .clip(lower=0)
    )

    df["is_resolved"] = (
        df["status"]
        .isin(["resolved", "closed"])
        .astype(int)
    )

    df["has_amount"] = (
        df["text_clean"]
        .str.contains(
            r"rs\.?\s*\d+|₹\s*\d+",
            case=False,
            regex=True
        )
        .astype(int)
    )

    df["mentions_portal"] = (
        df["text_clean"]
        .str.contains(
            r"portal|cpgrams|online|app",
            case=False,
            regex=True
        )
        .astype(int)
    )

    logger.info(
        "  Added temporal + NLP features"
    )

    return df

# ─────────────────────────────────────────────────────────────
# STEP 8 — FINAL FILTERING
# ─────────────────────────────────────────────────────────────

def final_filter(df):

    before = len(df)

    df = df[
        df["text_len"] >= 15
    ].reset_index(drop=True)

    removed = before - len(df)

    logger.info(
        f"Dropped {removed} short records"
    )

    return df

# ─────────────────────────────────────────────────────────────
# STEP 9 — SAVE
# ─────────────────────────────────────────────────────────────

def save_processed(df):

    os.makedirs(PROCESSED_PATH, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M")

    path = (
        f"{PROCESSED_PATH}/"
        f"grievances_clean_{ts}.csv"
    )

    df.to_csv(
        path,
        index=False,
        encoding="utf-8"
    )

    logger.info(
        f"Saved cleaned data → {path}"
    )

    return path

# ─────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────

def cleaning_report(df_raw, df_clean):

    report = {
        "raw_records": len(df_raw),
        "clean_records": len(df_clean),
        "removed": len(df_raw) - len(df_clean),
        "removal_pct": round(
            (len(df_raw)-len(df_clean))
            / len(df_raw) * 100,
            1
        ),
        "sources": df_clean["source"]
            .value_counts()
            .to_dict(),

        "status_dist": df_clean["status"]
            .value_counts()
            .to_dict(),

        "avg_word_count": round(
            df_clean["word_count"].mean(),
            1
        ),

        "resolved_pct": round(
            df_clean["is_resolved"].mean() * 100,
            1
        ),

        "date_range":
            f"{df_clean['date_filed_dt'].min().date()} "
            f"→ "
            f"{df_clean['date_filed_dt'].max().date()}"
    }

    return report

# ─────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline():

    logger.info("=" * 55)
    logger.info("GRIEVANCE CLEANING PIPELINE — START")
    logger.info("=" * 55)

    df_raw = load_from_db()

    df = df_raw.copy()

    df = parse_dates(df)

    df = clean_texts(df)

    df = remove_duplicates(df)

    df = detect_language(df)

    df = validate_categories(df)

    df = engineer_features(df)

    df = final_filter(df)

    save_processed(df)

    report = cleaning_report(df_raw, df)

    logger.info("=" * 55)
    logger.info("CLEANING REPORT")
    logger.info("=" * 55)

    for k, v in report.items():
        logger.info(f"{k:<20} : {v}")

    return df

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample",
        type=int,
        default=0
    )

    args = parser.parse_args()

    df = run_pipeline()

    if args.sample > 0:

        print("\n" + "=" * 60)
        print(f"SAMPLE CLEANED RECORDS ({args.sample})")
        print("=" * 60)

        sample_df = df.sample(
            min(args.sample, len(df))
        )

        for _, r in sample_df.iterrows():

            print(f"\n[{r['source']}]")
            print(f"Department : {r['department']}")
            print(f"State      : {r['state']}")
            print(f"Status     : {r['status']}")
            print(f"Words      : {r['word_count']}")
            print(f"Text       : {r['text_clean'][:150]}...")