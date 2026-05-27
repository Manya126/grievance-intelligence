"""
anomaly_detector.py
-------------------
Week 5 — Anomaly Detection

Two complementary anomaly detection approaches:

DETECTOR 1: Prophet Time-Series (volume surge detection)
  - Models daily grievance volume as a time series
  - Flags days where actual volume exceeds forecast by 2+ std deviations
  - Use case: "Sudden spike in Railway complaints on 15 March — investigate"

DETECTOR 2: Isolation Forest (outlier grievance detection)
  - Finds individual grievances that are statistically unusual
  - Based on text features, urgency, resolution time, escalation risk
  - Use case: "This complaint has unusually high escalation risk + long text"

DETECTOR 3: SLA Breach Alert
  - Flags departments where avg predicted resolution > 21 days
  - Simple but immediately actionable for dashboard

All results written to DB (is_anomaly, anomaly_score columns).

Run:
  python anomaly_detector.py          # run all detectors
  python anomaly_detector.py --stats  # show anomaly summary
"""

import os, sys, logging, argparse, sqlite3, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH        = os.getenv("DB_PATH", "data/grievances.db")
PROCESSED_PATH = "data/processed"
os.makedirs(PROCESSED_PATH, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM grievances", conn)
    conn.close()

    df["date_filed_dt"] = pd.to_datetime(df["date_filed"], errors="coerce", utc=False)
    if df["date_filed_dt"].dt.tz is not None:
        df["date_filed_dt"] = df["date_filed_dt"].dt.tz_localize(None)
    df["date_filed_dt"] = df["date_filed_dt"].fillna(pd.Timestamp.now())

    df["text_len"]    = df["text"].fillna("").str.len()
    df["word_count"]  = df["text"].fillna("").str.split().str.len()
    df["has_amount"]  = df["text"].fillna("").str.contains(
                            r"rs\.?\s*\d+|₹", case=False, regex=True).astype(int)

    logger.info(f"Loaded {len(df)} records")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DETECTOR 1: PROPHET TIME-SERIES ANOMALY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_prophet_detector(df: pd.DataFrame, department: str = None) -> pd.DataFrame:
    """
    Detect volume anomalies using rolling Z-score (replaces Prophet to avoid
    stan_backend compatibility issues). Same concept — flags days where
    actual volume deviates more than 2 std from the 7-day rolling mean.
    """
    logger.info("=" * 55)
    logger.info("DETECTOR 1: Volume Anomaly (Rolling Z-Score)")
    logger.info("=" * 55)

    subset = df[df["department"] == department].copy() if department else df.copy()
    label  = department or "All Departments"

    # Daily aggregation
    daily = (subset
             .groupby(subset["date_filed_dt"].dt.date)
             .size()
             .reset_index()
             .rename(columns={"date_filed_dt": "ds", 0: "actual"}))
    daily["ds"] = pd.to_datetime(daily["ds"])
    daily = daily.sort_values("ds").reset_index(drop=True)

    if len(daily) < 7:
        logger.warning("Not enough days for time-series analysis (need 7+).")
        return pd.DataFrame()

    logger.info("Analysing %d days of volume data (%s)", len(daily), label)

    # Rolling 7-day stats
    daily["rolling_mean"] = daily["actual"].rolling(7, min_periods=3, center=True).mean()
    daily["rolling_std"]  = daily["actual"].rolling(7, min_periods=3, center=True).std().fillna(1)
    daily["yhat"]         = daily["rolling_mean"]
    daily["yhat_upper"]   = daily["rolling_mean"] + 2 * daily["rolling_std"]
    daily["yhat_lower"]   = (daily["rolling_mean"] - 2 * daily["rolling_std"]).clip(lower=0)
    daily["z_score"]      = ((daily["actual"] - daily["rolling_mean"]) /
                              daily["rolling_std"].clip(lower=0.5)).round(3)
    daily["is_volume_anomaly"] = daily["z_score"].abs() > 2.0
    daily["anomaly_score"]     = daily["z_score"]

    n = daily["is_volume_anomaly"].sum()
    logger.info("Volume anomalies: %d days", n)

    if n > 0:
        top = daily[daily["is_volume_anomaly"]].sort_values("z_score", ascending=False)
        for _, row in top.head(5).iterrows():
            direction = "SURGE" if row["z_score"] > 0 else "DROP"
            logger.info("  %s  actual=%.0f  expected=%.1f  z=%.2f  %s",
                        row["ds"].date(), row["actual"], row["rolling_mean"],
                        row["z_score"], direction)

    path = PROCESSED_PATH + "/prophet_anomalies.csv"
    daily[["ds","actual","yhat","yhat_upper","yhat_lower",
           "is_volume_anomaly","anomaly_score"]].to_csv(path, index=False)
    logger.info("Saved -> %s", path)
    return daily



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DETECTOR 2: ISOLATION FOREST (per-record outliers)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

URGENCY_MAP = {"low": 0, "medium": 1, "high": 2, "critical": 3}

def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Detect outlier grievances using Isolation Forest.
    contamination: expected fraction of anomalies (default 5%)
    """
    logger.info("=" * 55)
    logger.info("DETECTOR 2: Isolation Forest (per-record outliers)")
    logger.info("=" * 55)

    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import LabelEncoder

    # Build feature matrix for anomaly detection
    features = pd.DataFrame({
        "text_len":         df["text_len"].fillna(0),
        "word_count":       df["word_count"].fillna(0),
        "has_amount":       df["has_amount"].fillna(0),
        "urgency_enc":      df["urgency"].map(URGENCY_MAP).fillna(1),
        "is_resolved":      (df["status"].isin(["resolved","closed"])).astype(int),
        "escalation_risk":  pd.to_numeric(df["escalation_risk"], errors="coerce").fillna(0.1),
        "pred_resolution":  pd.to_numeric(df["predicted_resolution_days"], errors="coerce").fillna(30),
        "month":            df["date_filed_dt"].dt.month.fillna(6),
        "weekday":          df["date_filed_dt"].dt.dayofweek.fillna(2),
    })

    features = features.fillna(0)

    logger.info(f"Running Isolation Forest on {len(features)} records (contamination={contamination})")

    iso = IsolationForest(
        n_estimators  = 200,
        contamination = contamination,
        random_state  = 42,
        n_jobs        = -1,
    )
    df = df.copy()
    df["if_label"] = iso.fit_predict(features)      # -1 = anomaly, 1 = normal
    df["if_score"] = iso.score_samples(features)    # lower = more anomalous

    df["is_if_anomaly"]  = (df["if_label"] == -1).astype(int)
    df["if_anomaly_score"] = (-df["if_score"]).round(4)  # flip: higher = more anomalous

    n_anomalies = df["is_if_anomaly"].sum()
    logger.info(f"Isolation Forest anomalies: {n_anomalies} records ({n_anomalies/len(df)*100:.1f}%)")

    # Show top anomalies
    top_anomalies = (df[df["is_if_anomaly"] == 1]
                     .sort_values("if_anomaly_score", ascending=False)
                     .head(10))
    logger.info("Top 10 anomalous records:")
    for _, row in top_anomalies.iterrows():
        logger.info(f"  [{row.get('department','?')[:25]:<25}] "
                    f"score={row['if_anomaly_score']:.3f} | "
                    f"{str(row.get('text',''))[:70]}")

    # Department breakdown of anomalies
    dept_anomalies = (df[df["is_if_anomaly"] == 1]["department"]
                      .value_counts().head(8))
    logger.info(f"\nAnomaly count by department:")
    for dept, cnt in dept_anomalies.items():
        logger.info(f"  {dept:<40} {cnt}")

    # Save results
    path = f"{PROCESSED_PATH}/isolation_forest_anomalies.csv"
    df[df["is_if_anomaly"] == 1][
        ["grievance_id","text","department","state","urgency",
         "if_anomaly_score","escalation_risk","predicted_resolution_days"]
    ].to_csv(path, index=False)
    logger.info(f"Isolation Forest results saved → {path}")

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DETECTOR 3: SLA BREACH ALERT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_sla_detector(df: pd.DataFrame, sla_days: int = 21) -> pd.DataFrame:
    """Flag departments consistently breaching the 21-day SLA."""
    logger.info("=" * 55)
    logger.info(f"DETECTOR 3: SLA Breach Alert (>{sla_days} days)")
    logger.info("=" * 55)

    pred_col = pd.to_numeric(df["predicted_resolution_days"], errors="coerce")

    sla_df = (df.assign(pred_days=pred_col)
               .groupby("department")
               .agg(
                   total        = ("grievance_id", "count"),
                   avg_pred_days= ("pred_days", "mean"),
                   sla_breaches = ("pred_days", lambda x: (x > sla_days).sum()),
                   high_urgency = ("urgency", lambda x: (x == "high").sum()),
               )
               .reset_index())

    sla_df["breach_rate"] = (sla_df["sla_breaches"] / sla_df["total"] * 100).round(1)
    sla_df["sla_status"]  = sla_df["avg_pred_days"].apply(
        lambda d: "🔴 CRITICAL" if d > 45 else ("🟡 WARNING" if d > 21 else "🟢 OK")
    )
    sla_df = sla_df.sort_values("avg_pred_days", ascending=False)

    logger.info(f"\nDepartment SLA Scorecard (SLA = {sla_days} days):")
    logger.info(f"  {'Department':<40} {'Avg Days':>8} {'Breach%':>8} {'Status'}")
    logger.info(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*10}")
    for _, row in sla_df.head(15).iterrows():
        logger.info(f"  {row['department']:<40} {row['avg_pred_days']:>8.1f} "
                    f"{row['breach_rate']:>7.1f}% {row['sla_status']}")

    critical = sla_df[sla_df["avg_pred_days"] > 45]
    warning  = sla_df[(sla_df["avg_pred_days"] > 21) & (sla_df["avg_pred_days"] <= 45)]
    ok       = sla_df[sla_df["avg_pred_days"] <= 21]

    logger.info(f"\n  🔴 Critical (>45d): {len(critical)} departments")
    logger.info(f"  🟡 Warning (21-45d): {len(warning)} departments")
    logger.info(f"  🟢 OK (≤21d):        {len(ok)} departments")

    path = f"{PROCESSED_PATH}/sla_scorecard.csv"
    sla_df.to_csv(path, index=False)
    logger.info(f"SLA scorecard saved → {path}")

    return sla_df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WRITE ANOMALY FLAGS TO DB
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def write_anomalies_to_db(df_with_anomalies: pd.DataFrame) -> int:
    """Write is_anomaly + anomaly_score back to DB for dashboard use."""
    if "is_if_anomaly" not in df_with_anomalies.columns:
        logger.warning("No Isolation Forest results to write")
        return 0

    conn    = sqlite3.connect(DB_PATH)
    cursor  = conn.cursor()
    updated = 0

    for _, row in df_with_anomalies.iterrows():
        cursor.execute("""
            UPDATE grievances
            SET is_anomaly   = ?,
                anomaly_score = ?
            WHERE grievance_id = ?
        """, (
            int(row.get("is_if_anomaly", 0)),
            float(row.get("if_anomaly_score", 0.0)),
            row["grievance_id"],
        ))
        updated += cursor.rowcount

    conn.commit()
    conn.close()
    logger.info(f"Anomaly flags written to DB: {updated} records")
    return updated


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANOMALY SUMMARY STATS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_stats():
    conn = sqlite3.connect(DB_PATH)
    total     = pd.read_sql("SELECT COUNT(*) as n FROM grievances", conn).iloc[0,0]
    anomalies = pd.read_sql("SELECT COUNT(*) as n FROM grievances WHERE is_anomaly=1", conn).iloc[0,0]
    top_dept  = pd.read_sql("""
        SELECT department, COUNT(*) as n
        FROM grievances WHERE is_anomaly=1
        GROUP BY department ORDER BY n DESC LIMIT 5
    """, conn)
    conn.close()

    print(f"\nANOMALY SUMMARY")
    print(f"  Total records:    {total:,}")
    print(f"  Flagged anomalies:{anomalies:,} ({anomalies/total*100:.1f}%)")
    print(f"\nTop departments with anomalies:")
    print(top_dept.to_string(index=False))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MASTER PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pipeline():
    logger.info("=" * 55)
    logger.info("WEEK 5 — ANOMALY DETECTION PIPELINE")
    logger.info("=" * 55)

    df = load_data()

    # Detector 1: Prophet volume anomaly
    prophet_df = run_prophet_detector(df)

    # Detector 2: Isolation Forest per-record
    df_with_anomalies = run_isolation_forest(df)

    # Detector 3: SLA breach scorecard
    sla_df = run_sla_detector(df)

    # Write anomaly flags to DB
    write_anomalies_to_db(df_with_anomalies)

    logger.info("=" * 55)
    logger.info("ANOMALY DETECTION COMPLETE")
    logger.info("=" * 55)
    logger.info("Files saved:")
    logger.info(f"  data/processed/prophet_anomalies.csv")
    logger.info(f"  data/processed/isolation_forest_anomalies.csv")
    logger.info(f"  data/processed/sla_scorecard.csv")
    logger.info("Next: python src/reporting/report_generator.py  (Week 6)")

    return prophet_df, df_with_anomalies, sla_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats",      action="store_true", help="Show anomaly stats from DB")
    parser.add_argument("--department", type=str, default=None,
                        help="Run Prophet for one department only")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    else:
        run_pipeline()