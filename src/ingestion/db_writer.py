"""
db_writer.py
------------
Creates the SQLite database schema and handles all writes.
All collectors (datagov, twitter, synthetic) call save_to_db()
so data always lands in one place with a consistent schema.

Schema: grievances table
    grievance_id    TEXT PRIMARY KEY
    text            TEXT
    department      TEXT
    state           TEXT
    status          TEXT
    date_filed      TEXT   (ISO format)
    date_resolved   TEXT   (ISO format, empty if unresolved)
    feedback_rating TEXT
    source          TEXT   (datagov_cpgrams | twitter | synthetic | reddit)
    scraped_at      TEXT   (ISO format)

    -- filled by classifier (Module 2, Week 3)
    category        TEXT
    urgency         TEXT   (low | medium | high | critical)
    sentiment       TEXT   (positive | negative | neutral)
    root_cause      TEXT

    -- filled by ML model (Module 3, Week 4)
    predicted_resolution_days  REAL
    escalation_risk            REAL   (0.0 – 1.0)
"""

import os
import sqlite3
import logging
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "data/grievances.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS grievances (
    grievance_id                TEXT PRIMARY KEY,
    text                        TEXT,
    department                  TEXT,
    state                       TEXT,
    status                      TEXT,
    date_filed                  TEXT,
    date_resolved               TEXT,
    feedback_rating             TEXT,
    source                      TEXT,
    query_used                  TEXT,
    lang                        TEXT,
    likes                       INTEGER DEFAULT 0,
    retweets                    INTEGER DEFAULT 0,
    scraped_at                  TEXT,

    category                    TEXT,
    urgency                     TEXT,
    sentiment                   TEXT,
    root_cause                  TEXT,

    predicted_resolution_days   REAL,
    escalation_risk             REAL,

    is_anomaly                  INTEGER DEFAULT 0,
    anomaly_score               REAL,

    created_at                  TEXT DEFAULT (datetime('now'))
);
"""

CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_department ON grievances(department);",
    "CREATE INDEX IF NOT EXISTS idx_state      ON grievances(state);",
    "CREATE INDEX IF NOT EXISTS idx_status     ON grievances(status);",
    "CREATE INDEX IF NOT EXISTS idx_date_filed ON grievances(date_filed);",
    "CREATE INDEX IF NOT EXISTS idx_source     ON grievances(source);",
    "CREATE INDEX IF NOT EXISTS idx_urgency    ON grievances(urgency);",
]


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection, creating the DB file if needed."""
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # allows dict-style access
    return conn


def initialise_db() -> None:
    """Create tables and indexes if they don't exist."""
    with get_connection() as conn:
        conn.execute(CREATE_TABLE_SQL)
        for idx_sql in CREATE_INDEX_SQL:
            conn.execute(idx_sql)
        conn.commit()
    logger.info(f"Database initialised at {DB_PATH}")


def save_to_db(df: pd.DataFrame) -> int:
    """
    Insert records from a DataFrame into the grievances table.
    Skips duplicates (INSERT OR IGNORE on grievance_id).
    Returns the number of new rows inserted.
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to save_to_db — nothing to save.")
        return 0

    initialise_db()

    # Ensure all expected columns exist (fill missing with empty string)
    expected_cols = [
        "grievance_id", "text", "department", "state", "status",
        "date_filed", "date_resolved", "feedback_rating", "source",
        "query_used", "lang", "likes", "retweets", "scraped_at",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[expected_cols].copy()
    df = df.fillna("")

    inserted = 0
    with get_connection() as conn:
        for _, row in df.iterrows():
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO grievances
                        (grievance_id, text, department, state, status,
                         date_filed, date_resolved, feedback_rating, source,
                         query_used, lang, likes, retweets, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["grievance_id"], row["text"], row["department"],
                        row["state"], row["status"], row["date_filed"],
                        row["date_resolved"], row["feedback_rating"], row["source"],
                        row["query_used"], row["lang"],
                        int(row["likes"] or 0), int(row["retweets"] or 0),
                        row["scraped_at"],
                    ),
                )
                inserted += conn.execute("SELECT changes()").fetchone()[0]
            except Exception as e:
                logger.error(f"Failed to insert {row.get('grievance_id')}: {e}")

        conn.commit()

    logger.info(f"Inserted {inserted} new records into {DB_PATH}")
    return inserted


def query_db(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run any SELECT and return results as a DataFrame."""
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def get_stats() -> dict:
    """Quick summary stats of the current database."""
    with get_connection() as conn:
        total    = conn.execute("SELECT COUNT(*) FROM grievances").fetchone()[0]
        by_src   = pd.read_sql_query("SELECT source, COUNT(*) as n FROM grievances GROUP BY source", conn)
        by_dept  = pd.read_sql_query("SELECT department, COUNT(*) as n FROM grievances GROUP BY department ORDER BY n DESC LIMIT 5", conn)
        by_state = pd.read_sql_query("SELECT state, COUNT(*) as n FROM grievances GROUP BY state ORDER BY n DESC LIMIT 5", conn)

    return {
        "total_records": total,
        "by_source":     by_src.to_dict("records"),
        "top_departments": by_dept.to_dict("records"),
        "top_states":    by_state.to_dict("records"),
    }


if __name__ == "__main__":
    initialise_db()
    stats = get_stats()
    print(f"\nDatabase: {DB_PATH}")
    print(f"Total records: {stats['total_records']}")
    print(f"By source: {stats['by_source']}")
    print(f"Top departments: {stats['top_departments']}")