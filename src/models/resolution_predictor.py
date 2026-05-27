"""
resolution_predictor.py
-----------------------
Week 4 — ML Models

Trains two models on the classified grievance dataset:

MODEL 1: Resolution Time Predictor (Regression)
  Target:  resolution_days (how many days to resolve a grievance)
  Output:  predicted_resolution_days column in DB
  Algo:    XGBoost Regressor
  Goal:    MAE < 10 days

MODEL 2: Escalation Risk Classifier (Binary Classification)
  Target:  is_escalated (1 if status=escalated, 0 otherwise)
  Output:  escalation_risk (0.0-1.0 probability) in DB
  Algo:    XGBoost Classifier
  Goal:    AUC > 0.75

Both models use SHAP for explainability.

Features used:
  - Department (one-hot encoded)
  - State (one-hot encoded)
  - Urgency (label encoded: low=0, medium=1, high=2, critical=3)
  - Root cause (one-hot encoded)
  - Sentiment (label encoded)
  - Text features: word_count, char_count, has_amount, has_date_ref, mentions_portal
  - Time features: month, weekday, quarter
  - Source type: is_live_data

Note: Sentence-BERT embeddings skipped to keep dependencies light.
      Add them in Week 4 notebook for even better accuracy.

Run:
  python resolution_predictor.py           # train both models
  python resolution_predictor.py --eval    # show evaluation metrics
  python resolution_predictor.py --predict # predict on new text interactively
"""

import os, sys, logging, argparse, sqlite3, joblib, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (mean_absolute_error, r2_score,
                             roc_auc_score, classification_report,
                             confusion_matrix)
import xgboost as xgb
import shap

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH        = os.getenv("DB_PATH", "data/grievances.db")
MODELS_DIR     = "models_saved"
PROCESSED_PATH = "data/processed"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING & FEATURE ENGINEERING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

URGENCY_MAP   = {"low": 0, "medium": 1, "high": 2, "critical": 3}
SENTIMENT_MAP = {"positive": 1, "neutral": 0, "negative": -1}

def load_features() -> pd.DataFrame:
    """Load cleaned + classified data and build feature matrix."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM grievances", conn)
    conn.close()
    logger.info(f"Loaded {len(df)} total records")

    # ── Parse dates ──────────────────────────────────────────────
    df["date_filed_dt"]    = pd.to_datetime(df["date_filed"],    errors="coerce", utc=False)
    df["date_resolved_dt"] = pd.to_datetime(df["date_resolved"], errors="coerce", utc=False)

    # Strip timezone if present
    for col in ["date_filed_dt", "date_resolved_dt"]:
        if df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_localize(None)

    # ── Resolution days (target for Model 1) ─────────────────────
    df["resolution_days"] = (
        (df["date_resolved_dt"] - df["date_filed_dt"])
        .dt.total_seconds() / 86400
    ).clip(lower=0)

    # ── Escalation flag (target for Model 2) ─────────────────────
    df["is_escalated"] = (df["status"] == "escalated").astype(int)

    # ── Text features ─────────────────────────────────────────────
    df["text_clean"]      = df["text"].fillna("").str.strip()
    df["word_count"]      = df["text_clean"].str.split().str.len().fillna(0)
    df["char_count"]      = df["text_clean"].str.len().fillna(0)
    df["has_amount"]      = df["text_clean"].str.contains(
                                r"rs\.?\s*\d+|₹\s*\d+", case=False, regex=True).astype(int)
    df["has_date_ref"]    = df["text_clean"].str.contains(
                                r"\d+\s*(?:day|month|year|week)", case=False, regex=True).astype(int)
    df["mentions_portal"] = df["text_clean"].str.contains(
                                r"portal|cpgrams|online|app|website", case=False, regex=True).astype(int)

    # ── Time features ─────────────────────────────────────────────
    df["month"]   = df["date_filed_dt"].dt.month.fillna(6).astype(int)
    df["weekday"] = df["date_filed_dt"].dt.dayofweek.fillna(2).astype(int)
    df["quarter"] = df["date_filed_dt"].dt.quarter.fillna(2).astype(int)

    # ── Encode LLM-derived features ───────────────────────────────
    df["urgency_enc"]   = df["urgency"].map(URGENCY_MAP).fillna(1).astype(int)
    df["sentiment_enc"] = df["sentiment"].map(SENTIMENT_MAP).fillna(-1).astype(int)

    # ── Source type ───────────────────────────────────────────────
    df["is_live_data"] = df["source"].isin({
        "guardian_live", "india_news_rss", "datagov_live", "newsapi_live"
    }).astype(int)

    # ── Fill missing categorical ──────────────────────────────────
    df["department"] = df["category"].fillna(df["department"]).fillna("Other")
    df["state"]      = df["state"].fillna("national")
    df["root_cause"] = df["root_cause"].fillna("other")

    logger.info(f"Features engineered. Shape: {df.shape}")
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Build X (features) ready for XGBoost."""

    CAT_FEATURES = ["department", "state", "root_cause"]
    NUM_FEATURES = ["urgency_enc", "sentiment_enc", "word_count", "char_count",
                    "has_amount", "has_date_ref", "mentions_portal",
                    "month", "weekday", "quarter", "is_live_data"]

    # One-hot encode categoricals
    dummies = pd.get_dummies(df[CAT_FEATURES], prefix=CAT_FEATURES, drop_first=False)
    X = pd.concat([df[NUM_FEATURES].reset_index(drop=True),
                   dummies.reset_index(drop=True)], axis=1)
    X = X.fillna(0)

    logger.info(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    return X, CAT_FEATURES, NUM_FEATURES


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL 1: RESOLUTION TIME PREDICTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_resolution_model(df: pd.DataFrame, X: pd.DataFrame):
    """Train XGBoost regressor to predict resolution_days."""
    logger.info("=" * 55)
    logger.info("MODEL 1: Resolution Time Predictor")
    logger.info("=" * 55)

    # Only use resolved records with real (non-synthetic) resolution dates
    # Synthetic resolution days are randomly generated — they hurt signal
    # datagov_cpgrams records have real resolution dates from official CPGRAMS
    real_sources = {"datagov_cpgrams", "datagov_live", "guardian_live",
                    "india_news_rss", "newsapi_live", "pib_live"}

    mask_resolved = df["resolution_days"].notna() & (df["resolution_days"] > 0)
    mask_real     = df["source"].isin(real_sources)

    # Use real sources if enough exist, else fall back to all resolved records
    if (mask_resolved & mask_real).sum() >= 50:
        mask = mask_resolved & mask_real
        logger.info(f"Using real-source records only for resolution model")
    else:
        mask = mask_resolved
        logger.info(f"Not enough real-source resolved records — using all sources")
        logger.info(f"  Note: synthetic resolution dates are random, expect weak R²")

    X_res    = X[mask].copy()
    y_res    = df.loc[mask, "resolution_days"]

    logger.info(f"Training samples (resolved records): {len(X_res)}")
    logger.info(f"  Sources: {df.loc[mask, 'source'].value_counts().to_dict()}")

    if len(X_res) < 20:
        logger.warning("Too few resolved records for reliable training (<20). "
                       "Collect more data with resolution dates.")
        logger.warning("Training on available data anyway...")

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators     = 200,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = 42,
        eval_metric      = "mae",
        verbosity        = 0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    logger.info(f"Test MAE:  {mae:.1f} days")
    logger.info(f"Test R²:   {r2:.3f}")
    logger.info(f"Baseline MAE (predict mean): {mean_absolute_error(y_test, [y_train.mean()]*len(y_test)):.1f} days")

    # Feature importance
    feat_imp = pd.Series(model.feature_importances_, index=X.columns)
    top_feats = feat_imp.nlargest(10)
    logger.info(f"Top 10 features:")
    for feat, imp in top_feats.items():
        logger.info(f"  {feat:<45} {imp:.4f}")

    return model, X_test, y_test, y_pred, feat_imp


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL 2: ESCALATION RISK CLASSIFIER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_escalation_model(df: pd.DataFrame, X: pd.DataFrame):
    """Train XGBoost classifier to predict escalation probability."""
    logger.info("=" * 55)
    logger.info("MODEL 2: Escalation Risk Classifier")
    logger.info("=" * 55)

    y_esc = df["is_escalated"]
    pos   = y_esc.sum()
    neg   = len(y_esc) - pos
    logger.info(f"Class distribution: escalated={pos} ({pos/len(y_esc)*100:.1f}%), "
                f"not_escalated={neg} ({neg/len(y_esc)*100:.1f}%)")

    if pos < 5:
        logger.warning("Very few escalated records — model may not generalise well.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_esc, test_size=0.2, random_state=42,
        stratify=y_esc if pos >= 10 else None
    )

    # scale_pos_weight handles class imbalance
    scale_pos = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators     = 200,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale_pos,
        random_state     = 42,
        eval_metric      = "auc",
        verbosity        = 0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    try:
        auc = roc_auc_score(y_test, y_prob)
        logger.info(f"Test AUC: {auc:.3f}")
    except Exception:
        logger.info("AUC: cannot compute (only one class in test set)")

    logger.info(f"Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred,
                                             target_names=["not_escalated", "escalated"],
                                             zero_division=0))

    feat_imp = pd.Series(model.feature_importances_, index=X.columns)
    top_feats = feat_imp.nlargest(10)
    logger.info("Top 10 features:")
    for feat, imp in top_feats.items():
        logger.info(f"  {feat:<45} {imp:.4f}")

    return model, X_test, y_test, y_prob, feat_imp


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHAP EXPLAINABILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_shap(model, X_sample: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Compute SHAP values and return a summary DataFrame.
    Saves SHAP plots to data/processed/.
    """
    logger.info(f"Computing SHAP values for {model_name}...")

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For classifiers shap_values is a list — take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Global feature importance from SHAP
        shap_df = pd.DataFrame({
            "feature":    X_sample.columns,
            "mean_|shap|": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_|shap|", ascending=False).reset_index(drop=True)

        logger.info(f"Top 10 SHAP features ({model_name}):")
        for _, row in shap_df.head(10).iterrows():
            bar = "█" * int(row["mean_|shap|"] * 50 / shap_df["mean_|shap|"].iloc[0])
            logger.info(f"  {row['feature']:<40} {bar} {row['mean_|shap|']:.4f}")

        # Save SHAP summary data
        path = f"{PROCESSED_PATH}/shap_{model_name}.csv"
        shap_df.to_csv(path, index=False)
        logger.info(f"SHAP summary saved → {path}")

        # Save individual SHAP values for notebook plotting
        shap_vals_df = pd.DataFrame(shap_values, columns=X_sample.columns)
        shap_vals_df.to_csv(f"{PROCESSED_PATH}/shap_values_{model_name}.csv", index=False)

        return shap_df

    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        return pd.DataFrame()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICT ON DB & SAVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def predict_and_save(df: pd.DataFrame, X: pd.DataFrame,
                     res_model, esc_model):
    """Run both models on all records and write predictions to DB."""
    logger.info("Running predictions on full dataset...")

    # Predictions
    res_preds = res_model.predict(X).clip(min=0)
    esc_probs = esc_model.predict_proba(X)[:, 1]

    # Write to DB
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    updated = 0

    for i, gid in enumerate(df["grievance_id"]):
        cursor.execute("""
            UPDATE grievances
            SET predicted_resolution_days = ?,
                escalation_risk           = ?
            WHERE grievance_id = ?
        """, (
            round(float(res_preds[i]), 1),
            round(float(esc_probs[i]), 3),
            gid,
        ))
        updated += cursor.rowcount

    conn.commit()
    conn.close()
    logger.info(f"Predictions written to DB: {updated} records")
    logger.info(f"  Avg predicted resolution: {res_preds.mean():.1f} days")
    logger.info(f"  High escalation risk (>0.5): {(esc_probs > 0.5).sum()} records")

    return res_preds, esc_probs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAVE & LOAD MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_models(res_model, esc_model, feature_cols: list):
    """Save trained models + feature column list."""
    joblib.dump(res_model,    f"{MODELS_DIR}/resolution_xgb.pkl")
    joblib.dump(esc_model,    f"{MODELS_DIR}/escalation_xgb.pkl")
    joblib.dump(feature_cols, f"{MODELS_DIR}/feature_columns.pkl")
    logger.info(f"Models saved to {MODELS_DIR}/")
    logger.info(f"  resolution_xgb.pkl")
    logger.info(f"  escalation_xgb.pkl")
    logger.info(f"  feature_columns.pkl")


def load_models():
    """Load saved models for inference."""
    res_model    = joblib.load(f"{MODELS_DIR}/resolution_xgb.pkl")
    esc_model    = joblib.load(f"{MODELS_DIR}/escalation_xgb.pkl")
    feature_cols = joblib.load(f"{MODELS_DIR}/feature_columns.pkl")
    return res_model, esc_model, feature_cols


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTERACTIVE PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def predict_single(text: str, department: str = "Other",
                   urgency: str = "medium", state: str = "national",
                   root_cause: str = "service_delay"):
    """Predict resolution time + escalation risk for a single grievance."""
    try:
        res_model, esc_model, feature_cols = load_models()
    except FileNotFoundError:
        print("Models not found. Run training first: python resolution_predictor.py")
        return

    # Build feature row
    row = {
        "urgency_enc":    URGENCY_MAP.get(urgency, 1),
        "sentiment_enc":  -1,   # assume negative (grievance)
        "word_count":     len(text.split()),
        "char_count":     len(text),
        "has_amount":     int(bool(__import__("re").search(r"rs\.?\s*\d+|₹", text, __import__("re").I))),
        "has_date_ref":   int(bool(__import__("re").search(r"\d+\s*(?:day|month|year|week)", text, __import__("re").I))),
        "mentions_portal":int(bool(__import__("re").search(r"portal|cpgrams|online|app", text, __import__("re").I))),
        "month":          datetime.now().month,
        "weekday":        datetime.now().weekday(),
        "quarter":        (datetime.now().month - 1) // 3 + 1,
        "is_live_data":   0,
    }

    # One-hot for department, state, root_cause
    for col in feature_cols:
        if col not in row:
            row[col] = 0

    # Set the right one-hot flags
    dept_col = f"department_{department}"
    state_col = f"state_{state}"
    rc_col    = f"root_cause_{root_cause}"
    if dept_col  in row: row[dept_col]  = 1
    if state_col in row: row[state_col] = 1
    if rc_col    in row: row[rc_col]    = 1

    X = pd.DataFrame([row])[feature_cols].fillna(0)

    res_days = float(res_model.predict(X)[0])
    esc_risk = float(esc_model.predict_proba(X)[0][1])

    print(f"\n{'='*50}")
    print(f"GRIEVANCE PREDICTION")
    print(f"{'='*50}")
    print(f"Text:                  {text[:80]}...")
    print(f"Department:            {department}")
    print(f"Predicted resolution:  {res_days:.1f} days")
    print(f"Escalation risk:       {esc_risk:.1%}")
    print(f"SLA status:            {'⚠ BREACH LIKELY' if res_days > 21 else '✓ Within SLA'}")
    print(f"{'='*50}")

    return {"resolution_days": res_days, "escalation_risk": esc_risk}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MASTER PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pipeline(eval_only: bool = False):
    """Full training pipeline."""
    logger.info("=" * 55)
    logger.info("WEEK 4 — ML TRAINING PIPELINE")
    logger.info("=" * 55)

    # Load & engineer features
    df = load_features()
    X, cat_cols, num_cols = build_feature_matrix(df)

    if eval_only:
        logger.info("Eval-only mode — loading saved models")
        res_model, esc_model, _ = load_models()
    else:
        # Train Model 1
        res_model, X_res_test, y_res_test, y_res_pred, res_imp = train_resolution_model(df, X)

        # SHAP for Model 1 (on test set sample)
        shap_sample = X_res_test.iloc[:min(100, len(X_res_test))]
        res_shap_df = compute_shap(res_model, shap_sample, "resolution")

        # Train Model 2
        esc_model, X_esc_test, y_esc_test, y_esc_prob, esc_imp = train_escalation_model(df, X)

        # SHAP for Model 2
        shap_sample2 = X.iloc[:min(100, len(X))]
        esc_shap_df  = compute_shap(esc_model, shap_sample2, "escalation")

        # Save models
        save_models(res_model, esc_model, list(X.columns))

        # Predict on full dataset and write to DB
        predict_and_save(df, X, res_model, esc_model)

        logger.info("=" * 55)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 55)
        logger.info("Next steps:")
        logger.info("  → Open notebooks/03_ml_prediction.ipynb")
        logger.info("  → Run: python anomaly_detector.py  (Week 5)")

    return res_model, esc_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train resolution + escalation models")
    parser.add_argument("--eval",    action="store_true", help="Evaluate saved models only")
    parser.add_argument("--predict", action="store_true", help="Interactive prediction mode")
    parser.add_argument("--shap",    action="store_true", help="Recompute SHAP only")
    args = parser.parse_args()

    if args.predict:
        text = input("Enter grievance text: ")
        dept = input("Department (e.g. Ministry of Railways): ") or "Other"
        urg  = input("Urgency (low/medium/high/critical): ") or "medium"
        predict_single(text, department=dept, urgency=urg)
    else:
        run_pipeline(eval_only=args.eval)