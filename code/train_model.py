"""
train_model.py
--------------
STEP 2 — Run after load_data.py
Trains Logistic Regression, Random Forest, Gradient Boosting
on UCI Credit Card Default data. Saves best model to outputs/.

Usage:
    python3 python/train_model.py
"""
install scikit-learn pandas numpy matplotlib

import sqlite3
import pickle
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, average_precision_score, brier_score_loss
)

# ── Column config ─────────────────────────────────────────────
DB_PATH    = "credit_risk.db"
MODEL_PATH = "outputs/model_artifacts.pkl"

NUMERIC_COLS = [
    "age", "credit_limit", "utilization_rate", "total_payments",
    "months_late", "avg_bill_amt",
    "pay_status_sep", "pay_status_aug", "pay_status_jul",
    "pay_status_jun", "pay_status_may", "pay_status_apr"
]
CATEGORICAL_COLS = ["sex", "education", "marriage"]
TARGET           = "is_default"
DROP_COLS        = ["borrower_id"]

# ── 1. Load data ──────────────────────────────────────────────
def load_features():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM vw_loan_features", conn)
    conn.close()
    print(f"✅  Loaded {len(df):,} rows from vw_loan_features")
    return df

# ── 2. Preprocess ─────────────────────────────────────────────
def preprocess(df):
    df = df.drop(columns=DROP_COLS, errors="ignore").copy()

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(str)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X, y

def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(),                                          NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ], remainder="drop")

# ── 3. Models ─────────────────────────────────────────────────
def build_models(prep):
    return {
        "Logistic Regression": Pipeline([
            ("prep",  prep),
            ("model", LogisticRegression(
                max_iter=1000, class_weight="balanced", C=1.0, random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("prep",  prep),
            ("model", RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight="balanced",
                min_samples_leaf=10, random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("prep",  prep),
            ("model", GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            ))
        ]),
    }

# ── 4. Evaluate ───────────────────────────────────────────────
def evaluate(name, model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    auc   = roc_auc_score(y_test, y_proba)
    ap    = average_precision_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    cm    = confusion_matrix(y_test, y_pred)

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"  Avg Precision : {ap:.4f}")
    print(f"  Brier Score   : {brier:.4f}  (lower is better)")
    print(f"\n  Confusion Matrix:")
    print(f"               Pred No-Default  Pred Default")
    print(f"  Actual No  :     {cm[0][0]:>6}           {cm[0][1]:>6}")
    print(f"  Actual Yes :     {cm[1][0]:>6}           {cm[1][1]:>6}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Default','Default'])}")

    return {
        "name": name, "auc": auc, "avg_precision": ap, "brier": brier,
        "model": model, "y_proba": y_proba, "y_test": y_test,
        "confusion_matrix": cm
    }

# ── 5. Feature names ──────────────────────────────────────────
def get_feature_names(model):
    prep = model.named_steps["prep"]
    ohe  = prep.named_transformers_["cat"]
    return NUMERIC_COLS + list(ohe.get_feature_names_out(CATEGORICAL_COLS))

# ── Main ──────────────────────────────────────────────────────
def main():
    os.makedirs("outputs", exist_ok=True)

    df   = load_features()
    X, y = preprocess(df)

    print(f"\n📊  Class Balance:")
    print(f"    No Default : {(y==0).sum():,}  ({100*(y==0).mean():.1f}%)")
    print(f"    Default    : {(y==1).sum():,}  ({100*(y==1).mean():.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\n    Train size : {len(X_train):,}")
    print(f"    Test size  : {len(X_test):,}")

    prep   = build_preprocessor()
    models = build_models(prep)

    results = []
    print(f"\n🚀  Training {len(models)} models …")
    for name, pipeline in models.items():
        print(f"\n    ⏳ Training {name} …")
        pipeline.fit(X_train, y_train)
        res = evaluate(name, pipeline, X_test, y_test)
        results.append(res)

    # Best model by AUC
    best = max(results, key=lambda r: r["auc"])
    print(f"\n{'='*55}")
    print(f"🏆  Best Model : {best['name']}")
    print(f"    ROC-AUC   : {best['auc']:.4f}")
    print(f"{'='*55}")

    # Save artifacts
    feat_names = get_feature_names(best["model"])
    artifacts  = {
        "best_model":       best["model"],
        "best_name":        best["name"],
        "all_results":      results,
        "feature_names":    feat_names,
        "numeric_cols":     NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "X_test":           X_test,
        "y_test":           y_test,
        "y_proba":          best["y_proba"],
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\n✅  Model saved → {MODEL_PATH}")
    print(f"    Run next   : python3 python/score_loans.py")

if __name__ == "__main__":
    main()
