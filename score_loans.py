"""
score_loans.py
--------------
STEP 3 — Run after train_model.py
Scores all 30,000 borrowers with the trained model.
Writes probability of default + risk grade (A–F) to loan_risk_scores table.

Usage:
    python3 python/score_loans.py
"""

import sqlite3
import pickle
import numpy as np
import pandas as pd

DB_PATH    = "credit_risk.db"
MODEL_PATH = "outputs/model_artifacts.pkl"
MODEL_VER  = "v1.0-uci"

# Risk grade thresholds (probability of default)
GRADE_THRESHOLDS = [
    (0.05, "A"),   # Very low risk
    (0.10, "B"),   # Low risk
    (0.20, "C"),   # Moderate risk
    (0.30, "D"),   # Elevated risk
    (0.45, "E"),   # High risk
    (1.00, "F"),   # Very high risk
]

def assign_grade(prob: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if prob <= threshold:
            return grade
    return "F"

def main():
    # ── Load model ───────────────────────────────────────────
    with open(MODEL_PATH, "rb") as f:
        art = pickle.load(f)
    model = art["best_model"]
    print(f"✅  Model loaded: {art['best_name']}")

    # ── Load all borrowers ───────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM vw_loan_features", conn)
    print(f"✅  Loaded {len(df):,} borrowers to score")

    # ── Prepare features ─────────────────────────────────────
    num_cols = art["numeric_cols"]
    cat_cols = art["categorical_cols"]

    X = df.drop(columns=["borrower_id", "is_default"], errors="ignore")
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].fillna(0).astype(str)
    for col in num_cols:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    # ── Score ────────────────────────────────────────────────
    probas = model.predict_proba(X)[:, 1]
    grades = [assign_grade(p) for p in probas]

    # ── Write to DB ──────────────────────────────────────────
    conn.execute("DELETE FROM loan_risk_scores WHERE model_version = ?", (MODEL_VER,))
    records = [
        (int(bid), MODEL_VER, float(prob), grade)
        for bid, prob, grade in zip(df["borrower_id"], probas, grades)
    ]
    conn.executemany("""
        INSERT INTO loan_risk_scores (borrower_id, model_version, probability_default, risk_grade)
        VALUES (?,?,?,?)
    """, records)
    conn.commit()
    print(f"✅  Scored {len(records):,} borrowers → loan_risk_scores table")

    # ── Grade distribution ───────────────────────────────────
    grade_counts = pd.Series(grades).value_counts().sort_index()
    total        = len(grades)
    print(f"\n📊  Risk Grade Distribution:")
    print(f"    {'Grade':<8} {'Count':>6}  {'Share':>6}  Chart")
    print(f"    {'─'*45}")
    for grade, count in grade_counts.items():
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"    {grade:<8} {count:>6,}  {pct:>5.1f}%  {bar}")

    # ── Sample high-risk borrowers ───────────────────────────
    df["prob_default"] = probas
    df["risk_grade"]   = grades
    high_risk = df[df["risk_grade"].isin(["E", "F"])].nlargest(8, "prob_default")

    print(f"\n⚠️   Top High-Risk Borrowers (Grade E & F):")
    print(f"    {'ID':>6}  {'Age':>4}  {'CreditLimit':>12}  {'UtilRate':>9}  {'MthsLate':>9}  {'ProbDef':>8}  {'Grade'}")
    print(f"    {'─'*70}")
    for _, row in high_risk.iterrows():
        print(f"    {int(row.borrower_id):>6}  {int(row.age):>4}  "
              f"NT${row.credit_limit:>10,.0f}  "
              f"{row.utilization_rate*100:>8.1f}%  "
              f"{int(row.months_late):>9}  "
              f"{row.prob_default:>8.3f}  "
              f"{row.risk_grade}")

    conn.close()
    print(f"\n✅  Done! Run next: python3 python/dashboard.py")

if __name__ == "__main__":
    main()
