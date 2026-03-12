"""
load_data.py  (final fixed version)
-------------------------------------
STEP 1 — Run this first.
Loads UCI Credit Card Default CSV into SQLite database.

The UCI CSV from Kaggle has a corrupted header row where all column
names are wrapped in double-quotes inside a single outer quote.
This script parses it correctly.

Usage (run from your project root folder):
    python3 python/load_data.py
"""

import sqlite3
import pandas as pd
import io
import os

# ── Config ────────────────────────────────────────────────────
DB_PATH     = "credit_risk.db"
CSV_PATH    = "data/UCI_Credit_Card.csv"
SCHEMA_PATH = "sql/01_schema.sql"

# ── Load CSV (handles the UCI Kaggle header quirk) ────────────
def load_uci_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Fix the mangled header line:
    # Original: "ID,""LIMIT_BAL"",""SEX"",...""default.payment.next.month"""
    # Fixed   : ['ID', 'LIMIT_BAL', 'SEX', ..., 'default.payment.next.month']
    raw_header = lines[0].strip().strip('"')
    cols = [c.replace('""', '').replace('"', '').strip()
            for c in raw_header.split(',')]

    # Read the data rows (everything after the header)
    data_str = "".join(lines[1:])
    df = pd.read_csv(io.StringIO(data_str), header=None, names=cols)

    return df

def main():
    # ── 1. Check file ─────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"❌  File not found: {CSV_PATH}")
        print(f"    Put UCI_Credit_Card.csv inside the  data/  folder.")
        return

    # ── 2. Load CSV ───────────────────────────────────────────
    df = load_uci_csv(CSV_PATH)
    print(f"✅  CSV loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"    Columns: {df.columns.tolist()}\n")

    # ── 3. Rename columns ─────────────────────────────────────
    df = df.rename(columns={
        "ID":                         "borrower_id",
        "LIMIT_BAL":                  "credit_limit",
        "AGE":                        "age",
        "SEX":                        "sex",
        "EDUCATION":                  "education",
        "MARRIAGE":                   "marriage",
        "PAY_0":                      "pay_status_sep",
        "PAY_2":                      "pay_status_aug",
        "PAY_3":                      "pay_status_jul",
        "PAY_4":                      "pay_status_jun",
        "PAY_5":                      "pay_status_may",
        "PAY_6":                      "pay_status_apr",
        "BILL_AMT1":                  "bill_amt_sep",
        "BILL_AMT2":                  "bill_amt_aug",
        "BILL_AMT3":                  "bill_amt_jul",
        "BILL_AMT4":                  "bill_amt_jun",
        "BILL_AMT5":                  "bill_amt_may",
        "BILL_AMT6":                  "bill_amt_apr",
        "PAY_AMT1":                   "pay_amt_sep",
        "PAY_AMT2":                   "pay_amt_aug",
        "PAY_AMT3":                   "pay_amt_jul",
        "PAY_AMT4":                   "pay_amt_jun",
        "PAY_AMT5":                   "pay_amt_may",
        "PAY_AMT6":                   "pay_amt_apr",
        "default.payment.next.month": "is_default",
    })

    # Verify key columns exist
    required = ["borrower_id", "credit_limit", "age", "is_default"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌  Missing columns: {missing}")
        print(f"    Available: {df.columns.tolist()}")
        return

    # ── 4. Feature Engineering ────────────────────────────────
    bill_cols = ["bill_amt_sep","bill_amt_aug","bill_amt_jul",
                 "bill_amt_jun","bill_amt_may","bill_amt_apr"]
    pay_cols  = ["pay_amt_sep","pay_amt_aug","pay_amt_jul",
                 "pay_amt_jun","pay_amt_may","pay_amt_apr"]
    pay_status_cols = ["pay_status_sep","pay_status_aug","pay_status_jul",
                       "pay_status_jun","pay_status_may","pay_status_apr"]

    df["avg_bill_amt"]     = df[bill_cols].mean(axis=1).round(2)
    df["utilization_rate"] = (
        df["avg_bill_amt"] / df["credit_limit"].replace(0, 1)
    ).clip(0, 1).round(4)
    df["total_payments"]   = df[pay_cols].sum(axis=1).round(2)
    df["months_late"]      = (df[pay_status_cols] > 0).sum(axis=1)

    df["sex_label"]       = df["sex"].map({1:"Male", 2:"Female"})
    df["education_label"] = df["education"].map({
        1:"Graduate", 2:"University", 3:"High School",
        4:"Others",   5:"Unknown",   6:"Unknown", 0:"Unknown"
    })
    df["marriage_label"]  = df["marriage"].map({
        0:"Other", 1:"Married", 2:"Single", 3:"Other"
    })

    # ── 5. Create database ────────────────────────────────────
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    os.makedirs("outputs", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    print("✅  Schema created")

    # ── 6. Insert data ────────────────────────────────────────
    df.to_sql("credit_card_default", conn, if_exists="replace", index=False)
    conn.commit()
    print(f"✅  {len(df):,} records inserted into credit_card_default")

    # ── 7. Summary ────────────────────────────────────────────
    total    = len(df)
    defaults = int(df["is_default"].sum())
    rate     = 100 * defaults / total

    print(f"\n📊  Dataset Summary:")
    print(f"    Total records  : {total:,}")
    print(f"    Defaults       : {defaults:,}  ({rate:.1f}% default rate)")
    print(f"    Age range      : {int(df.age.min())} – {int(df.age.max())}")
    print(f"    Credit limit   : NT${df.credit_limit.min():,.0f} – NT${df.credit_limit.max():,.0f}")
    print(f"    Avg util. rate : {df.utilization_rate.mean()*100:.1f}%")

    print(f"\n    Default Rate by Education:")
    edu = df.groupby("education_label")["is_default"].agg(["mean","count"])
    for name, row in edu.iterrows():
        print(f"      {str(name):<15}  {row['mean']*100:.1f}%  (n={int(row['count']):,})")

    print(f"\n    DB saved → {DB_PATH}")
    print(f"\n✅  Done! Run next:  python3 python/train_model.py")
    conn.close()

if __name__ == "__main__":
    main()
