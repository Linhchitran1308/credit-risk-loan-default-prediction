"""
dashboard.py
------------
STEP 4 — Run after score_loans.py
Generates a full HTML dashboard saved to outputs/credit_risk_dashboard.html

Usage:
    python3 python/dashboard.py
    open outputs/credit_risk_dashboard.html
"""

import sqlite3
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

DB_PATH     = "credit_risk.db"
MODEL_PATH  = "outputs/model_artifacts.pkl"
OUTPUT_HTML = "outputs/credit_risk_dashboard.html"

C = {
    "primary":  "#1d4ed8",
    "danger":   "#dc2626",
    "success":  "#16a34a",
    "warning":  "#d97706",
    "purple":   "#7c3aed",
    "neutral":  "#6b7280",
    "bg":       "#f8fafc",
    "grid":     "#e2e8f0",
}

# ── Utility ───────────────────────────────────────────────────
def fig_to_b64(fig):
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def style(ax, title=""):
    ax.set_facecolor(C["bg"])
    ax.grid(True, color=C["grid"], linewidth=0.6, linestyle="--", alpha=0.8)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(C["grid"])
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color="#1e293b")

# ── Plot functions ────────────────────────────────────────────

def plot_default_by_education(conn):
    df = pd.read_sql("""
        SELECT education_label,
               COUNT(*) AS total,
               ROUND(100.0*SUM(is_default)/COUNT(*),2) AS default_rate
        FROM credit_card_default
        GROUP BY education_label
        ORDER BY default_rate DESC
    """, conn)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=C["bg"])
    colors  = [C["danger"] if r > 25 else C["warning"] if r > 20 else C["success"]
               for r in df.default_rate]
    bars = ax.bar(df.education_label, df.default_rate, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, df.default_rate):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Default Rate (%)")
    ax.set_ylim(0, df.default_rate.max() + 5)
    style(ax, "Default Rate by Education Level")
    plt.tight_layout()
    return fig

def plot_age_default(conn):
    df = pd.read_sql("SELECT age, is_default FROM credit_card_default", conn)
    df["age_bin"] = pd.cut(df.age, bins=[20,30,40,50,60,80],
                           labels=["20s","30s","40s","50s","60+"])
    agg = df.groupby("age_bin", observed=True).agg(
        total=("is_default","count"),
        defaults=("is_default","sum")
    ).reset_index()
    agg["rate"] = 100 * agg.defaults / agg.total

    fig, ax1 = plt.subplots(figsize=(7, 4), facecolor=C["bg"])
    ax2 = ax1.twinx()
    x   = range(len(agg))
    ax1.bar(x, agg.total, color=C["primary"], alpha=0.35, label="Total Borrowers")
    ax2.plot(x, agg.rate, "o-", color=C["danger"], lw=2.5, ms=7, label="Default Rate %")
    ax2.fill_between(x, agg.rate, alpha=0.1, color=C["danger"])
    ax1.set_xticks(x); ax1.set_xticklabels(agg.age_bin)
    ax1.set_xlabel("Age Group")
    ax1.set_ylabel("Borrower Count", color=C["primary"])
    ax2.set_ylabel("Default Rate (%)", color=C["danger"])
    ax1.tick_params(axis="y", colors=C["primary"])
    ax2.tick_params(axis="y", colors=C["danger"])
    style(ax1, "Borrower Count & Default Rate by Age")
    plt.tight_layout()
    return fig

def plot_credit_limit_dist(conn):
    df = pd.read_sql("SELECT credit_limit, is_default FROM credit_card_default", conn)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=C["bg"])
    for val, label, color in [(0,"No Default",C["success"]), (1,"Default",C["danger"])]:
        sub = df[df.is_default==val].credit_limit / 1000
        ax.hist(sub, bins=40, alpha=0.55, label=label, color=color, edgecolor="white")
    ax.set_xlabel("Credit Limit (NT$ thousands)")
    ax.set_ylabel("Count")
    ax.legend()
    style(ax, "Credit Limit Distribution by Default Status")
    plt.tight_layout()
    return fig

def plot_payment_status_heatmap(conn):
    df = pd.read_sql("""
        SELECT pay_status_sep, pay_status_aug, pay_status_jul,
               pay_status_jun, pay_status_may, pay_status_apr,
               is_default
        FROM credit_card_default
    """, conn)
    months  = ["Sep","Aug","Jul","Jun","May","Apr"]
    cols    = ["pay_status_sep","pay_status_aug","pay_status_jul",
               "pay_status_jun","pay_status_may","pay_status_apr"]
    buckets = [-2,-1,0,1,2,3,4,5,6,7,8,9]

    data = []
    for col, month in zip(cols, months):
        for bucket in buckets:
            sub  = df[df[col] == bucket]
            rate = sub["is_default"].mean() * 100 if len(sub) > 0 else 0
            data.append({"month": month, "status": bucket, "default_rate": rate, "n": len(sub)})
    pivot = pd.DataFrame(data).pivot(index="status", columns="month", values="default_rate")

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C["bg"])
    im = ax.imshow(pivot.fillna(0), cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=60)
    ax.set_xticks(range(len(months)));  ax.set_xticklabels(months)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Month")
    ax.set_ylabel("Payment Status")
    plt.colorbar(im, ax=ax, label="Default Rate (%)")
    style(ax, "Default Rate by Payment Status × Month")
    plt.tight_layout()
    return fig

def plot_utilization_default(conn):
    df = pd.read_sql("SELECT utilization_rate, is_default FROM credit_card_default", conn)
    df["util_bin"] = pd.cut(df.utilization_rate, bins=[0,0.2,0.4,0.6,0.8,1.0],
                            labels=["0-20%","20-40%","40-60%","60-80%","80-100%"])
    agg = df.groupby("util_bin", observed=True)["is_default"].agg(["mean","count"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=C["bg"])
    colors  = [C["success"] if r < 0.2 else C["warning"] if r < 0.28 else C["danger"]
               for r in agg["mean"]]
    bars = ax.bar(agg.util_bin, agg["mean"]*100, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, agg["mean"]*100):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Credit Utilization Rate")
    ax.set_ylabel("Default Rate (%)")
    style(ax, "Default Rate by Credit Utilization")
    plt.tight_layout()
    return fig

def plot_roc(y_test, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=C["bg"])
    ax.plot(fpr, tpr, color=C["primary"], lw=2.5, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1],"--", color=C["neutral"], lw=1)
    ax.fill_between(fpr, tpr, alpha=0.07, color=C["primary"])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=10)
    style(ax, f"ROC Curve — {model_name}")
    plt.tight_layout()
    return fig

def plot_pr(y_test, y_proba, model_name):
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=C["bg"])
    ax.plot(rec, prec, color=C["warning"], lw=2.5, label=f"Avg Precision = {ap:.3f}")
    ax.fill_between(rec, prec, alpha=0.08, color=C["warning"])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=10)
    style(ax, f"Precision-Recall Curve — {model_name}")
    plt.tight_layout()
    return fig

def plot_score_dist(y_test, y_proba):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=C["bg"])
    ax.hist(y_proba[y_test==0], bins=50, alpha=0.6, color=C["success"],
            label="No Default", density=True, edgecolor="white")
    ax.hist(y_proba[y_test==1], bins=50, alpha=0.6, color=C["danger"],
            label="Default", density=True, edgecolor="white")
    ax.axvline(0.5, color="#1e293b", linestyle="--", lw=1.5, label="Threshold = 0.5")
    ax.set_xlabel("Predicted Probability of Default")
    ax.set_ylabel("Density")
    ax.legend()
    style(ax, "Model Score Distribution (Test Set)")
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names):
    step = model.named_steps["model"]
    if hasattr(step, "feature_importances_"):
        importances = step.feature_importances_
    else:
        importances = np.abs(step.coef_[0])

    top_n = 15
    idx   = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=C["bg"])
    colors  = [C["primary"] if v > np.percentile(vals, 70) else C["neutral"] for v in vals]
    ax.barh(names, vals, color=colors, height=0.65, edgecolor="white")
    ax.set_xlabel("Feature Importance Score")
    style(ax, f"Top {top_n} Most Important Features")
    plt.tight_layout()
    return fig

def plot_grade_distribution(conn):
    df = pd.read_sql("""
        SELECT risk_grade, COUNT(*) as cnt
        FROM loan_risk_scores
        GROUP BY risk_grade ORDER BY risk_grade
    """, conn)
    grade_colors = {"A":C["success"],"B":"#65a30d","C":C["warning"],
                    "D":"#ea580c","E":C["danger"],"F":"#7f1d1d"}
    colors = [grade_colors.get(g, C["neutral"]) for g in df.risk_grade]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=C["bg"])
    bars = ax.bar(df.risk_grade, df.cnt, color=colors, width=0.5, edgecolor="white")
    total = df.cnt.sum()
    for bar, val in zip(bars, df.cnt):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                f"{val:,}\n({100*val/total:.1f}%)", ha="center", fontsize=8.5)
    ax.set_xlabel("Risk Grade"); ax.set_ylabel("Number of Borrowers")
    style(ax, "Borrower Risk Grade Distribution")
    plt.tight_layout()
    return fig

# ── KPIs ──────────────────────────────────────────────────────
def get_kpis(conn, y_test, y_proba):
    r = pd.read_sql("""
        SELECT COUNT(*) as n,
               SUM(is_default) as def_n,
               ROUND(100.0*AVG(is_default),2) as dr,
               ROUND(AVG(credit_limit),0) as avg_limit,
               ROUND(AVG(utilization_rate)*100,1) as avg_util
        FROM credit_card_default
    """, conn).iloc[0]
    auc = roc_auc_score(y_test, y_proba)
    return {
        "Total Borrowers":    f"{int(r.n):,}",
        "Total Defaults":     f"{int(r.def_n):,}",
        "Default Rate":       f"{r.dr:.1f}%",
        "Avg Credit Limit":   f"NT${int(r.avg_limit):,}",
        "Avg Utilization":    f"{r.avg_util:.1f}%",
        "Model AUC":          f"{auc:.3f}",
    }

# ── HTML Template ─────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Credit Risk Dashboard — UCI Dataset</title>
<style>
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f1f5f9;color:#0f172a}}
  header{{background:linear-gradient(120deg,#1d4ed8 0%,#1e3a8a 100%);
          color:white;padding:22px 36px;display:flex;align-items:center;gap:16px;
          box-shadow:0 2px 12px rgba(0,0,0,.15)}}
  header .badge{{background:rgba(255,255,255,.15);border-radius:6px;
                 padding:4px 10px;font-size:.75rem;letter-spacing:.05em}}
  header h1{{font-size:1.55rem;font-weight:700}}
  header p{{font-size:.82rem;opacity:.8;margin-top:3px}}
  .container{{max-width:1300px;margin:0 auto;padding:28px 24px}}
  .section-title{{font-size:1rem;font-weight:700;color:#334155;
                  border-left:4px solid #1d4ed8;padding-left:10px;
                  margin:32px 0 14px;text-transform:uppercase;letter-spacing:.05em}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px}}
  .kpi-card{{background:white;border-radius:12px;padding:18px 20px;
             box-shadow:0 1px 6px rgba(0,0,0,.07);
             border-top:4px solid #1d4ed8;transition:transform .15s}}
  .kpi-card:hover{{transform:translateY(-2px)}}
  .kpi-card .label{{font-size:.72rem;color:#64748b;text-transform:uppercase;
                    letter-spacing:.06em;margin-bottom:6px}}
  .kpi-card .value{{font-size:1.75rem;font-weight:800;color:#0f172a}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
  .grid-1{{display:grid;gap:18px}}
  .card{{background:white;border-radius:12px;padding:18px;
         box-shadow:0 1px 6px rgba(0,0,0,.07)}}
  .card img{{width:100%;height:auto;border-radius:6px}}
  .info-box{{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
             padding:14px 18px;margin-bottom:24px;font-size:.85rem;color:#1e40af;
             line-height:1.6}}
  footer{{text-align:center;font-size:.75rem;color:#94a3b8;
          padding:24px;margin-top:36px;border-top:1px solid #e2e8f0}}
  @media(max-width:768px){{.grid-2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
  <div>
    <h1>📊 Credit Risk Dashboard</h1>
    <p>UCI Credit Card Default Dataset &nbsp;·&nbsp; 30,000 Taiwanese Credit Card Holders &nbsp;·&nbsp; 2005</p>
  </div>
  <div class="badge">ML-Powered</div>
</header>
<div class="container">

  <div class="info-box">
    ℹ️ This dashboard analyzes the <strong>UCI Credit Card Default Dataset</strong> — 30,000 credit card clients in Taiwan.
    The target variable predicts whether a client will <strong>default on next month's payment</strong>.
    Features include credit limit, demographics, 6-month payment history, bill amounts, and payment amounts.
  </div>

  <div class="section-title">Portfolio KPIs</div>
  <div class="kpi-grid">{kpi_cards}</div>

  <div class="section-title">Portfolio Analytics</div>
  <div class="grid-2">
    <div class="card"><img src="data:image/png;base64,{img_edu}"   alt="Default by Education"></div>
    <div class="card"><img src="data:image/png;base64,{img_util}"  alt="Utilization vs Default"></div>
  </div>
  <div class="grid-2" style="margin-top:18px">
    <div class="card"><img src="data:image/png;base64,{img_age}"   alt="Age vs Default"></div>
    <div class="card"><img src="data:image/png;base64,{img_limit}" alt="Credit Limit Distribution"></div>
  </div>
  <div class="grid-1" style="margin-top:18px">
    <div class="card"><img src="data:image/png;base64,{img_heat}"  alt="Payment Status Heatmap"></div>
  </div>

  <div class="section-title">Model Performance — {model_name}</div>
  <div class="grid-2">
    <div class="card"><img src="data:image/png;base64,{img_roc}" alt="ROC Curve"></div>
    <div class="card"><img src="data:image/png;base64,{img_pr}"  alt="Precision Recall"></div>
  </div>
  <div class="grid-1" style="margin-top:18px">
    <div class="card"><img src="data:image/png;base64,{img_scores}" alt="Score Distribution"></div>
  </div>

  <div class="section-title">Risk Scoring</div>
  <div class="grid-2">
    <div class="card"><img src="data:image/png;base64,{img_grades}" alt="Grade Distribution"></div>
    <div class="card"><img src="data:image/png;base64,{img_fi}"     alt="Feature Importance"></div>
  </div>

</div>
<footer>Credit Risk Analytics · UCI Credit Card Default Dataset · Built with Python + SQLite + scikit-learn</footer>
</body>
</html>
"""

def make_kpi_cards(kpis):
    return "".join(
        f'<div class="kpi-card"><div class="label">{k}</div><div class="value">{v}</div></div>'
        for k, v in kpis.items()
    )

# ── Main ──────────────────────────────────────────────────────
def main():
    os.makedirs("outputs", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    with open(MODEL_PATH, "rb") as f:
        art = pickle.load(f)

    y_test     = art["y_test"].values if hasattr(art["y_test"], "values") else art["y_test"]
    y_proba    = art["y_proba"]
    model      = art["best_model"]
    model_name = art["best_name"]
    feat_names = art["feature_names"]

    print("🎨  Generating charts …")
    imgs = {
        "edu":    fig_to_b64(plot_default_by_education(conn)),
        "util":   fig_to_b64(plot_utilization_default(conn)),
        "age":    fig_to_b64(plot_age_default(conn)),
        "limit":  fig_to_b64(plot_credit_limit_dist(conn)),
        "heat":   fig_to_b64(plot_payment_status_heatmap(conn)),
        "roc":    fig_to_b64(plot_roc(y_test, y_proba, model_name)),
        "pr":     fig_to_b64(plot_pr(y_test, y_proba, model_name)),
        "scores": fig_to_b64(plot_score_dist(y_test, y_proba)),
        "grades": fig_to_b64(plot_grade_distribution(conn)),
        "fi":     fig_to_b64(plot_feature_importance(model, feat_names)),
    }
    print(f"    ✅  {len(imgs)} charts generated")

    kpis = get_kpis(conn, y_test, y_proba)
    conn.close()
    plt.close("all")

    html = HTML.format(
        kpi_cards  = make_kpi_cards(kpis),
        model_name = model_name,
        **{f"img_{k}": v for k, v in imgs.items()}
    )

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)

    print(f"\n✅  Dashboard saved → {OUTPUT_HTML}")
    print(f"    Open with   : open {OUTPUT_HTML}")

if __name__ == "__main__":
    main()
