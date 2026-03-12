# 📊 Credit Risk — Loan Default Prediction

Predicting credit card default using machine learning on the UCI Credit Card Default Dataset.

## 📁 Dataset
- **Source:** [UCI Credit Card Default Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- **Size:** 30,000 Taiwanese credit card holders
- **Target:** Predict `default.payment.next.month`

## 📊 Results
| Model | ROC-AUC | Accuracy |
|-------|---------|----------|
| Logistic Regression | 0.7439 | 76% |
| Random Forest | 0.7769 | 78% |
| **Gradient Boosting** 🏆 | **0.7803** | **82%** |

## 🗂️ Project Structure
| File | Purpose |
|------|---------|
| `python/load_data.py` | Load & engineer features from CSV into SQLite |
| `python/train_model.py` | Train & evaluate 3 ML models |
| `python/score_loans.py` | Score all borrowers with risk grades A–F |
| `python/dashboard.py` | Generate HTML visual dashboard |
| `sql/01_schema.sql` | Database schema & analytical views |

## 🔑 Key Findings
- **22.1% default rate** across 30,000 borrowers
- **months_late** is the strongest default predictor
- High School graduates default at **25.2%** vs 19.2% for graduates
- **13.8%** of borrowers scored Grade F (Very High Risk)

## 🛠️ Tech Stack
- **Python** — pandas, scikit-learn, matplotlib
- **SQL** — SQLite with normalized schema & views
- **ML Models** — Logistic Regression, Random Forest, Gradient Boosting

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn matplotlib
python python/load_data.py
python python/train_model.py
python python/score_loans.py
python python/dashboard.py
open outputs/credit_risk_dashboard.html
```
