-- ============================================================
-- CREDIT RISK DATABASE SCHEMA
-- UCI Credit Card Default Dataset
-- ============================================================

DROP TABLE IF EXISTS loan_risk_scores;
DROP TABLE IF EXISTS credit_card_default;

-- ============================================================
-- TABLE: credit_card_default
-- Main table storing all UCI dataset fields + engineered features
-- ============================================================
CREATE TABLE credit_card_default (
    borrower_id         INTEGER PRIMARY KEY,
    -- Demographics
    credit_limit        REAL,
    age                 INTEGER,
    sex                 INTEGER,        -- 1=Male, 2=Female
    education           INTEGER,        -- 1=Graduate, 2=University, 3=High School, 4=Others
    marriage            INTEGER,        -- 1=Married, 2=Single, 3=Others
    -- Payment status (negative=paid early, 0=on time, 1-9=months delayed)
    pay_status_sep      INTEGER,
    pay_status_aug      INTEGER,
    pay_status_jul      INTEGER,
    pay_status_jun      INTEGER,
    pay_status_may      INTEGER,
    pay_status_apr      INTEGER,
    -- Bill amounts (last 6 months)
    bill_amt_sep        REAL,
    bill_amt_aug        REAL,
    bill_amt_jul        REAL,
    bill_amt_jun        REAL,
    bill_amt_may        REAL,
    bill_amt_apr        REAL,
    -- Payment amounts (last 6 months)
    pay_amt_sep         REAL,
    pay_amt_aug         REAL,
    pay_amt_jul         REAL,
    pay_amt_jun         REAL,
    pay_amt_may         REAL,
    pay_amt_apr         REAL,
    -- Engineered features
    avg_bill_amt        REAL,
    utilization_rate    REAL,
    total_payments      REAL,
    months_late         INTEGER,
    -- Readable labels
    sex_label           TEXT,
    education_label     TEXT,
    marriage_label      TEXT,
    -- Target variable
    is_default          INTEGER CHECK (is_default IN (0, 1))
);

-- ============================================================
-- TABLE: loan_risk_scores
-- Model-generated risk scores
-- ============================================================
CREATE TABLE loan_risk_scores (
    score_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    borrower_id         INTEGER NOT NULL REFERENCES credit_card_default(borrower_id),
    model_version       TEXT NOT NULL,
    probability_default REAL CHECK (probability_default BETWEEN 0 AND 1),
    risk_grade          TEXT CHECK (risk_grade IN ('A','B','C','D','E','F')),
    scored_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX idx_default     ON credit_card_default(is_default);
CREATE INDEX idx_age         ON credit_card_default(age);
CREATE INDEX idx_education   ON credit_card_default(education);
CREATE INDEX idx_scores_bid  ON loan_risk_scores(borrower_id);

-- ============================================================
-- VIEWS
-- ============================================================

-- ML feature view
CREATE VIEW vw_loan_features AS
SELECT
    borrower_id,
    age,
    credit_limit,
    sex,
    education,
    marriage,
    utilization_rate,
    total_payments,
    months_late,
    avg_bill_amt,
    pay_status_sep,
    pay_status_aug,
    pay_status_jul,
    pay_status_jun,
    pay_status_may,
    pay_status_apr,
    is_default
FROM credit_card_default;

-- Portfolio summary view
CREATE VIEW vw_portfolio_summary AS
SELECT
    education_label,
    marriage_label,
    sex_label,
    COUNT(*)                                         AS total_records,
    ROUND(AVG(credit_limit), 0)                      AS avg_credit_limit,
    ROUND(AVG(utilization_rate) * 100, 1)            AS avg_utilization_pct,
    SUM(is_default)                                  AS total_defaults,
    ROUND(100.0 * SUM(is_default) / COUNT(*), 2)     AS default_rate_pct
FROM credit_card_default
GROUP BY education_label, marriage_label, sex_label
ORDER BY default_rate_pct DESC;

-- Age bucket risk view
CREATE VIEW vw_age_risk AS
SELECT
    CASE
        WHEN age BETWEEN 20 AND 29 THEN '20-29'
        WHEN age BETWEEN 30 AND 39 THEN '30-39'
        WHEN age BETWEEN 40 AND 49 THEN '40-49'
        WHEN age BETWEEN 50 AND 59 THEN '50-59'
        ELSE '60+'
    END AS age_group,
    COUNT(*)                                         AS total,
    ROUND(AVG(credit_limit), 0)                      AS avg_limit,
    ROUND(100.0 * SUM(is_default) / COUNT(*), 2)     AS default_rate_pct
FROM credit_card_default
GROUP BY age_group
ORDER BY age_group;
