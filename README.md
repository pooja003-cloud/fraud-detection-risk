# 💳 Chargeback Reduction: A Cost-Optimised Fraud Risk Framework

> An end-to-end fraud risk project that links a **corporate risk assessment** to a **machine-learning model**, and measures success in **dollars saved and customer friction**, not accuracy.

🚧 **Status:** in progress (see [Roadmap](#roadmap))

---

## The business problem

A fictional online retailer, **NovaCart**, is losing about **2.5% of revenue** to card-not-present fraud. That puts it above Visa's and Mastercard's monitoring thresholds (1.5% in 2026) and at risk of losing its payment processor.

**Goal:** bring fraud below **0.5%** while keeping friction for genuine customers low.

📄 Read the full one-page risk assessment: [`docs/business_brief.md`](docs/business_brief.md)

## The solution: a 3-stage risk lifecycle

```
[Transaction] ──> [1. ML Engine] ──> [2. Risk Policy] ──> [3. Action]
                   fraud score        cost-optimised       🟢 Approve
                   (LightGBM)         thresholds           🟡 Step-up (SMS / 3DS)
                                                           🔴 Decline + manual review
```

1. **ML engine:** LightGBM scores every transaction from 0 to 1.
2. **Risk policy:** score cut-offs chosen to **minimise total financial cost**, not to maximise accuracy.
3. **Impact dashboard:** a Streamlit app comparing "no screening" vs "model + policy" on cost, fraud ratio and friction.

## Why not just report accuracy?

Over 96% of transactions are genuine, so a model that approves everything is "96% accurate" and useless. This project uses:

| Metric | Why it matters |
|---|---|
| **Total cost of fraud ($)** | Missed fraud, fees, lost customers and verification costs together |
| **Fraud chargeback ratio** | What Visa and Mastercard actually monitor |
| **False positive ratio** | Genuine customers wrongly blocked |
| **Merchant friction** | Share of genuine customers who face an extra step |
| **PR-AUC** | Model quality on a heavily imbalanced dataset |

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (real e-commerce transactions, provided by Vesta Corporation).
The data is **not stored in this repo**. See [`data/README.md`](data/README.md).

## Roadmap

- [x] **Step 1:** Business risk assessment and repo setup
- [ ] **Step 2:** Exploratory data analysis and "cost of doing nothing" baseline
- [x] **Step 3:** Baseline LightGBM model (time-based split, PR-AUC) ([notebook](notebooks/03_baseline_lightgbm.ipynb)): test PR-AUC 0.50, 14× better than random
- [x] **Step 4:** Financial cost framework and three-tier threshold optimisation ([notebook](notebooks/04_cost_optimised_policy.ipynb)): recommended policy cuts the cost of fraud by 60% (about $334k/month), fraud ratio 3.5% → 1.27%, 91% instant approval
- [ ] **Step 5:** Explainability (SHAP)
- [ ] **Step 6:** Streamlit financial impact dashboard
- [ ] **Step 7:** Results write-up

## Project structure

```
chargeback-reduction/
├── docs/              # Business brief and write-ups
├── notebooks/         # Step-by-step analysis notebooks (run on Kaggle)
├── src/               # Reusable Python code (cost functions, features)
├── app/               # Streamlit dashboard
├── data/              # Local data only (git-ignored)
├── reports/figures/   # Saved charts
├── requirements.txt
└── README.md
```

## How to run

Notebooks are designed for **Kaggle Notebooks**. Add the IEEE-CIS competition data to the notebook and run top to bottom.
To run locally: `pip install -r requirements.txt` and place the data files in `data/`.

## Results

*Coming in Step 7.*

---

*Portfolio project. NovaCart is fictional; industry figures are sourced in the business brief.*
