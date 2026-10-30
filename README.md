# 📉 Customer Churn Intelligence Platform
**End-to-end Data Science project:** messy data → cleaning → EDA → ML model → interactive dashboard → AI-driven retention decisions.

## What it does
| Stage | What happens | File |
|---|---|---|
| 1. Data | Generates a realistic *dirty* telecom dataset (5,060 rows: duplicates, missing values, outliers, inconsistent labels) | `src/generate_data.py` |
| 2. Cleaning | Dedupes, standardises categories, fixes outliers, group-wise imputation, feature engineering, data-quality report | `src/clean.py` |
| 3. Analysis | Churn drivers, distributions, correlation heatmap, auto-written insights | `src/eda.py` |
| 4. ML | Compares Logistic Regression / Random Forest / Gradient Boosting with 5-fold CV, picks best, evaluates (ROC-AUC, precision, recall, F1), permutation feature importance, out-of-fold scoring | `src/train.py` |
| 5. AI decisions | Expected-value targeting (ROI of retention offers) + personalised retention advice via **LLM (Claude API)** with rule-based fallback | `src/ai_advisor.py` |
| 6. Dashboard | 5-tab Streamlit app: Overview, Insights, Model, Predict, AI Decision Center | `app.py` |

## Results (on included synthetic data)
- Test ROC-AUC ≈ **0.77**, recall ≈ **0.72** (class-balanced; tuned to catch churners)
- Top drivers: contract type, tenure, internet service, payment method, support tickets
- Month-to-month customers churn ~4.4× more than two-year contracts
- AI Decision Center ranks customers by **expected net benefit** and sizes the ROI of a campaign

## Run it
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py          # generate -> clean -> EDA -> train -> score
streamlit run app.py            # open dashboard
```
**Optional LLM mode:** `export ANTHROPIC_API_KEY=your_key` (Windows: `set ANTHROPIC_API_KEY=...`) before launching the dashboard.

## Use your own data
Replace `data/raw/customers_raw.csv` with a real dataset (e.g. Kaggle *Telco Customer Churn*), then adapt column names in `src/clean.py` and `FEATURES` in `src/train.py`.

## Tech stack
Python · pandas · NumPy · scikit-learn · Matplotlib · Plotly · Streamlit · joblib · Anthropic Claude API

## Project structure
```
churn-ai-project/
├── app.py                 # Streamlit dashboard
├── run_pipeline.py        # one-command pipeline
├── src/                   # generate_data, clean, eda, train, ai_advisor
├── data/{raw,processed}/  # datasets
├── models/                # saved sklearn pipeline
└── reports/               # metrics.json, insights.txt, figures/
```

## Resume bullets (copy/paste)
- Built an end-to-end churn prediction platform in Python: cleaned a 5K-row messy dataset (dedup, outlier handling, group-wise imputation), engineered features, and benchmarked 3 ML models with 5-fold cross-validation (ROC-AUC 0.77, recall 0.72).
- Developed an interactive 5-tab Streamlit dashboard (Plotly) for KPI tracking, churn-driver analysis, real-time customer scoring and model monitoring.
- Designed an AI decision layer combining expected-value optimisation with LLM-generated (Claude API) retention strategies, quantifying campaign ROI and ranking customers by expected net benefit.
