# Customer Churn Prediction

A small project where I try to predict which telecom customers will cancel, and then work out which of them are actually worth spending money to keep.

The prediction part is fairly standard. The part I found more interesting was the last step: a customer with a high churn probability isn't always worth chasing. If they pay a low monthly bill, a retention offer can cost more than the revenue you'd save. So the project ranks customers by expected benefit instead of just by risk.

**Note on the data:** the dataset is synthetic. I generated it with a script (`src/generate_data.py`) and deliberately made it messy, with duplicate rows, missing values, outliers and inconsistent labels, so there was something to clean. Churn is produced from rules I wrote, so the patterns the model finds are ones I put there. The numbers below say something about the pipeline, not about real customers. Swapping in a real dataset is on my list (see the end).

## What's in here

- `src/generate_data.py` creates about 5,000 customers plus 60 duplicate rows
- `src/clean.py` removes duplicates, standardises text values, handles outliers and missing values, and builds a few extra features
- `src/eda.py` produces the charts in `reports/figures` and a short `insights.txt`
- `src/train.py` compares three models, evaluates the best one, and saves it
- `src/ai_advisor.py` handles the retention logic (details below)
- `app.py` is the Streamlit dashboard
- `run_pipeline.py` runs all of the above in order

## Cleaning

The raw file has 5,060 rows. After dropping duplicates, 5,000 remain. The other fixes:

- contract and gender values like `"month-to-month "`, `"MONTH-TO-MONTH"`, `"M"` and `"F "` are mapped to one spelling each
- tenure values above 72 months are impossible for this business, so I treated them as missing and filled them with the median
- missing monthly charges are filled with the median for the same internet service type, and missing total charges with monthly charges times tenure
- new columns: average monthly spend, tickets per year and a tenure group

## Model

I compared logistic regression, random forest and gradient boosting using 5-fold cross-validation on 80% of the data and held out 20% for testing. Cross-validated ROC-AUC:

| Model | ROC-AUC |
|---|---|
| Logistic regression | 0.783 |
| Gradient boosting | 0.767 |
| Random forest | 0.759 |

Logistic regression won, which isn't surprising since the synthetic churn rule is roughly linear. On the held-out test set:

- ROC-AUC 0.77
- accuracy 0.67
- precision 0.44
- recall 0.72

Precision is low. I used balanced class weights to catch more of the people who leave, so the model flags a lot of customers who would have stayed. Whether that trade is acceptable depends on how cheap the retention offer is, which is why the next step exists.

## Churn patterns in the data

Overall churn is 27.8%. The biggest differences:

- month-to-month contracts churn at 37.6%, two-year contracts at 8.5%
- customers in their first year churn at 46.5%
- fiber optic customers churn more than DSL or no-internet customers
- electronic check payers churn more than card payers

Again, these come from how I generated the data.

## Retention decisions

`src/ai_advisor.py` does two things.

**Who to target.** For each customer it computes:

```
expected benefit = churn probability x offer success rate x 12 months of revenue - offer cost
```

I assumed a $100 offer cost and a 30% success rate. These are guesses, not measured values, and you can change them in the dashboard. Customers with a positive expected benefit are worth contacting, ranked from highest to lowest.

**What to offer.** For targeted customers it suggests an action such as moving month-to-month customers to a longer contract, offering free tech support, or escalating customers with many support tickets. By default these suggestions come from simple if/else rules. If you set an `ANTHROPIC_API_KEY` environment variable, it asks the Claude API for a recommendation instead. The prompt is basic, and the rule-based version is the more predictable of the two.

## Running it

You need Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate        # on Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

`run_pipeline.py` regenerates the data, cleans it, trains the models and scores every customer. The dashboard then reads those results. It has five tabs: overview, churn breakdowns, model results, a form to score a single customer, and the targeting list with the cost assumptions.

## Limitations

- Synthetic data, as mentioned above
- No hyperparameter tuning, I used mostly default settings
- The business assumptions (offer cost, success rate) are not based on any real campaign data
- The LLM option is experimental, and the rule-based advice is the default

