import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/customers_raw.csv")


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(1, n + 1)],
        "gender": rng.choice(["Male", "Female"], n),
        "senior_citizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "tenure_months": rng.integers(1, 73, n),
        "contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]),
        "internet_service": rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.44, 0.34, 0.22]),
        "payment_method": rng.choice(
            ["Electronic check", "Credit card", "Bank transfer", "Mailed check"], n, p=[0.34, 0.22, 0.22, 0.22]),
        "paperless_billing": rng.choice(["Yes", "No"], n, p=[0.6, 0.4]),
        "tech_support": rng.choice(["Yes", "No"], n, p=[0.35, 0.65]),
        "online_security": rng.choice(["Yes", "No"], n, p=[0.35, 0.65]),
        "support_tickets": rng.poisson(1.5, n),
    })
    base = df["internet_service"].map({"Fiber optic": 80, "DSL": 55, "No": 20})
    df["monthly_charges"] = (base + rng.normal(0, 8, n)).round(2).clip(18, 120)
    df["total_charges"] = (df["monthly_charges"] * df["tenure_months"] * rng.uniform(0.95, 1.05, n)).round(2)

    # Churn driven by realistic factors
    z = (-1.2
         + 1.3 * (df["contract"] == "Month-to-month")
         - 1.0 * (df["contract"] == "Two year")
         + 0.7 * (df["internet_service"] == "Fiber optic")
         + 0.6 * (df["payment_method"] == "Electronic check")
         - 0.035 * df["tenure_months"]
         + 0.012 * (df["monthly_charges"] - 60)
         + 0.25 * df["support_tickets"]
         - 0.5 * (df["tech_support"] == "Yes")
         - 0.4 * (df["online_security"] == "Yes")
         + 0.3 * df["senior_citizen"]
         + rng.normal(0, 0.5, n))
    df["churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "Yes", "No")

    # ---- inject mess ----
    idx = lambda frac: rng.choice(n, int(n * frac), replace=False)
    df.loc[idx(0.04), "total_charges"] = np.nan
    df.loc[idx(0.03), "monthly_charges"] = np.nan
    df.loc[idx(0.02), "payment_method"] = np.nan
    df.loc[idx(0.01), "tenure_months"] = rng.integers(200, 999, int(n * 0.01))   # outliers
    for i in idx(0.10):                                                          # inconsistent labels
        df.at[i, "contract"] = rng.choice(["month-to-month ", "MONTH-TO-MONTH", "Month to month"])
    for i in idx(0.05):
        df.at[i, "gender"] = rng.choice(["male", "FEMALE", "M", "F "])
    df = pd.concat([df, df.sample(60, random_state=seed)], ignore_index=True)    # duplicates
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = generate()
    data.to_csv(RAW_PATH, index=False)
    print(f"Generated raw dataset: {data.shape} -> {RAW_PATH}")
