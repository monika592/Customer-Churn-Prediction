import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/customers_raw.csv")
CLEAN = Path("data/processed/customers_clean.csv")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"rows_raw": len(df)}

    # 1. Remove duplicates
    df = df.drop_duplicates().copy()
    df = df.drop_duplicates(subset="customer_id")
    report["duplicates_removed"] = report["rows_raw"] - len(df)

    # 2. Standardise text categories
    text_cols = ["gender", "contract", "internet_service", "payment_method",
                 "paperless_billing", "tech_support", "online_security", "churn"]
    for col in text_cols:
        df[col] = df[col].str.strip()
    df["contract"] = (df["contract"].str.lower().str.replace(" ", "-", regex=False)
                      .map({"month-to-month": "Month-to-month", "one-year": "One year", "two-year": "Two year"}))
    df["gender"] = df["gender"].str.lower().str[0].map({"m": "Male", "f": "Female"})

    # 3. Missing values
    report["missing_before"] = df.isna().sum()[lambda s: s > 0].to_dict()
    df["payment_method"] = df["payment_method"].fillna(df["payment_method"].mode()[0])

    # 4. Outliers: tenure cannot exceed 72 months for this business
    report["tenure_outliers"] = int((df["tenure_months"] > 72).sum())
    df.loc[df["tenure_months"] > 72, "tenure_months"] = np.nan
    df["tenure_months"] = df["tenure_months"].fillna(df["tenure_months"].median())

    # 5. Smart imputation for charges
    df["monthly_charges"] = df["monthly_charges"].fillna(
        df.groupby("internet_service")["monthly_charges"].transform("median"))
    df["total_charges"] = df["total_charges"].fillna(df["monthly_charges"] * df["tenure_months"])

    # 6. Feature engineering
    df["avg_monthly_spend"] = (df["total_charges"] / df["tenure_months"].clip(lower=1)).round(2)
    df["tickets_per_year"] = (df["support_tickets"] / (df["tenure_months"].clip(lower=1) / 12)).round(2)
    df["tenure_group"] = pd.cut(df["tenure_months"], [0, 12, 24, 48, 72],
                                labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]).astype(str)
    df["churn_flag"] = (df["churn"] == "Yes").astype(int)

    report["rows_clean"] = len(df)
    report["missing_after"] = int(df.isna().sum().sum())
    return df.reset_index(drop=True), report


if __name__ == "__main__":
    raw = pd.read_csv(RAW)
    out, rep = clean(raw)
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLEAN, index=False)
    print("=== DATA QUALITY REPORT ===")
    for k, v in rep.items():
        print(f"{k:>20}: {v}")
    print(f"Saved clean data -> {CLEAN}")
