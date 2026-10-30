import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path("reports/figures")


def run(path="data/processed/customers_clean.csv"):
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    insights = [f"Overall churn rate: {df.churn_flag.mean():.1%} ({df.churn_flag.sum()} of {len(df)} customers)"]

    # Churn rate by categorical drivers
    for col in ["contract", "internet_service", "payment_method", "tenure_group", "tech_support"]:
        rate = df.groupby(col)["churn_flag"].mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots(figsize=(6, 3.5))
        rate.plot.bar(ax=ax, color="#4C78A8")
        ax.set_ylabel("Churn rate (%)"); ax.set_title(f"Churn rate by {col}")
        plt.xticks(rotation=20); plt.tight_layout()
        fig.savefig(FIG / f"churn_by_{col}.png", dpi=120); plt.close(fig)
        insights.append(f"Highest churn by {col}: '{rate.index[0]}' at {rate.iloc[0]:.1f}% vs lowest '{rate.index[-1]}' at {rate.iloc[-1]:.1f}%")

    # Numeric distributions
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, col in zip(axes, ["tenure_months", "monthly_charges"]):
        for lbl, g in df.groupby("churn"):
            ax.hist(g[col], bins=25, alpha=.6, label=f"Churn={lbl}")
        ax.set_title(col); ax.legend()
    plt.tight_layout(); fig.savefig(FIG / "numeric_distributions.png", dpi=120); plt.close(fig)

    # Correlation heatmap
    num = df[["tenure_months", "monthly_charges", "total_charges", "support_tickets",
              "senior_citizen", "avg_monthly_spend", "churn_flag"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(num, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num))); ax.set_xticklabels(num.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(num))); ax.set_yticklabels(num.columns)
    fig.colorbar(im); plt.tight_layout(); fig.savefig(FIG / "correlation.png", dpi=120); plt.close(fig)

    c = num["churn_flag"].drop("churn_flag").abs().sort_values(ascending=False)
    insights.append(f"Numeric feature most correlated with churn: {c.index[0]} (|r|={c.iloc[0]:.2f})")
    Path("reports/insights.txt").write_text("\n".join(insights))
    print("\n".join(insights))


if __name__ == "__main__":
    run()
