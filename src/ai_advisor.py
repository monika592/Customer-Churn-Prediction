import os
import pandas as pd

# Business assumptions (editable in the dashboard sidebar)
DEFAULTS = {"offer_cost": 100.0, "success_rate": 0.30, "horizon_months": 12}


def add_business_value(df: pd.DataFrame, offer_cost=100.0, success_rate=0.30, horizon_months=12) -> pd.DataFrame:
    """Expected net benefit of sending a retention offer to each customer."""
    d = df.copy()
    d["revenue_at_risk"] = d["monthly_charges"] * horizon_months
    d["expected_loss"] = d["churn_probability"] * d["revenue_at_risk"]
    d["expected_net_benefit"] = d["churn_probability"] * success_rate * d["revenue_at_risk"] - offer_cost
    d["target_offer"] = d["expected_net_benefit"] > 0
    return d.sort_values("expected_net_benefit", ascending=False)


def rule_based_advice(row) -> str:
    """Transparent expert rules mapped from the main churn drivers."""
    tips = []
    if row["contract"] == "Month-to-month":
        tips.append("Offer a 10-15% discount to move to a 1- or 2-year contract.")
    if row["payment_method"] == "Electronic check":
        tips.append("Incentivise switching to auto-pay (card/bank) with a bill credit.")
    if row["tech_support"] == "No" and row["internet_service"] != "No":
        tips.append("Bundle 3 months of free Tech Support.")
    if row["online_security"] == "No" and row["internet_service"] != "No":
        tips.append("Offer free Online Security trial.")
    if row["support_tickets"] >= 3:
        tips.append("Escalate to a senior agent for a proactive service-quality call.")
    if row["internet_service"] == "Fiber optic" and row["monthly_charges"] > 85:
        tips.append("Review pricing - high fiber bill; offer a loyalty price lock.")
    if row["tenure_months"] <= 12:
        tips.append("Trigger an onboarding check-in; new customers churn most.")
    return " ".join(tips[:3]) or "Light-touch engagement: satisfaction survey and loyalty points."


def llm_advice(row, top_drivers: list[str]) -> str | None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        prompt = (
            "You are a telecom customer-retention strategist. Given this at-risk customer and the model's "
            f"top churn drivers ({', '.join(top_drivers)}), recommend the single best retention action, "
            "a one-line rationale, and a suggested message tone. Max 80 words.\n\n"
            f"Customer: contract={row['contract']}, tenure={row['tenure_months']} months, "
            f"monthly_charges={row['monthly_charges']}, internet={row['internet_service']}, "
            f"payment={row['payment_method']}, tickets={row['support_tickets']}, "
            f"tech_support={row['tech_support']}, churn_probability={row['churn_probability']:.0%}"
        )
        msg = client.messages.create(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
                                     max_tokens=300, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()
    except Exception as e:  # network/key issues -> graceful fallback
        return f"(LLM unavailable: {type(e).__name__}) " + rule_based_advice(row)


def advise(row, top_drivers: list[str]) -> tuple[str, str]:
    """Returns (advice, source)."""
    out = llm_advice(row, top_drivers)
    return (out, "LLM") if out else (rule_based_advice(row), "Rule-based")


def executive_summary(scored: pd.DataFrame, **assump) -> str:
    v = add_business_value(scored, **assump)
    t = v[v.target_offer]
    return (f"Target {len(t)} of {len(v)} customers ({len(t)/len(v):.0%}). Expected revenue at risk in this group: "
            f"{t.expected_loss.sum():,.0f}; expected net saving after offer cost: {t.expected_net_benefit.sum():,.0f}.")
