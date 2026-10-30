import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src import ai_advisor
from src.train import FEATURES

st.set_page_config(page_title="Churn Intelligence", page_icon="📉", layout="wide")


@st.cache_data
def load():
    df = pd.read_csv("data/processed/scored_customers.csv")
    metrics = json.load(open("reports/metrics.json"))
    imp = pd.read_csv("reports/feature_importance.csv")
    return df, metrics, imp


if not Path("data/processed/scored_customers.csv").exists():
    st.error("Run `python run_pipeline.py` first to generate data and train the model.")
    st.stop()

df, metrics, imp = load()
model = joblib.load("models/churn_model.joblib")
top_drivers = imp.head(4)["feature"].tolist()

# ---------------- Sidebar filters ----------------
st.sidebar.title("Filters")
contracts = st.sidebar.multiselect("Contract", df.contract.unique(), df.contract.unique())
internet = st.sidebar.multiselect("Internet service", df.internet_service.unique(), df.internet_service.unique())
view = df[df.contract.isin(contracts) & df.internet_service.isin(internet)]

st.title("📉 Customer Churn Intelligence Platform")
# st.caption("Data cleaning → analysis → ML prediction → AI-driven retention decisions")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🔎 Insights", "🤖 Model", "🎯 Predict a customer", "🧠 AI Decision Center"])

# ---------------- Overview ----------------
with tab1:
    c = st.columns(4)
    c[0].metric("Customers", f"{len(view):,}")
    c[1].metric("Churn rate", f"{view.churn_flag.mean():.1%}")
    c[2].metric("Avg monthly charge", f"{view.monthly_charges.mean():,.2f}")
    c[3].metric("High-risk customers", f"{(view.risk_band == 'High').sum():,}")
    a, b = st.columns(2)
    a.plotly_chart(px.pie(view, names="churn", title="Churn split", hole=.5,
                          color_discrete_sequence=["#4C78A8", "#E45756"]), use_container_width=True)
    b.plotly_chart(px.histogram(view, x="churn_probability", color="churn", nbins=30, barmode="overlay",
                                title="Predicted churn probability"), use_container_width=True)

# ---------------- Insights ----------------
with tab2:
    col = st.selectbox("Break churn down by", ["contract", "internet_service", "payment_method",
                                                "tenure_group", "tech_support", "online_security", "paperless_billing"])
    g = view.groupby(col)["churn_flag"].agg(["mean", "count"]).reset_index()
    g["churn_rate_%"] = (g["mean"] * 100).round(1)
    st.plotly_chart(px.bar(g, x=col, y="churn_rate_%", text="churn_rate_%", color="churn_rate_%",
                           color_continuous_scale="Reds", title=f"Churn rate by {col}"), use_container_width=True)
    st.plotly_chart(px.box(view, x="churn", y="monthly_charges", color="churn", title="Monthly charges vs churn"),
                    use_container_width=True)
    st.plotly_chart(px.scatter(view.sample(min(1500, len(view)), random_state=1), x="tenure_months",
                               y="monthly_charges", color="churn", opacity=.6, title="Tenure vs charges"),
                    use_container_width=True)

# ---------------- Model ----------------
with tab3:
    m = st.columns(5)
    for i, k in enumerate(["test_roc_auc", "test_accuracy", "test_precision", "test_recall", "test_f1"]):
        m[i].metric(k.replace("test_", "").upper(), metrics[k])
    st.write(f"**Selected model:** {metrics['best_model']} (best 5-fold CV ROC-AUC)")
    st.dataframe(pd.DataFrame(metrics["model_comparison"]).T, use_container_width=True)
    st.plotly_chart(px.bar(imp.head(10).iloc[::-1], x="importance", y="feature", orientation="h",
                           title="Top churn drivers (permutation importance)"), use_container_width=True)
    a, b = st.columns(2)
    a.image("reports/figures/roc_curve.png"); b.image("reports/figures/confusion_matrix.png")

# ---------------- Predict ----------------
with tab4:
    st.subheader("Score a single customer")
    f1, f2, f3 = st.columns(3)
    contract = f1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_s = f1.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    payment = f1.selectbox("Payment method", ["Electronic check", "Credit card", "Bank transfer", "Mailed check"])
    tenure = f2.slider("Tenure (months)", 1, 72, 6)
    monthly = f2.slider("Monthly charges", 18.0, 120.0, 85.0)
    tickets = f2.slider("Support tickets", 0, 10, 3)
    gender = f3.selectbox("Gender", ["Male", "Female"])
    senior = f3.selectbox("Senior citizen", [0, 1])
    tech = f3.selectbox("Tech support", ["No", "Yes"])
    sec = f3.selectbox("Online security", ["No", "Yes"])
    paperless = f3.selectbox("Paperless billing", ["Yes", "No"])

    if st.button("Predict churn risk", type="primary"):
        total = monthly * tenure
        row = pd.DataFrame([{
            "senior_citizen": senior, "tenure_months": tenure, "monthly_charges": monthly, "total_charges": total,
            "support_tickets": tickets, "avg_monthly_spend": total / tenure, "tickets_per_year": tickets / (tenure / 12),
            "gender": gender, "contract": contract, "internet_service": internet_s, "payment_method": payment,
            "paperless_billing": paperless, "tech_support": tech, "online_security": sec}])[FEATURES]
        p = float(model.predict_proba(row)[0, 1])
        band = "High" if p > .6 else "Medium" if p > .3 else "Low"
        st.metric("Churn probability", f"{p:.1%}", band + " risk")
        st.progress(min(p, 1.0))
        r = row.iloc[0].to_dict(); r["churn_probability"] = p
        advice, src = ai_advisor.advise(r, top_drivers)
        st.info(f"**Recommended action ({src}):** {advice}")

# ---------------- AI Decision Center ----------------
with tab5:
    st.subheader("Who should we target, and with what?")
    s1, s2, s3 = st.columns(3)
    cost = s1.number_input("Retention offer cost per customer", 0.0, 1000.0, ai_advisor.DEFAULTS["offer_cost"])
    succ = s2.slider("Offer success rate", 0.05, 0.9, ai_advisor.DEFAULTS["success_rate"])
    hor = s3.slider("Revenue horizon (months)", 3, 36, ai_advisor.DEFAULTS["horizon_months"])

    val = ai_advisor.add_business_value(view, cost, succ, hor)
    tgt = val[val.target_offer]
    k = st.columns(3)
    k[0].metric("Customers to target", f"{len(tgt):,}", f"{len(tgt)/max(len(val),1):.0%} of base")
    k[1].metric("Revenue at risk (targeted)", f"{tgt.expected_loss.sum():,.0f}")
    k[2].metric("Expected net saving", f"{tgt.expected_net_benefit.sum():,.0f}")
    st.success(ai_advisor.executive_summary(view, offer_cost=cost, success_rate=succ, horizon_months=hor))

    st.markdown("#### Priority list (ranked by expected net benefit)")
    n = st.slider("Show top N", 5, 50, 15)
    top = tgt.head(n)[["customer_id", "contract", "tenure_months", "monthly_charges",
                       "churn_probability", "expected_net_benefit"]]
    st.dataframe(top, use_container_width=True)
    st.download_button("⬇ Download target list (CSV)", tgt.to_csv(index=False), "retention_targets.csv")

    st.markdown("#### AI retention advisor")
    st.caption("Uses an LLM when ANTHROPIC_API_KEY is set; otherwise a rule-based expert system.")
    pick = st.selectbox("Choose a customer", tgt.head(50)["customer_id"])
    if st.button("Generate recommendation"):
        r = tgt[tgt.customer_id == pick].iloc[0].to_dict()
        advice, src = ai_advisor.advise(r, top_drivers)
        st.write(f"**Churn probability:** {r['churn_probability']:.0%} | **Contract:** {r['contract']} | "
                 f"**Tenure:** {r['tenure_months']:.0f} months")
        st.info(f"**{src} recommendation:** {advice}")
