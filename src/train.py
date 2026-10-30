"""Train, compare and evaluate models. Saves best pipeline, metrics and scored customers."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

NUM = ["senior_citizen", "tenure_months", "monthly_charges", "total_charges",
       "support_tickets", "avg_monthly_spend", "tickets_per_year"]
CAT = ["gender", "contract", "internet_service", "payment_method",
       "paperless_billing", "tech_support", "online_security"]
FEATURES = NUM + CAT


def build(model):
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("model", model)])


def run(path="data/processed/customers_clean.csv"):
    df = pd.read_csv(path)
    X, y = df[FEATURES], df["churn_flag"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    results = {}
    for name, m in candidates.items():
        auc = cross_val_score(build(m), X_tr, y_tr, cv=cv, scoring="roc_auc")
        results[name] = {"cv_auc_mean": round(auc.mean(), 4), "cv_auc_std": round(auc.std(), 4)}
        print(f"{name:<22} CV ROC-AUC = {auc.mean():.4f} (+/- {auc.std():.4f})")

    best_name = max(results, key=lambda k: results[k]["cv_auc_mean"])
    best = build(candidates[best_name]).fit(X_tr, y_tr)

    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "best_model": best_name, "model_comparison": results,
        "test_roc_auc": round(roc_auc_score(y_te, proba), 4),
        "test_accuracy": round(accuracy_score(y_te, pred), 4),
        "test_precision": round(precision_score(y_te, pred), 4),
        "test_recall": round(recall_score(y_te, pred), 4),
        "test_f1": round(f1_score(y_te, pred), 4),
        "train_rows": len(X_tr), "test_rows": len(X_te),
    }
    print(f"\nBEST: {best_name}\n", json.dumps({k: v for k, v in metrics.items() if k.startswith('test')}, indent=2))

    # Plots
    fig_dir = Path("reports/figures"); fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4)); RocCurveDisplay.from_predictions(y_te, proba, ax=ax)
    ax.set_title(f"ROC curve - {best_name}"); fig.savefig(fig_dir / "roc_curve.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 4)); ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax, cmap="Blues")
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=120); plt.close(fig)

    # Model-agnostic feature importance (on raw features -> readable for business)
    imp = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=8, random_state=42)
    imp_df = pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean}).sort_values("importance", ascending=False)
    imp_df.to_csv("reports/feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); imp_df.head(10)[::-1].plot.barh(x="feature", y="importance", ax=ax, legend=False, color="#F58518")
    ax.set_title("Top drivers of churn (permutation importance)"); plt.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=120); plt.close(fig)

    # Out-of-fold probabilities for ALL customers (honest scores for the dashboard)
    oof = cross_val_predict(build(candidates[best_name]), X, y, cv=cv, method="predict_proba")[:, 1]
    scored = df.copy(); scored["churn_probability"] = oof.round(4)
    scored["risk_band"] = pd.cut(oof, [-.01, .3, .6, 1.01], labels=["Low", "Medium", "High"]).astype(str)
    scored.to_csv("data/processed/scored_customers.csv", index=False)

    # Final model refit on all data for deployment
    final = build(candidates[best_name]).fit(X, y)
    Path("models").mkdir(exist_ok=True); joblib.dump(final, "models/churn_model.joblib")
    json.dump(metrics, open("reports/metrics.json", "w"), indent=2)
    print("Saved model, metrics, importance and scored customers.")


if __name__ == "__main__":
    run()
