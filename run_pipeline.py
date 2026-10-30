"""One command to run the whole pipeline: python run_pipeline.py"""
import pandas as pd
from src import generate_data, clean, eda, train, ai_advisor

print("\n[1/5] Generating raw (messy) data")
raw = generate_data.generate(); generate_data.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
raw.to_csv(generate_data.RAW_PATH, index=False); print(raw.shape)

print("\n[2/5] Cleaning data")
df, rep = clean.clean(pd.read_csv(clean.RAW)); df.to_csv(clean.CLEAN, index=False)
for k, v in rep.items(): print(f"  {k}: {v}")

print("\n[3/5] Exploratory analysis"); eda.run()
print("\n[4/5] Training models"); train.run()

print("\n[5/5] AI decision summary")
scored = pd.read_csv("data/processed/scored_customers.csv")
print(" ", ai_advisor.executive_summary(scored))
print("\nDone. Launch the dashboard with:  streamlit run app.py")
