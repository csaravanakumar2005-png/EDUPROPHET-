import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 55)
print("  EDUPROPHET AI — Setup")
print("=" * 55)

# ── Step 1: Generate data ─────────────────────────────────
print("\n[1/2] Generating synthetic datasets …")
from data.generate_data import generate_dropout_data, generate_placement_data

os.makedirs("data", exist_ok=True)
generate_dropout_data().to_csv("data/dropout_data.csv",   index=False)
generate_placement_data().to_csv("data/placement_data.csv", index=False)
print("  ✅ data/dropout_data.csv   created")
print("  ✅ data/placement_data.csv created")

# ── Step 2: Train models ──────────────────────────────────
print("\n[2/2] Training ML models …")
from src.dropout_model   import train_and_evaluate as train_d
from src.placement_model import train_and_evaluate as train_p

d_results = train_d(csv_path="data/dropout_data.csv",   model_dir="models")
p_results = train_p(csv_path="data/placement_data.csv", model_dir="models")

print("\n  Dropout Risk — Model Comparison")
for m, v in d_results.items():
    if m == "best_model": continue
    star = "★" if m == d_results["best_model"] else " "
    print(f"    {star} {m:<22} acc={v['accuracy']}%  cv={v['cv_accuracy']}%")

print("\n  Placement Readiness — Model Comparison")
for m, v in p_results.items():
    if m in ("best_model", "regressor"): continue
    star = "★" if m == p_results["best_model"] else " "
    print(f"    {star} {m:<22} acc={v['accuracy']}%  cv={v['cv_accuracy']}%")

print("\n" + "=" * 55)
print("  ✅ Setup complete!")
print("  Run:  streamlit run app.py")
print("=" * 55)