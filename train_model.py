"""
FloodSense — Model Training Script
===================================
Generates synthetic dataset using your Monte Carlo simulation (floodnew.py),
trains Random Forest and Gradient Boosting models, compares them,
and saves the best performing model to model.pkl.

Run once:  python train_model.py
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

from floodnew import generate_dataset

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
N = 1000    # number of scenarios
M = 10000   # Monte Carlo simulations per scenario
SEED = 42

np.random.seed(SEED)

# ─────────────────────────────────────────────
# GENERATE DATASET
# ─────────────────────────────────────────────
print("=" * 55)
print("  FloodSense — Model Training")
print("=" * 55)
print(f"\n[1/4] Generating dataset: {N} scenarios × {M} Monte Carlo simulations...")
print("      (This may take 30–60 seconds)\n")

data = generate_dataset(N, M)
df = pd.DataFrame(data, columns=["rain", "infi", "drain", "runoff", "slope", "flood_prob"])

print(f"      ✓ Dataset generated — {len(df)} samples")
print(f"      Flood probability range: {df['flood_prob'].min():.3f} → {df['flood_prob'].max():.3f}")
print(f"      Mean flood probability:  {df['flood_prob'].mean():.3f}\n")

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X = df[["rain", "infi", "drain", "runoff", "slope"]]
y = df["flood_prob"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# ─────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────
print("[2/4] Training models...\n")

# Random Forest
print("      Training Random Forest...")
rf = RandomForestRegressor(n_estimators=150, max_depth=None, random_state=SEED)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mse  = mean_squared_error(y_test, rf_pred)
rf_r2   = r2_score(y_test, rf_pred)
print(f"      ✓ Random Forest   — R²: {rf_r2:.4f}  |  MSE: {rf_mse:.6f}")

# Gradient Boosting
print("      Training Gradient Boosting...")
gb = GradientBoostingRegressor(random_state=SEED)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_mse  = mean_squared_error(y_test, gb_pred)
gb_r2   = r2_score(y_test, gb_pred)
print(f"      ✓ Gradient Boosting — R²: {gb_r2:.4f}  |  MSE: {gb_mse:.6f}\n")

# ─────────────────────────────────────────────
# SELECT BEST MODEL
# ─────────────────────────────────────────────
print("[3/4] Selecting best model...")

if gb_r2 >= rf_r2:
    best_model = gb
    best_name  = "Gradient Boosting"
    best_r2    = gb_r2
    best_mse   = gb_mse
else:
    best_model = rf
    best_name  = "Random Forest"
    best_r2    = rf_r2
    best_mse   = rf_mse

print(f"      ✓ Winner: {best_name}  (R²: {best_r2:.4f})\n")

# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("[4/4] Saving model to model.pkl... ✓")

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────
importance = pd.Series(best_model.feature_importances_, index=X.columns)

plt.figure(figsize=(7, 4))
ax = importance.sort_values().plot(kind='barh', color='#0080ff', edgecolor='none')
plt.title(f"Feature Importance — {best_name}", fontsize=12, fontweight='bold')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()
print("      Feature importance chart saved to feature_importance.png\n")

print("=" * 55)
print("  Training complete. Ready to run the app:")
print("  streamlit run floodsense_app.py")
print("=" * 55)