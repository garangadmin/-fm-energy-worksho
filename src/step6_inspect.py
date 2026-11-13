"""Step 6 - Inspect example predictions and save the work.

- Plot actual vs. predicted over the two-week test window.
- Save the fitted model to models/gbr_model.joblib.
- Save the feature order to models/feature_order.json.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/one_meter.csv"
TEST_HOURS = 14 * 24  # 336
MODELS_DIR = "models"
PLOT_PATH = os.path.join(MODELS_DIR, "test_predictions.png")

# ---- Load & sort by time --------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ---- Build features (same as Step 2) --------------------------------------
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.day_of_week  # Monday=0
df["lag_1"] = df["electricity_kwh"].shift(1)
df["lag_2"] = df["electricity_kwh"].shift(2)
df["lag_3"] = df["electricity_kwh"].shift(3)
df["lag_24"] = df["electricity_kwh"].shift(24)
df["lag_168"] = df["electricity_kwh"].shift(168)
df["target"] = df["electricity_kwh"]

FEATURES = ["hour", "dayofweek", "lag_1", "lag_2", "lag_3", "lag_24", "lag_168"]
TARGET = "target"

# Drop first 168 rows, then split chronologically
df_model = df.iloc[168:].reset_index(drop=True)
train = df_model.iloc[:-TEST_HOURS].reset_index(drop=True)
test = df_model.iloc[-TEST_HOURS:].reset_index(drop=True)

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# ---- Train the model ------------------------------------------------------
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# ---- Predict on test set --------------------------------------------------
pred_model = model.predict(X_test)

# ---- Metrics (recap) ------------------------------------------------------
mae_model = mean_absolute_error(y_test, pred_model)
rmse_model = np.sqrt(mean_squared_error(y_test, pred_model))
mae_base = mean_absolute_error(y_test, test["lag_1"].values)
rmse_base = np.sqrt(mean_squared_error(y_test, test["lag_1"].values))

print("=" * 70)
print("STEP 6 - INSPECT PREDICTIONS & SAVE WORK")
print("=" * 70)

print("\n--- Test metrics recap ---")
print(f"Model    MAE: {mae_model:.4f} kWh   RMSE: {rmse_model:.4f} kWh")
print(f"Baseline MAE: {mae_base:.4f} kWh   RMSE: {rmse_base:.4f} kWh")

# ---- Plot actual vs predicted --------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test["timestamp"], y_test.values, label="Actual", linewidth=1.4)
ax.plot(test["timestamp"], pred_model, label="Predicted (GBR)",
        linewidth=1.4, alpha=0.8)
ax.set_title("Actual vs Predicted Electricity Consumption (Test Set: Feb 16–29)")
ax.set_xlabel("Date")
ax.set_ylabel("Electricity (kWh)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()

os.makedirs(MODELS_DIR, exist_ok=True)
fig.savefig(PLOT_PATH, dpi=150)
plt.close(fig)
print(f"\n--- Plot saved ---")
print(f"Path: {PLOT_PATH}")

# ---- Save the fitted model ------------------------------------------------
model_path = os.path.join(MODELS_DIR, "gbr_model.joblib")
dump(model, model_path)
print(f"\n--- Model saved ---")
print(f"Path: {model_path}")

# ---- Save the feature order ----------------------------------------------
feature_path = os.path.join(MODELS_DIR, "feature_order.json")
with open(feature_path, "w") as f:
    json.dump(FEATURES, f, indent=2)
print(f"\n--- Feature order saved ---")
print(f"Path: {feature_path}")
print(f"Features: {FEATURES}")

# ---- Verify saved artifacts load back ------------------------------------
from joblib import load

loaded_model = load(model_path)
with open(feature_path) as f:
    loaded_features = json.load(f)
sample_pred = loaded_model.predict(X_test.iloc[:3])

print("\n--- Verification ---")
print(f"Loaded model type: {type(loaded_model).__name__}")
print(f"Loaded features:   {loaded_features}")
print(f"Sample predictions (first 3 test rows): "
      f"{[round(p, 2) for p in sample_pred]}")

print("\n" + "=" * 70)
print("STEP 6 COMPLETE")
print("=" * 70)
