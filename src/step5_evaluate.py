"""Step 5 - Evaluate the model.

Compare the GradientBoostingRegressor's MAE and RMSE (in kWh) against a
baseline that simply uses the previous hour's consumption (lag_1) as the
prediction, on the same test rows.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/one_meter.csv"
TEST_HOURS = 14 * 24  # 336

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

# Baseline: previous hour's consumption = lag_1
pred_baseline = test["lag_1"].values

# ---- Metrics --------------------------------------------------------------
mae_model = mean_absolute_error(y_test, pred_model)
rmse_model = np.sqrt(mean_squared_error(y_test, pred_model))

mae_base = mean_absolute_error(y_test, pred_baseline)
rmse_base = np.sqrt(mean_squared_error(y_test, pred_baseline))

print("=" * 70)
print("STEP 5 - EVALUATE THE MODEL")
print("=" * 70)

print("\n--- Test set ---")
print(f"Rows:          {len(test)}")
print(f"Date range:    {test['timestamp'].min()}  ->  {test['timestamp'].max()}")

print("\n--- Results (lower is better) ---")
print(f"{'Metric':<10} {'Baseline (lag_1)':<22} {'Model (GBR)':<18} {'Difference'}")
print("-" * 70)
print(f"{'MAE':<10} {mae_base:<22.4f} {mae_model:<18.4f} {mae_model - mae_base:+.4f} kWh")
print(f"{'RMSE':<10} {rmse_base:<22.4f} {rmse_model:<18.4f} {rmse_model - rmse_base:+.4f} kWh")

print("\n--- Improvement over baseline ---")
mae_improve = (mae_base - mae_model) / mae_base * 100
rmse_improve = (rmse_base - rmse_model) / rmse_base * 100
print(f"MAE improvement:  {mae_improve:.1f}%")
print(f"RMSE improvement: {rmse_improve:.1f}%")

print("\n" + "=" * 70)
print("STEP 5 COMPLETE")
print("=" * 70)
