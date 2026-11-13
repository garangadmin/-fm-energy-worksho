"""Step 4 - Train a simple boosted-tree model.

Use scikit-learn GradientBoostingRegressor with random_state=42.
Train on the chronological train set (everything before the last two weeks).
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

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

# ---- Train the model ------------------------------------------------------
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

print("=" * 70)
print("STEP 4 - TRAIN A BOOSTED-TREE MODEL")
print("=" * 70)

print("\n--- Model ---")
print(f"Type:          {type(model).__name__}")
print(f"random_state:  {model.random_state}")
print(f"n_estimators:  {model.n_estimators}")
print(f"max_depth:     {model.max_depth}")
print(f"learning_rate: {model.learning_rate}")

print("\n--- Training set ---")
print(f"Rows:          {len(train)}")
print(f"Features:      {FEATURES}")

# ---- Training-set fit (in-sample) ----------------------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

pred_train = model.predict(X_train)
mae_train = mean_absolute_error(y_train, pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
r2_train = model.score(X_train, y_train)

print("\n--- In-sample (training) metrics ---")
print(f"R^2:   {r2_train:.4f}")
print(f"MAE:   {mae_train:.4f} kWh")
print(f"RMSE:  {rmse_train:.4f} kWh")

# ---- Feature importances --------------------------------------------------
print("\n--- Feature importances ---")
importances = pd.Series(model.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)
for feat, imp in importances.items():
    print(f"  {feat:<12} {imp:.4f}")

print("\n" + "=" * 70)
print("STEP 4 COMPLETE")
print("=" * 70)
