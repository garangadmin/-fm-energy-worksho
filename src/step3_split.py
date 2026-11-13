"""Step 3 - Split into train and test sets.

Hold back the last two weeks (14 days * 24 = 336 hours) as a test set
that the model never sees during training. Train on everything before.
"""
import pandas as pd

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

# Drop first 168 rows (no full week of history)
df_model = df.iloc[168:].reset_index(drop=True)

# ---- Chronological split --------------------------------------------------
train = df_model.iloc[:-TEST_HOURS].reset_index(drop=True)
test = df_model.iloc[-TEST_HOURS:].reset_index(drop=True)

print("=" * 70)
print("STEP 3 - SPLIT INTO TRAIN AND TEST SETS")
print("=" * 70)

print("\n--- Split configuration ---")
print(f"Test set size:  last {TEST_HOURS} hours = {TEST_HOURS // 24} days")

print("\n--- Sizes ---")
print(f"Total model-ready rows: {len(df_model)}")
print(f"Train rows:             {len(train)}")
print(f"Test rows:              {len(test)}")
print(f"Train + Test:           {len(train) + len(test)}")

print("\n--- Date boundaries ---")
print(f"Train: {train['timestamp'].min()}  ->  {train['timestamp'].max()}")
print(f"Test:  {test['timestamp'].min()}  ->  {test['timestamp'].max()}")

# ---- Checks ---------------------------------------------------------------
print("\n--- Check: no overlap between train and test ---")
train_end = train["timestamp"].max()
test_start = test["timestamp"].min()
gap_hours = (test_start - train_end).total_seconds() / 3600
print(f"Train ends: {train_end}")
print(f"Test starts: {test_start}")
print(f"Gap between them: {gap_hours} hours (positive = test is strictly later)")

print("\n--- Check: missing values ---")
print("Train NaNs:", train[FEATURES + [TARGET]].isna().sum().sum())
print("Test  NaNs:", test[FEATURES + [TARGET]].isna().sum().sum())

print("\n--- First 3 test rows ---")
print(test[["timestamp"] + FEATURES + [TARGET]].head(3).to_string())

print("\n--- Last 3 test rows ---")
print(test[["timestamp"] + FEATURES + [TARGET]].tail(3).to_string())

print("\n" + "=" * 70)
print("STEP 3 COMPLETE")
print("=" * 70)
