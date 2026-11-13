"""Step 2 - Prepare the data.

Build prediction features on the time-sorted hourly series:
  - hour of day (0-23)
  - day of week (Monday=0)
  - lag_1, lag_2, lag_3, lag_24, lag_168
Then drop the first 168 rows that lack a full week of history.
"""
import pandas as pd

DATA_PATH = "data/one_meter.csv"

# ---- Load & sort by time --------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("=" * 70)
print("STEP 2 - PREPARE THE DATA")
print("=" * 70)

print("\n--- Loaded shape ---")
print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")

# ---- Calendar features ----------------------------------------------------
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.day_of_week  # Monday=0

# ---- Lag features (information available BEFORE hour t) -------------------
# shift(n) pulls the value from n rows earlier, so for row t it gives the
# reading at t-n.  None of these peek into the future.
df["lag_1"] = df["electricity_kwh"].shift(1)
df["lag_2"] = df["electricity_kwh"].shift(2)
df["lag_3"] = df["electricity_kwh"].shift(3)
df["lag_24"] = df["electricity_kwh"].shift(24)
df["lag_168"] = df["electricity_kwh"].shift(168)

# Target is the current hour's consumption
df["target"] = df["electricity_kwh"]

# ---- Define feature order (used later for the model & dashboard) ----------
FEATURES = ["hour", "dayofweek", "lag_1", "lag_2", "lag_3", "lag_24", "lag_168"]
TARGET = "target"

print("\n--- Feature columns ---")
for f in FEATURES:
    print(f"  {f}")
print(f"  target: {TARGET}")

# ---- Drop the first 168 rows (no full week of history yet) ----------------
rows_before = len(df)
df_model = df.iloc[168:].reset_index(drop=True)
rows_after = len(df_model)

print("\n--- Dropping first 168 rows ---")
print(f"Rows before: {rows_before}")
print(f"Rows after:  {rows_after}")
print(f"Removed:     {rows_before - rows_after}")

# ---- Checks ---------------------------------------------------------------
print("\n--- Check: any missing values in feature/target columns? ---")
print(df_model[FEATURES + [TARGET]].isna().sum().to_string())

print("\n--- Check: feature dtypes ---")
print(df_model[FEATURES].dtypes.to_string())

print("\n--- First 5 model-ready rows ---")
print(df_model[["timestamp"] + FEATURES + [TARGET]].head().to_string())

print("\n--- Last 5 model-ready rows ---")
print(df_model[["timestamp"] + FEATURES + [TARGET]].tail().to_string())

print("\n--- Summary of all features + target ---")
print(df_model[FEATURES + [TARGET]].describe().to_string())

print("\n" + "=" * 70)
print("STEP 2 COMPLETE")
print("=" * 70)
