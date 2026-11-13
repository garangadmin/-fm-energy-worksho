"""Step 1 - Understand the data.

Load data/one_meter.csv, describe it, and flag anything that needs
cleaning or preprocessing.
"""
import pandas as pd

DATA_PATH = "data/one_meter.csv"

# ---- Load -----------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("STEP 1 - UNDERSTAND THE DATA")
print("=" * 70)

# ---- Basic shape & columns ------------------------------------------------
print("\n--- Shape ---")
print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")

print("\n--- Column names & dtypes ---")
print(df.dtypes)

print("\n--- First 5 rows ---")
print(df.head().to_string())

print("\n--- Last 5 rows ---")
print(df.tail().to_string())

# ---- Missing values -------------------------------------------------------
print("\n--- Missing values per column ---")
print(df.isna().sum())

# ---- Unique meters / buildings -------------------------------------------
print("\n--- Unique meter_id values ---")
print(df["meter_id"].unique())

print("\n--- Unique building_id values ---")
print(df["building_id"].unique())

# ---- Timestamp parsing & continuity --------------------------------------
print("\n--- Timestamp handling ---")
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Min timestamp: {df['timestamp'].min()}")
print(f"Max timestamp: {df['timestamp'].max()}")
print(f"Span (days):   {(df['timestamp'].max() - df['timestamp'].min()).days}")

# Check for duplicate timestamps
dupes = df["timestamp"].duplicated().sum()
print(f"Duplicate timestamps: {dupes}")

# Check for hourly gaps by sorting and comparing diffs
df_sorted = df.sort_values("timestamp").reset_index(drop=True)
diffs = df_sorted["timestamp"].diff().dropna()
print("\n--- Time-step distribution (hours between consecutive rows) ---")
print(diffs.value_counts())

expected_hours = (df_sorted["timestamp"].max() - df_sorted["timestamp"].min()) \
    .total_seconds() / 3600 + 1
print(f"\nExpected hourly rows: {int(expected_hours)}")
print(f"Actual rows:          {len(df_sorted)}")

# ---- Numeric target summary ----------------------------------------------
print("\n--- electricity_kwh summary ---")
print(df_sorted["electricity_kwh"].describe().to_string())

print("\n" + "=" * 70)
print("STEP 1 COMPLETE")
print("=" * 70)
