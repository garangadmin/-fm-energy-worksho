"""
Upload three_meters.csv to Neon via the HTTP SQL API (POST /sql).

Uses the requests library to talk to Neon's SQL-over-HTTP proxy on port 443,
avoiding the need for a direct libpq connection on port 5432.
"""

import csv
import json
import os
import sys
from urllib.parse import urlparse

import requests

# Force unbuffered stdout so progress is visible in real time
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "three_meters.csv")
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
BATCH_SIZE = 100  # rows per HTTP request


def load_database_url() -> str:
    """Read DATABASE_URL from .env (simple parser, no extra deps)."""
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if line.startswith("DATABASE_URL"):
                # extract the value between quotes
                _, _, value = line.partition("=")
                value = value.strip().strip('"').strip("'")
                # Strip channel_binding=require if present
                value = value.replace("&channel_binding=require", "")
                return value
    raise RuntimeError("DATABASE_URL not found in .env")


def build_sql_url(conn_str: str) -> str:
    """Build the https://<host>/sql endpoint from the connection string."""
    parsed = urlparse(conn_str)
    return f"https://{parsed.hostname}/sql"


def run_sql(url: str, conn_str: str, query: str, params: list | None = None) -> dict:
    """Execute a single SQL statement via the Neon HTTP API."""
    headers = {
        "Neon-Connection-String": conn_str,
        "Content-Type": "application/json",
    }
    body = {"query": query}
    if params:
        body["params"] = params
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code}: {resp.text[:500]}", file=sys.stderr)
        resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        print(f"  SQL error: {data['error']}", file=sys.stderr)
        raise RuntimeError(data["error"])
    return data


def main():
    # --- Load connection string -------------------------------------------------
    conn_str = load_database_url()
    print(f"Connection string (sanitised): ...{conn_str[-40:]}")

    sql_url = build_sql_url(conn_str)
    print(f"SQL API endpoint: {sql_url}")

    # --- Quick connectivity test -------------------------------------------------
    print("\n[1/5] Testing connection...")
    result = run_sql(sql_url, conn_str, "SELECT 1 AS ok")
    print(f"  OK – server returned: {result.get('rows')}")

    # --- Drop & create table ----------------------------------------------------
    print("\n[2/5] Creating table meter_readings_source...")
    run_sql(
        sql_url,
        conn_str,
        "DROP TABLE IF EXISTS meter_readings_source",
    )
    run_sql(
        sql_url,
        conn_str,
        """
        CREATE TABLE meter_readings_source (
            timestamp      TIMESTAMP NOT NULL,
            meter_id       TEXT      NOT NULL,
            building_id    TEXT      NOT NULL,
            electricity_kwh NUMERIC   NOT NULL,
            UNIQUE (meter_id, timestamp)
        )
        """,
    )
    print("  Table created with UNIQUE(meter_id, timestamp)")

    # --- Read CSV and batch-insert ----------------------------------------------
    print("\n[3/5] Reading CSV and uploading rows...")
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    print(f"  CSV contains {total} data rows")

    # Build parameterised multi-row INSERTs in batches
    for start in range(0, total, BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        placeholders = []
        params = []
        for row in batch:
            base = len(placeholders) // 4 * 4  # not used, just clarity
            n = len(params)
            placeholders.append(
                f"(${n + 1}::timestamp, ${n + 2}::text, ${n + 3}::text, ${n + 4}::numeric)"
            )
            params.extend([
                row["timestamp"],
                row["meter_id"],
                row["building_id"],
                row["electricity_kwh"],
            ])
        query = (
            "INSERT INTO meter_readings_source "
            "(timestamp, meter_id, building_id, electricity_kwh) VALUES "
            + ", ".join(placeholders)
        )
        run_sql(sql_url, conn_str, query, params)
        done = start + len(batch)
        print(f"  Uploaded {done}/{total} rows", end="\r")

    print(f"\n  All {total} rows uploaded.")

    # --- Verify: row count, min/max timestamp per meter --------------------------
    print("\n[4/5] Verifying upload...")
    result = run_sql(
        sql_url,
        conn_str,
        """
        SELECT
            COUNT(*) AS total_rows
        FROM meter_readings_source
        """,
    )
    print(f"  Total rows in table: {result['rows'][0]['total_rows']}")

    print("\n[5/5] Per-meter summary (row count, earliest & latest timestamp):")
    print("-" * 80)
    result = run_sql(
        sql_url,
        conn_str,
        """
        SELECT
            meter_id,
            COUNT(*)                       AS row_count,
            MIN(timestamp)::TEXT            AS earliest_timestamp,
            MAX(timestamp)::TEXT            AS latest_timestamp
        FROM meter_readings_source
        GROUP BY meter_id
        ORDER BY meter_id
        """,
    )
    # Print as a table
    print(f"{'meter_id':<12} {'row_count':>10} {'earliest_timestamp':<22} {'latest_timestamp':<22}")
    print("-" * 80)
    for row in result["rows"]:
        print(f"{row['meter_id']:<12} {row['row_count']:>10} {row['earliest_timestamp']:<22} {row['latest_timestamp']:<22}")

    print("\nDone.")


if __name__ == "__main__":
    main()
