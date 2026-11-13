"""FM Energy Workshop — Streamlit Dashboard.

Displays hourly electricity readings from three energy meters stored in a
Neon PostgreSQL database.  The data is labelled *historical* — it is a static
snapshot, not a live sensor feed.

Run from the project root (fm-energy-workshop/) with:

    ..\\vibe\\Scripts\\streamlit.exe run dashboard\\app.py --server.headless true

or, if you have activated the venv:

    streamlit run dashboard/app.py --server.headless true
"""

from __future__ import annotations

import os
import sys
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# The script lives in  <project_root>/dashboard/app.py
# The .env file lives in <project_root>/.env
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

# Distinct colour per meter (wrapped in a list for color_discrete_sequence)
METER_COLORS = ["#00B4D8", "#FB8500", "#2A9D8F"]


# ---------------------------------------------------------------------------
# .env loader (no python-dotenv dependency)
# ---------------------------------------------------------------------------
def load_env(path: str) -> dict[str, str]:
    """Parse a simple .env file into a dict."""
    env: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


# ---------------------------------------------------------------------------
# Neon HTTP SQL API helper
# ---------------------------------------------------------------------------
def get_neon_connection_string() -> str:
    """Return the DATABASE_URL from .env with channel_binding stripped."""
    env = load_env(ENV_PATH)
    conn_str = env.get("DATABASE_URL")
    if not conn_str:
        st.error("DATABASE_URL not found in .env")
        st.stop()
    # Strip channel_binding=require — it can break the HTTP proxy path
    conn_str = conn_str.replace("&channel_binding=require", "")
    return conn_str


def get_sql_url(conn_str: str) -> str:
    """Build the https://<host>/sql endpoint."""
    parsed = urlparse(conn_str)
    return f"https://{parsed.hostname}/sql"


def run_sql(url: str, conn_str: str, query: str, params: list | None = None) -> dict:
    """Execute one SQL statement via the Neon HTTP SQL API."""
    headers = {
        "Neon-Connection-String": conn_str,
        "Content-Type": "application/json",
    }
    body: dict = {"query": query}
    if params:
        body["params"] = params
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"SQL error: {data['error']}")
    return data


def fetch_readings() -> pd.DataFrame:
    """Fetch all rows from meter_readings_source and return a tidy DataFrame."""
    conn_str = get_neon_connection_string()
    sql_url = get_sql_url(conn_str)

    result = run_sql(
        sql_url,
        conn_str,
        """
        SELECT
            meter_id,
            timestamp::text,
            building_id,
            electricity_kwh
        FROM meter_readings_source
        ORDER BY meter_id, timestamp
        """,
    )

    df = pd.DataFrame(result["rows"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["electricity_kwh"] = pd.to_numeric(df["electricity_kwh"])
    return df


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FM Energy Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚡ FM Energy Dashboard")
    st.markdown("---")
    st.markdown("**Data source:** Neon PostgreSQL")
    st.markdown("**Data type:** 📦 Historical")
    st.markdown(
        "Readings are loaded from the `meter_readings_source` table. "
        "This is a static historical snapshot — not a live sensor feed."
    )
    st.markdown("---")
    st.markdown("**Run command** (from project root):")
    st.code(
        "..\\vibe\\Scripts\\streamlit.exe run dashboard\\app.py "
        "--server.headless true",
        language="bash",
    )


# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def cached_fetch() -> pd.DataFrame:
    return fetch_readings()


df = cached_fetch()

meters = sorted(df["meter_id"].unique())
buildings: dict[str, str] = (
    df.groupby("meter_id")["building_id"].first().to_dict()
)

# Summary metrics in the sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("**Summary**")
    st.metric("Meters", len(meters))
    st.metric("Total rows", f"{len(df):,}")
    st.metric("Date range",
              f"{df['timestamp'].min():%Y-%m-%d} → {df['timestamp'].max():%Y-%m-%d}")


# ---------------------------------------------------------------------------
# Title bar
# ---------------------------------------------------------------------------
st.title("⚡ Energy Usage Dashboard")
st.caption("Historical hourly electricity consumption · 3 meters · Neon PostgreSQL")
st.divider()

# ---------------------------------------------------------------------------
# Section 1 — Three meters side by side
# ---------------------------------------------------------------------------
st.subheader("Hourly Electricity — Individual Meters")

cols = st.columns(3)

for i, meter in enumerate(meters):
    with cols[i]:
        sub = df[df["meter_id"] == meter].sort_values("timestamp")
        building = buildings.get(meter, "")
        st.markdown(f"**{meter}**")
        st.caption(building)

        fig = px.line(
            sub,
            x="timestamp",
            y="electricity_kwh",
            labels={
                "timestamp": "Time",
                "electricity_kwh": "kWh",
            },
            color_discrete_sequence=[METER_COLORS[i % len(METER_COLORS)]],
            template="plotly_white",
        )
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None,
            yaxis_title="kWh",
            showlegend=False,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Total energy usage
# ---------------------------------------------------------------------------
st.subheader("Total Energy Usage — All Meters Combined")

total_df = (
    df.groupby("timestamp", as_index=False)["electricity_kwh"]
    .sum()
    .sort_values("timestamp")
)

fig_total = go.Figure(
    go.Scatter(
        x=total_df["timestamp"],
        y=total_df["electricity_kwh"],
        fill="tozeroy",
        fillcolor="rgba(0, 180, 216, 0.12)",
        line=dict(color="#0077B6", width=2),
        name="Total kWh",
        hovertemplate="Time: %{x}<br>Total: %{y:.2f} kWh<extra></extra>",
    )
)
fig_total.update_layout(
    template="plotly_white",
    height=350,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title=None,
    yaxis_title="kWh",
    hovermode="x unified",
)
st.plotly_chart(fig_total, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Placeholder: Prediction
# ---------------------------------------------------------------------------
st.subheader("🔮 Energy Usage Prediction")
st.info(
    "Prediction feature coming soon. "
    "This section will use the trained GradientBoostingRegressor "
    "(`models/gbr_model.joblib`) to forecast the next hour's consumption."
)

st.divider()

# ---------------------------------------------------------------------------
# Section 4 — Placeholder: LLM Analysis
# ---------------------------------------------------------------------------
st.subheader("🤖 LLM-Powered Analysis")
st.info(
    "LLM analysis feature coming soon. "
    "This section will send recent readings and predictions to an LLM "
    "for a natural-language energy-usage summary."
)
