"""FM Energy Workshop — Streamlit Dashboard v2 (Simulation).

Replays historical meter readings from Neon as a simulated live feed.
At startup, January 2016 data is loaded for all three meters.  Every real
minute the next recorded hour is fetched from Neon and appended to the
charts, until the full dataset (through Feb 29) is loaded.

Run from the project root (fm-energy-workshop/) with:

    ..\\vibe\\Scripts\\streamlit.exe run dashboard\\app_v2.py --server.headless true
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

METER_COLORS = ["#00B4D8", "#FB8500", "#2A9D8F"]

# Start of the dataset and end of the initial January load
DATA_START = pd.Timestamp("2016-01-01 00:00:00")
JAN_END = pd.Timestamp("2016-01-31 23:00:00")


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
# Neon HTTP SQL API helpers
# ---------------------------------------------------------------------------
def get_neon_connection_string() -> str:
    """Return the DATABASE_URL from .env with channel_binding stripped."""
    env = load_env(ENV_PATH)
    conn_str = env.get("DATABASE_URL")
    if not conn_str:
        st.error("DATABASE_URL not found in .env")
        st.stop()
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


# ---------------------------------------------------------------------------
# Data fetch functions
# ---------------------------------------------------------------------------
def _rows_to_df(result: dict) -> pd.DataFrame:
    """Convert Neon API response rows to a typed DataFrame."""
    rows = result.get("rows", [])
    if not rows:
        return pd.DataFrame(
            columns=["meter_id", "timestamp", "building_id", "electricity_kwh"]
        )
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["electricity_kwh"] = pd.to_numeric(df["electricity_kwh"])
    return df


@st.cache_data(ttl=3600, show_spinner="Loading January data from Neon…")
def fetch_initial_data(conn_str: str, sql_url: str) -> pd.DataFrame:
    """Fetch all January readings (Jan 1 00:00 through Jan 31 23:00)."""
    result = run_sql(
        sql_url,
        conn_str,
        """
        SELECT meter_id, timestamp::text, building_id, electricity_kwh
        FROM meter_readings_source
        WHERE timestamp <= '2016-01-31 23:00:00'
        ORDER BY meter_id, timestamp
        """,
    )
    return _rows_to_df(result)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_max_timestamp(conn_str: str, sql_url: str) -> pd.Timestamp:
    """Get the latest timestamp in the database."""
    result = run_sql(
        sql_url,
        conn_str,
        "SELECT MAX(timestamp)::TEXT AS max_ts FROM meter_readings_source",
    )
    return pd.to_datetime(result["rows"][0]["max_ts"])


def fetch_next_hour(conn_str: str, sql_url: str, ts: pd.Timestamp) -> pd.DataFrame:
    """Fetch readings for one specific timestamp across all meters."""
    result = run_sql(
        sql_url,
        conn_str,
        """
        SELECT meter_id, timestamp::text, building_id, electricity_kwh
        FROM meter_readings_source
        WHERE timestamp = $1::timestamp
        ORDER BY meter_id
        """,
        [ts.strftime("%Y-%m-%d %H:%M:%S")],
    )
    return _rows_to_df(result)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FM Energy Dashboard v2",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Resolve Neon connection (after page config so st.error/st.stop work)
# ---------------------------------------------------------------------------
CONN_STR = get_neon_connection_string()
SQL_URL = get_sql_url(CONN_STR)


# ---------------------------------------------------------------------------
# Session state — simulation control
# ---------------------------------------------------------------------------
def init_simulation():
    """Load January data and set the simulation to its starting state."""
    jan_df = fetch_initial_data(CONN_STR, SQL_URL)
    max_ts = fetch_max_timestamp(CONN_STR, SQL_URL)

    st.session_state["sim_data"] = jan_df
    st.session_state["sim_current_ts"] = jan_df["timestamp"].max()
    st.session_state["sim_last_ts"] = max_ts
    st.session_state["sim_finished"] = st.session_state["sim_current_ts"] >= max_ts
    st.session_state["last_advance"] = datetime.now()


if "sim_data" not in st.session_state:
    init_simulation()


# ---------------------------------------------------------------------------
# Auto-advance fragment — fires every real minute
# ---------------------------------------------------------------------------
@st.fragment(run_every=timedelta(minutes=1))
def auto_tick():
    """Fetch the next recorded hour from Neon and refresh the page.

    The fragment body also runs during a normal script rerun (e.g. after
    ``st.rerun``), so a 55-second guard prevents double-advancing: only
    advance if enough wall-clock time has passed since the last advance.
    """
    if st.session_state.get("sim_finished", True):
        return

    now = datetime.now()
    last = st.session_state.get("last_advance", now)
    if (now - last) < timedelta(seconds=55):
        return

    next_ts = st.session_state["sim_current_ts"] + pd.Timedelta(hours=1)

    try:
        new_rows = fetch_next_hour(CONN_STR, SQL_URL, next_ts)
    except Exception as exc:
        st.warning(f"Could not fetch next hour: {exc}")
        return

    if len(new_rows) == 0:
        st.session_state["sim_finished"] = True
        return

    st.session_state["sim_data"] = pd.concat(
        [st.session_state["sim_data"], new_rows], ignore_index=True
    )
    st.session_state["sim_current_ts"] = next_ts
    st.session_state["last_advance"] = now

    if next_ts >= st.session_state["sim_last_ts"]:
        st.session_state["sim_finished"] = True

    st.rerun()


# Register the fragment so the timer starts
auto_tick()


# ---------------------------------------------------------------------------
# Pull current state into locals for rendering
# ---------------------------------------------------------------------------
df = st.session_state["sim_data"]
sim_current = st.session_state["sim_current_ts"]
sim_last = st.session_state["sim_last_ts"]
sim_finished = st.session_state["sim_finished"]

meters = sorted(df["meter_id"].unique())
buildings: dict[str, str] = df.groupby("meter_id")["building_id"].first().to_dict()

hours_loaded = df["timestamp"].nunique()
total_hours = int((sim_last - DATA_START) / pd.Timedelta(hours=1)) + 1


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚡ FM Energy Dashboard v2")
    st.markdown("**Simulation mode**")
    st.markdown("---")
    st.markdown("**Data source:** Neon PostgreSQL")
    st.markdown("**Data type:** 📦 Historical (simulated live)")
    st.markdown(
        "January 1–31 loads at startup. One new recorded hour "
        "is fetched every real minute until Feb 29 is reached."
    )
    st.markdown("---")

    if sim_finished:
        st.success("✅ Simulation complete — all records loaded")
    else:
        st.info("▶ Simulation running — adding 1 hour per minute")

    st.metric("Simulated time", f"{sim_current:%Y-%m-%d %H:%M}")
    st.metric("Hours loaded", f"{hours_loaded} / {total_hours}")
    st.progress(min(hours_loaded / total_hours, 1.0))

    st.markdown("---")

    if st.button("🔄 Restart simulation", use_container_width=True):
        init_simulation()
        st.rerun()

    st.markdown("---")
    st.markdown("**Run command** (from project root):")
    st.code(
        "..\\vibe\\Scripts\\streamlit.exe run dashboard\\app_v2.py "
        "--server.headless true",
        language="bash",
    )


# ---------------------------------------------------------------------------
# Title bar
# ---------------------------------------------------------------------------
st.title("⚡ Energy Usage Dashboard")
st.caption("Historical hourly electricity · simulated live feed · 3 meters · Neon PostgreSQL")
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
