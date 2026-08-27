"""
HTFE Studio — Hybrid Temporal Forecasting Engine
Interactive Streamlit dashboard (Member 2 - Garusingarachchi Y.B · IT23415836)
SLIIT IT4010 Research Project J26-DS-322
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.live_match import ANCHOR
from lib.load import load_studio
from lib.style import inject
from views import comparison, dashboard, data, features, novelty, results, training, uncertainty

st.set_page_config(
    page_title="HTFE Studio · Member 2 Forecaster",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject()

PAGES = {
    "Dashboard": dashboard.render,
    "Results": results.render,
    "Data": data.render,
    "Features": features.render,
    "Training": training.render,
    "Comparison": comparison.render,
    "Uncertainty": uncertainty.render,
    "Novelty": novelty.render,
}

if "dash_intensity_pct" not in st.session_state:
    st.session_state.dash_intensity_pct = int(ANCHOR * 100)

names = list(PAGES)
requested = st.query_params.get("page", "Dashboard")
if requested not in PAGES:
    requested = "Dashboard"
if "nav" not in st.session_state:
    st.session_state.nav = requested


def _sync_page() -> None:
    st.query_params["page"] = st.session_state.nav


with st.sidebar:
    st.caption("J26-DS-322 · IT23415836")
    st.title("Price Forecast")
    st.caption("Weekly vegetable prices with a confidence range")
    page = st.radio("Navigate", names, key="nav", on_change=_sync_page, label_visibility="collapsed")
    st.markdown(
        "<p class='mist'>No backend in this demo. Every number is scored on observed wholesale prices after a chronological split.</p>",
        unsafe_allow_html=True,
    )

try:
    studio = load_studio()
except FileNotFoundError as exc:
    st.error("Could not load demo artifacts.")
    st.caption(str(exc))
    st.stop()

PAGES[page](studio)
