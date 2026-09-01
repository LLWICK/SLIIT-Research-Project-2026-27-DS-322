"""
HTFE Studio — Hybrid Temporal Forecasting Engine
Interactive Streamlit dashboard (Member 2 - Garusingarachchi Y.B · IT23415836)
SLIIT IT4010 Research Project J26-DS-322

Member 2 owns only the forecast studio. The header switches the research chain.
Members 3 and 4 are opened from their existing apps — those files are not edited.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.embed import run_existing_app
from lib.live_match import ANCHOR
from lib.load import load_studio
from lib.nav import render_chain_nav, render_hero
from lib.style import inject
from views import comparison, dashboard, data, features, novelty, results, training, uncertainty
from views import member1

st.set_page_config(
    page_title="Cobweb DSS · Member 2 Forecaster",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded",
)

STUDIO_PAGES = {
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

names = list(STUDIO_PAGES)
requested_page = st.query_params.get("page", "Dashboard")
if requested_page not in STUDIO_PAGES:
    requested_page = "Dashboard"
if "nav" not in st.session_state:
    st.session_state.nav = requested_page


def _sync_page() -> None:
    st.query_params["page"] = st.session_state.nav


render_hero()
active_member = render_chain_nav()
inject()

if active_member == "3":
    st.caption("Member 3’s existing DSS · `prototype/member3_MSRS_DCVS` — opened as-is, not rewritten.")
    run_existing_app("3")
    inject()
elif active_member == "4":
    st.caption("Member 4’s existing simulator · `prototype/member4_AgenticSimulation/veggie-abm` — opened as-is, not rewritten.")
    run_existing_app("4")
    inject()
elif active_member == "1":
    with st.sidebar:
        st.caption("J26-DS-322 · Member 2 studio")
        st.title("Member 1 · Data Fusion")
        st.caption("Their extract is the 1 → 2 contract. Their full dashboard stays on their branch.")
    try:
        member1.render(load_studio())
    except FileNotFoundError as exc:
        st.error("Could not load demo artifacts.")
        st.caption(str(exc))
else:
    with st.sidebar:
        st.caption("J26-DS-322 · IT23415836")
        st.title("Price Forecast")
        st.caption("Weekly vegetable prices with a confidence range")
        st.radio("Navigate", names, key="nav", on_change=_sync_page, label_visibility="collapsed")
        st.caption("No backend in this demo. Every number is scored on observed wholesale prices after a chronological split.")
    try:
        studio = load_studio()
    except FileNotFoundError as exc:
        st.error("Could not load demo artifacts.")
        st.caption(str(exc))
        st.stop()
    STUDIO_PAGES[st.session_state.nav](studio)
