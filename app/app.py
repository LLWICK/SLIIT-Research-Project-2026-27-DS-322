"""
HTFE Studio — Hybrid Temporal Forecasting Engine
Interactive Streamlit dashboard (Member 2 - Garusingarachchi Y.B · IT23415836)
SLIIT IT4010 Research Project J26-DS-322

Member 2 owns only the forecast studio. This branch is the forecast studio only.
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
from lib.nav import render_chain_nav, render_hero
from lib.style import inject
from lib.experiment_c import experiment_c
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
    "What we did": training.render,
    "Results": results.render,
    "Which model": comparison.render,
    "Cultivation proof": novelty.render,
    "Uncertainty": uncertainty.render,
    "Data": data.render,
    "Features": features.render,
}
PAGE_ALIASES = {
    "Training": "What we did",
    "Comparison": "Which model",
    "Novelty": "Cultivation proof",
}

if "dash_intensity_pct" not in st.session_state:
    st.session_state.dash_intensity_pct = int(ANCHOR * 100)

names = list(STUDIO_PAGES)
requested_page = PAGE_ALIASES.get(st.query_params.get("page", "Dashboard"), st.query_params.get("page", "Dashboard"))
if requested_page not in STUDIO_PAGES:
    requested_page = "Dashboard"
if "nav" not in st.session_state:
    st.session_state.nav = requested_page
elif st.session_state.nav in PAGE_ALIASES:
    st.session_state.nav = PAGE_ALIASES[st.session_state.nav]


def _sync_page() -> None:
    st.query_params["page"] = st.session_state.nav


render_hero()
active_member = render_chain_nav()
inject()

if active_member in {"3", "4"}:
    st.info("This branch is Member 2 only. Crop viability and the season simulator stay on the team branch.")
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
        run = experiment_c()
        if run.get("trained"):
            st.caption(
                "Walk the pages in order: what we did, the full-sample results, which model is best, then the matched-row cultivation proof. "
                "The dashboard forecast is the runtime model. It is not the Experiment B or Experiment C accuracy result."
            )
        else:
            st.caption("Scores are the 2024–2025 test. The split is chronological. Experiment C has not been trained in this run.")
    try:
        studio = load_studio()
    except FileNotFoundError as exc:
        st.error("Could not load demo artifacts.")
        st.caption(str(exc))
        st.stop()
    STUDIO_PAGES[st.session_state.nav](studio)
