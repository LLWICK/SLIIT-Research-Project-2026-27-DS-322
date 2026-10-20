"""
HTFE Studio — Hybrid Temporal Forecasting Engine
Interactive Streamlit dashboard (Member 2 - Garusingarachchi Y.B · IT23415836)
SLIIT IT4010 Research Project J26-DS-322

Member 2 owns only the forecast studio. This branch is the forecast studio only.
Studio pages read the saved run. They do not retrain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.experiment_c import experiment_c
from lib.live_match import ANCHOR
from lib.load import load_studio
from lib.nav import query_text, render_chain_nav, render_external_member, render_hero
from lib.style import inject
from views import comparison, dashboard, data, features, novelty, results, training, uncertainty
from views import member1

st.set_page_config(
    page_title="Cobweb forecast studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Glide paints table cells with theme.backgroundColor, which is the green shell.
# Data cells get a cream styler. The index column uses a different CSS id
# (T_<uuid>_levelN_rowR) that set_properties does not emit, so showing the
# index paints shell green. Add those ids without changing the shell color.
_orig_dataframe = st.dataframe
_INDEX_STYLE_UUID = "deskidx"
_CELL_BG = "#F7F4EE"
_CELL_INK = "#1C1915"


def _cell_text(value):
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:.10g}"
    return value


def _cream_table(frame: pd.DataFrame):
    styler = frame.style.format(_cell_text, na_rep="").set_properties(
        **{"background-color": _CELL_BG, "color": _CELL_INK}
    )
    styler.set_uuid(_INDEX_STYLE_UUID)
    selectors = [
        f"#T_{_INDEX_STYLE_UUID}_level{level}_row{row}"
        for level in range(frame.index.nlevels)
        for row in range(len(frame.index))
    ]
    if selectors:
        styler.set_table_styles(
            [
                {
                    "selector": ", ".join(selectors),
                    "props": [("background-color", _CELL_BG), ("color", _CELL_INK)],
                }
            ],
            overwrite=False,
        )
    return styler


def _readable_dataframe(data=None, *args, **kwargs):
    if isinstance(data, pd.DataFrame):
        data = _cream_table(data)
    return _orig_dataframe(data, *args, **kwargs)


st.dataframe = _readable_dataframe

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


def _known_page(raw: str | None) -> str | None:
    if not raw:
        return None
    if raw in PAGE_ALIASES:
        return PAGE_ALIASES[raw]
    if raw in STUDIO_PAGES:
        return raw
    folded = raw.casefold()
    for alias, target in PAGE_ALIASES.items():
        if alias.casefold() == folded:
            return target
    for name in STUDIO_PAGES:
        if name.casefold() == folded:
            return name
    return None


raw_page = query_text("page")
known_page = _known_page(raw_page)
url_changed = "nav" not in st.session_state or st.session_state.get("_url_page") != raw_page
if url_changed:
    if known_page:
        st.session_state.nav = known_page
    elif raw_page:
        st.session_state.nav = "Dashboard"
    elif "nav" not in st.session_state:
        st.session_state.nav = "Dashboard"
if st.session_state.get("nav") not in STUDIO_PAGES:
    st.session_state.nav = "Dashboard"
st.session_state["_url_page"] = raw_page


def _sync_page() -> None:
    st.query_params["page"] = st.session_state.nav
    st.session_state["_url_page"] = st.session_state.nav


st.session_state["_paint"] = int(st.session_state.get("_paint") or 0) + 1
inject()
render_hero()
if raw_page and known_page is None:
    st.caption(f"{raw_page} is not a page in this studio.")
active_member = render_chain_nav()
missed_member = st.session_state.pop("_unknown_member", None)
if missed_member:
    st.caption(f"Member {missed_member} is not in this chain. Showing the price forecast.")

if active_member in {"3", "4"}:
    with st.sidebar:
        st.caption("J26-DS-322 · IT23415836")
        st.title("Price forecast")
        st.caption("This studio does not open crop viability or the season simulator.")
    render_external_member(active_member)
elif active_member == "1":
    with st.sidebar:
        st.caption("J26-DS-322 · Member 2 studio")
        st.title("Member 1")
        st.caption("The weekly panel is the handoff. Member 1's own dashboard stays on their branch.")
    try:
        member1.render(load_studio())
    except FileNotFoundError as exc:
        st.caption(f"Could not open the studio files. {exc}")
else:
    with st.sidebar:
        st.caption("J26-DS-322 · IT23415836")
        st.title("Price forecast")
        st.caption("Weekly wholesale price, LKR per kg")
        st.radio("Navigate", names, key="nav", on_change=_sync_page, label_visibility="collapsed")
        run = experiment_c()
        if run.get("trained"):
            st.caption(
                "Order: What we did, Results, Which model, Cultivation proof. "
                "The dashboard is the runtime model. It is not the Experiment B or Experiment C score."
            )
        else:
            st.caption("Scores are the 2024–2025 test. The split follows issue week. Experiment C is not trained in this run.")
    try:
        studio = load_studio()
    except FileNotFoundError as exc:
        st.caption(f"Could not open the studio files. {exc}")
        st.stop()
    STUDIO_PAGES[st.session_state.nav](studio)
