"""Shared 1 → 2 → 3 → 4 research-chain navigation."""

from __future__ import annotations

import streamlit as st

CHAIN = [
    ("1", "01  Data Fusion", "Member 1", "Weekly panel and reliability"),
    ("2", "02  Price Forecast", "Member 2", "Calibrated interval for the week"),
    ("3", "03  Crop Viability", "Member 3", "MSRS / DCVS planting advice"),
    ("4", "04  Season Simulation", "Member 4", "Market-scale cobweb evaluation"),
]

LABEL_BY_ID = {item[0]: item[1] for item in CHAIN}
ID_BY_LABEL = {item[1]: item[0] for item in CHAIN}


def render_hero() -> None:
    st.markdown(
        """
        <div class="team-hero">
            <h1>Cobweb Decision Support System</h1>
            <p>Data-driven prototype to mitigate boom–bust vegetable prices in Sri Lanka</p>
            <span class="tag">J26-DS-322 &nbsp;·&nbsp; MEMBER 1 → 2 → 3 → 4</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sync_query() -> None:
    st.query_params["member"] = st.session_state.active_member


def go_member(member_id: str) -> None:
    if member_id not in LABEL_BY_ID:
        member_id = "2"
    st.session_state.active_member = member_id
    st.session_state.chain_label = LABEL_BY_ID[member_id]
    st.query_params["member"] = member_id
    st.rerun()


def render_chain_nav() -> str:
    requested = str(st.query_params.get("member", "2"))
    if requested not in LABEL_BY_ID:
        requested = "2"
    if "active_member" not in st.session_state:
        st.session_state.active_member = requested

    cols = st.columns(4)
    for col, (member_id, label, _name, _blurb) in zip(cols, CHAIN):
        with col:
            active = st.session_state.active_member == member_id
            if st.button(label, key=f"chain_btn_{member_id}", type="primary" if active else "secondary", use_container_width=True):
                go_member(member_id)

    _sync_query()
    return st.session_state.active_member
