"""Shared 1 → 2 → 3 → 4 research-chain navigation."""

from __future__ import annotations

import streamlit as st

CHAIN = [
    ("1", "Data fusion", "Member 1", "Weekly panel and reliability"),
    ("2", "Price forecast", "Member 2", "Calibrated interval for the week"),
    ("3", "Crop viability", "Member 3", "MSRS / DCVS planting advice"),
    ("4", "Season simulation", "Member 4", "Market-scale cobweb evaluation"),
]

LABEL_BY_ID = {item[0]: item[1] for item in CHAIN}
ID_BY_LABEL = {item[1]: item[0] for item in CHAIN}


def render_hero() -> None:
    st.markdown(
        """
        <header class="studio-mast">
            <p class="studio-name">Cobweb forecast studio</p>
            <p class="studio-line">Member 2 · J26-DS-322 · weekly wholesale prices</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


MEMBER_NOTE = (
    "This branch is Member 2 only. Crop viability and the season simulator stay on the team branch."
)


def query_text(name: str) -> str | None:
    raw = st.query_params.get(name)
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def go_member(member_id: str) -> None:
    if member_id not in LABEL_BY_ID:
        member_id = "2"
    st.session_state.active_member = member_id
    st.session_state.chain_label = LABEL_BY_ID[member_id]
    st.session_state["_member_qp"] = member_id
    if query_text("member") != member_id:
        st.query_params["member"] = member_id
    st.rerun()


def render_chain_nav() -> str:
    raw = query_text("member")
    if raw is None:
        requested = "2"
    elif raw in LABEL_BY_ID:
        requested = raw
    else:
        requested = "2"
        st.session_state["_unknown_member"] = raw
    if st.session_state.get("active_member") != requested or st.session_state.get("_member_qp") != raw:
        st.session_state.active_member = requested
        st.session_state.chain_label = LABEL_BY_ID[requested]
        st.session_state["_member_qp"] = raw

    cols = st.columns(4)
    for col, (member_id, label, _name, _blurb) in zip(cols, CHAIN):
        with col:
            active = st.session_state.active_member == member_id
            if st.button(
                label,
                key=f"chain_btn_{member_id}",
                type="primary" if active else "secondary",
                use_container_width=True,
            ):
                go_member(member_id)

    return st.session_state.active_member


def render_external_member(member_id: str) -> None:
    """Member 3 and Member 4 stay on this note. Their apps are not started."""
    from lib.style import page_header

    title = "Crop viability" if member_id == "3" else "Season simulation"
    page_header("", f"Member {member_id} · {title}", MEMBER_NOTE)
    back, forward = st.columns(2)
    if member_id == "3":
        with back:
            if st.button("Back to Member 2", key="back_m3", use_container_width=True):
                go_member("2")
        with forward:
            if st.button("Continue to Member 4", key="cont_m4", type="primary", use_container_width=True):
                go_member("4")
    else:
        with back:
            if st.button("Back to Member 3", key="back_m4", use_container_width=True):
                go_member("3")
        with forward:
            if st.button("Back to Member 2", key="cont_m2", type="primary", use_container_width=True):
                go_member("2")
