"""Theme helpers. Light matches Members 3/4. Dark is the Member 2 studio look."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import plotly.graph_objects as go
import streamlit as st

BLUE = "#2563eb"
NAVY = "#1e3a8a"
SLATE = "#475569"
LINE = "#e2e8f0"
PANEL = "#f8fafc"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED = "#ef4444"
SKY = GREEN
HARVEST = BLUE

CHART_CONFIG = {"displayModeBar": False, "responsive": True}

SHARED_CSS = """
    [data-testid="stMainBlockContainer"] {
        padding-top: 1.1rem;
        padding-bottom: 2.4rem;
        max-width: 1280px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px;
        padding: 0.42rem 0.7rem !important;
        margin-bottom: 0.15rem;
    }
    .team-hero {
        background: linear-gradient(120deg, #0f3d2e 0%, #1e3a8a 58%, #2563eb 100%);
        color: #f8fafc;
        padding: 1.15rem 1.35rem 1.2rem;
        border-radius: 14px;
        margin-bottom: 0.85rem;
    }
    .team-hero h1 {
        font-size: 1.45rem;
        margin: 0 0 0.2rem 0;
        letter-spacing: -0.02em;
        color: #f8fafc !important;
    }
    .team-hero p { margin: 0; color: #dbeafe; font-size: 0.95rem; }
    .team-hero .tag {
        display: inline-block;
        margin-top: 0.65rem;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.22);
        padding: 0.18rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        letter-spacing: 0.04em;
    }
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.02rem; margin-bottom: 1.1rem; }
    .metric-card {
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }
    .safe-alert, .warn-alert, .critical-alert {
        padding: 15px;
        border-radius: 8px;
        margin: 0.6rem 0;
    }
    .safe-alert { background: #d1fae5; border-left: 5px solid #10b981; color: #064e3b; }
    .warn-alert { background: #fef3c7; border-left: 5px solid #f59e0b; color: #78350f; }
    .critical-alert { background: #fee2e2; border-left: 5px solid #ef4444; color: #7f1d1d; }
    .recommendation-card {
        background: #f0fdf4;
        border: 2px solid #22c55e;
        padding: 20px;
        border-radius: 10px;
        margin-top: 16px;
        color: #14532d;
    }
    .diag-box, .diag-box-warn {
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
    }
    .diag-box { background: #f0fdf4; border-left: 5px solid #22c55e; color: #14532d; }
    .diag-box-warn { background: #fef2f2; border-left: 5px solid #ef4444; color: #7f1d1d; }
    .stage-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.65rem;
        margin: 0.15rem 0 0.85rem;
    }
    .stage-card {
        border-radius: 10px;
        padding: 0.85rem 0.9rem 1rem;
    }
    .stage-role {
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-weight: 600;
    }
    .stage-title { font-size: 1.05rem; font-weight: 700; margin: 0.28rem 0 0.2rem; }
    .stage-body { font-size: 0.88rem; }
    .price-row {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0.55rem;
        margin: 0.7rem 0 0.75rem;
    }
    .price-tile {
        border-radius: 10px;
        padding: 0.75rem 0.7rem 0.85rem;
    }
    .price-label {
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .price-value {
        font-size: 1.3rem;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        line-height: 1.15;
    }
    .range-track {
        height: 7px;
        border-radius: 99px;
        position: relative;
        margin: 0.35rem 0 0.45rem;
    }
    .range-fill { height: 100%; border-radius: 99px; width: 100%; }
    .range-mark {
        position: absolute;
        top: 50%;
        width: 11px;
        height: 11px;
        border-radius: 50%;
        transform: translate(-50%, -50%);
    }
    .fact-row {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.38rem 0;
        font-size: 0.92rem;
    }
    .fact-row b { font-weight: 600; }
    .fact-row b.ready { color: #059669; }
    .mist { font-size: 0.9rem; line-height: 1.45; }
    .slider-ends {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        margin-top: -0.25rem;
    }
"""

LIGHT_CSS = """
    .stApp { background: #ffffff; }
    [data-testid="stHeader"] { background: #ffffff; }
    [data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { color: #334155; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: #dbeafe; color: #1e3a8a; font-weight: 600;
    }
    .main-header { color: #1e3a8a; }
    .sub-header { color: #475569; }
    .metric-card { background: #f8fafc; }
    .stage-card { background: #f8fafc; border: 1px solid #e2e8f0; }
    .stage-card.active { border-color: #3b82f6; border-left: 5px solid #3b82f6; background: #eff6ff; }
    .stage-role, .stage-body, .price-label, .fact-row span, .mist, .slider-ends { color: #64748b; }
    .stage-title { color: #1e3a8a; }
    .price-tile { border: 1px solid #e2e8f0; background: #f8fafc; }
    .price-tile.accent { border-color: #3b82f6; background: #eff6ff; border-left: 5px solid #3b82f6; }
    .price-value { color: #0f172a; }
    .price-tile.accent .price-value { color: #1e3a8a; }
    .range-track { background: #e2e8f0; }
    .range-fill { background: #93c5fd; }
    .range-mark { background: #2563eb; }
    .fact-row { border-bottom: 1px solid #e2e8f0; }
    .fact-row b { color: #0f172a; }
"""

DARK_CSS = """
    :root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {
        --text-color: #eaf3ec !important;
        --textColor: #eaf3ec !important;
        --secondary-text-color: #8aa396 !important;
        --background-color: #07110e !important;
        --backgroundColor: #07110e !important;
        --secondary-background-color: #10211b !important;
        --secondaryBackgroundColor: #10211b !important;
        color-scheme: dark;
    }
    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(37,99,235,0.12), transparent 50%),
            radial-gradient(900px 500px at 110% 10%, rgba(245,185,66,0.07), transparent 45%),
            #07110e;
        color: #eaf3ec !important;
    }
    [data-testid="stHeader"] { background: #07110e !important; }
    [data-testid="stSidebar"] { background: #0c1914 !important; border-right: 1px solid #1d3a30; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { color: rgba(234,243,236,0.88) !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: #2563eb; color: #f8fafc !important; font-weight: 600;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] [data-testid="stCaption"],
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: #eaf3ec !important;
    }
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stMarkdown"],
    [data-testid="stMarkdown"] p,
    [data-testid="stMarkdown"] span,
    [data-testid="stHeading"] *,
    [data-testid="stWidgetLabel"] *,
    [data-testid="stCaption"],
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {
        color: #eaf3ec !important;
    }
    [data-testid="stCaption"],
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {
        color: #9bb8ac !important;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * { color: #9bb8ac !important; }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * { color: #eaf3ec !important; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #10211b !important;
        border-color: #1d3a30 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] p,
    [data-testid="stVerticalBlockBorderWrapper"] span,
    [data-testid="stVerticalBlockBorderWrapper"] label,
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdown"] {
        color: #eaf3ec !important;
    }
    [data-testid="stTabs"] button { color: #eaf3ec !important; background: #10211b !important; }
    [data-testid="stTabs"] button[aria-selected="true"] { background: #2563eb !important; color: #ffffff !important; }
    [data-testid="stMainBlockContainer"] .stButton button[kind="secondary"],
    [data-testid="stMainBlockContainer"] button[data-testid="stBaseButton-secondary"] {
        background: #10211b !important;
        color: #eaf3ec !important;
        border: 1px solid #334155 !important;
    }
    [data-testid="stMainBlockContainer"] .stButton button[kind="primary"],
    [data-testid="stMainBlockContainer"] button[data-testid="stBaseButton-primary"] {
        background: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #2563eb !important;
    }
    .stApp .main-header { color: #eaf3ec !important; }
    .stApp .sub-header { color: #9bb8ac !important; }
    .metric-card { background: #10211b; color: #eaf3ec; }
    .metric-card * { color: #eaf3ec !important; }
    .stage-card { background: #10211b; border: 1px solid #1d3a30; }
    .stage-card.active { border-color: #3b82f6; border-left: 5px solid #3b82f6; background: #132a46; }
    .stage-role, .stage-body, .price-label, .fact-row span, .mist, .slider-ends { color: #9bb8ac; }
    .stage-title { color: #eaf3ec; }
    .price-tile { border: 1px solid #1d3a30; background: #0c1914; }
    .price-tile.accent { border-color: #3b82f6; background: rgba(37,99,235,0.16); border-left: 5px solid #3b82f6; }
    .price-value { color: #eaf3ec; }
    .price-tile.accent .price-value { color: #93c5fd; }
    .range-track { background: #07110e; border: 1px solid #1d3a30; }
    .range-fill { background: rgba(37,99,235,0.35); }
    .range-mark { background: #60a5fa; }
    .fact-row { border-bottom: 1px solid #1d3a30; }
    .fact-row b { color: #eaf3ec; }
    .fact-row b.ready { color: #4ade80; }
    [data-testid="stInfo"], [data-testid="stSuccess"], [data-testid="stWarning"] { color: inherit; }
"""


def current_theme() -> str:
    return "dark"


def inject() -> None:
    st.markdown(
        f"<style>{SHARED_CSS}{DARK_CSS}</style>",
        unsafe_allow_html=True,
    )


def page_header(kicker_text: str, title: str, blurb: str) -> None:
    st.caption(kicker_text.upper())
    st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{blurb}</div>', unsafe_allow_html=True)


def section_title(kicker_text: str, title: str) -> None:
    st.caption(kicker_text.upper())
    st.subheader(title)


def stage_row(items: list[dict[str, Any]]) -> None:
    cards = []
    for item in items:
        active = " active" if item.get("active") else ""
        role = f'<div class="stage-role">{item.get("role", "")}</div>' if item.get("role") else ""
        body = item.get("body") or ""
        cards.append(
            f'<div class="stage-card{active}">{role}'
            f'<div class="stage-title">{item.get("title", "")}</div>'
            f'<div class="stage-body">{body}</div></div>'
        )
    st.markdown(f'<div class="stage-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def card_grid(items: list[dict[str, Any]], columns: int = 4) -> None:
    cards = []
    for item in items:
        active = " active" if item.get("active") else ""
        role = f'<div class="stage-role">{item.get("role", "")}</div>' if item.get("role") else ""
        body = f'<div class="stage-body">{item.get("body") or ""}</div>' if item.get("body") else ""
        rows = "".join(
            f'<div class="fact-row"><span>{label}</span><b>{value}</b></div>'
            for label, value in (item.get("rows") or [])
        )
        note = f'<p class="mist">{item["note"]}</p>' if item.get("note") else ""
        cards.append(
            f'<div class="stage-card{active}">{role}'
            f'<div class="stage-title">{item.get("title", "")}</div>'
            f"{body}{rows}{note}</div>"
        )
    st.markdown(
        f'<div class="stage-row" style="grid-template-columns:repeat({columns},1fr)">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def fact(label: str, value: str, ready: bool = False) -> None:
    klass = "ready" if ready else ""
    st.markdown(
        f'<div class="fact-row"><span>{label}</span><b class="{klass}">{value}</b></div>',
        unsafe_allow_html=True,
    )


def range_bar(low: float, mid: float, high: float) -> None:
    span = max(high - low, 1)
    pos = min(100.0, max(0.0, 100.0 * (mid - low) / span))
    st.markdown(
        f'<div class="range-track"><div class="range-fill"></div>'
        f'<div class="range-mark" style="left:{pos:.1f}%"></div></div>'
        f'<p class="mist">90% likely range · expected sits inside the band</p>',
        unsafe_allow_html=True,
    )


def chart_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    dark = current_theme() == "dark"
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,14,0.35)" if dark else "#f8fafc",
        font=dict(color="#8aa396" if dark else SLATE, size=12),
        margin=dict(l=40, r=16, t=18, b=78),
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),
        hovermode="x unified",
        uirevision="htfe-studio",
    )
    grid = "#1d3a30" if dark else LINE
    fig.update_xaxes(gridcolor=grid, zeroline=False, linecolor=grid)
    fig.update_yaxes(gridcolor=grid, zeroline=False, linecolor=grid)
    return fig


def show_chart(fig: go.Figure, height: int = 360) -> None:
    st.plotly_chart(chart_layout(fig, height), width="stretch", config=CHART_CONFIG)


def pick(
    label: str,
    options: Sequence[str],
    key: str,
    preferred: str | None = None,
    format_func: Callable[[str], str] | None = None,
) -> str:
    opts = list(options)
    if not opts:
        raise ValueError(f"No options for {label}")
    if st.session_state.get(key) not in opts:
        st.session_state[key] = preferred if preferred in opts else opts[0]
    return st.selectbox(label, opts, key=key, format_func=format_func or (lambda value: str(value)))
