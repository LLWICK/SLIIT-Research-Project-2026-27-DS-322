"""Field-green theme for the Member 2 forecast studio."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from html import escape
from typing import Any

import plotly.graph_objects as go
import streamlit as st

INK = "#1C1915"
PAPER = "#F7F4EE"
SHELL = "#16352B"
LEAF = "#2F6B4F"
GREEN = LEAF
HARVEST = LEAF
SKY = "#3F6B56"
STONE = "#3F3A34"
AMBER = "#8C5A3C"
SLATE = STONE
LINE = "#E4DDD2"
PANEL = PAPER
RED = "#8C3A32"
CLAY = AMBER
NAVY = INK
BLUE = GREEN

CHART_FONT = '"Source Sans 3", "Segoe UI", sans-serif'
CHART_CONFIG = {"displayModeBar": False, "responsive": True}

DESK_CSS = """
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"], section.main, [data-testid="stBottom"] { background: #16352B !important; }
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stStatusWidget"] { color: #F4F0E6 !important; }
[data-testid="stHeader"] button, [data-testid="stHeader"] span, [data-testid="stHeader"] p,
[data-testid="stToolbar"] button, [data-testid="stStatusWidget"] button, [data-testid="stStatusWidget"] span, [data-testid="stStatusWidget"] p { color: #F4F0E6 !important; }
[data-testid="stStatusWidget"] button { min-width: 4.5rem; }
[data-testid="stDecoration"], footer { display: none; }
.element-container { animation: none !important; transition: none !important; }
[data-testid="stMainBlockContainer"] { background: #F7F4EE !important; color: #1C1915 !important; flex: 0 0 auto !important; height: auto !important; min-height: 100% !important; width: 100%; max-width: 1180px; padding: 4.5rem 1.25rem 2.2rem; box-sizing: border-box; align-self: stretch; }
[data-testid="stMain"] [data-testid="stVerticalBlock"] { background-color: #F7F4EE; }
[data-testid="stMain"] { font-family: "Source Sans 3", "Segoe UI", sans-serif; }
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3 { color: #1C1915 !important; }
[data-testid="stCaptionContainer"], .stCaption, [data-testid="stCaptionContainer"] p { opacity: 1 !important; }
[data-testid="stMain"] [data-testid="stCaptionContainer"] p,
[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p.page-note,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p.section-note,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p.studio-line,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p.mist { color: #3F3A34 !important; opacity: 1 !important; }
[data-testid="stMain"] a { color: #2F6B4F !important; }
[data-testid="stMain"] [data-testid="stMetricLabel"] p,
[data-testid="stMain"] [data-testid="stMetricDelta"],
[data-testid="stMain"] [data-testid="stMetricDelta"] * { color: #3F3A34 !important; opacity: 1 !important; }
[data-testid="stMain"] [data-testid="stMetricValue"], [data-testid="stMain"] [data-testid="stMetricValue"] * { color: #1C1915 !important; font-variant-numeric: tabular-nums; opacity: 1 !important; }
[data-testid="stSidebar"], [data-testid="stSidebarContent"], [data-testid="stSidebarHeader"] { background: #122A22 !important; }
[data-testid="stSidebar"] { border-right: 1px solid #1C3A30; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1 { color: #F4F0E6; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #F4F0E6 !important; opacity: 1 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius: 2px !important; padding: 0.28rem 0.55rem; margin: 0 0 0.12rem; background: transparent !important; border: 0 !important; box-shadow: none !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background: #2F6B4F !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) * { color: #F7F4EE !important; font-weight: 600; }
.studio-mast { margin: 0 0 0.7rem; padding-bottom: 0.45rem; border-bottom: 1px solid #E4DDD2; }
.studio-name { font-size: 1.05rem; font-weight: 600; margin: 0; color: #1C1915 !important; }
.studio-line, .page-note, .section-note, .mist { color: #3F3A34 !important; font-size: 0.92rem; line-height: 1.45; margin: 0.12rem 0 0; }
.page-title { font-family: Georgia, "Times New Roman", serif; font-size: 1.35rem; font-weight: 600; line-height: 1.2; margin: 0.35rem 0 0.15rem; color: #1C1915 !important; }
.page-note { max-width: 68rem; margin-bottom: 0.7rem; }
.section-title { font-size: 1rem; font-weight: 600; margin: 0; color: #1C1915 !important; }
.desk-grid { display: grid; gap: 0.55rem; margin: 0.2rem 0 0.55rem; }
.desk-cell { background: #F7F4EE; border: 1px solid #E4DDD2; border-radius: 2px; padding: 0.5rem 0.7rem; box-shadow: none; }
.desk-cell.active, .price-tile.accent { border-top: 2px solid #2F6B4F; }
.desk-title { font-size: 0.98rem; font-weight: 600; color: #1C1915 !important; }
.desk-body { font-size: 0.88rem; color: #3F3A34 !important; }
.price-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.45rem; margin: 0.35rem 0; }
.price-tile { border-top: 1px solid #E4DDD2; padding-top: 0.28rem; min-width: 0; }
.price-label { font-size: 0.78rem; font-weight: 600; color: #3F3A34; }
.price-value { font-size: 1.2rem; font-weight: 600; font-variant-numeric: tabular-nums; color: #1C1915; }
.price-tile.accent .price-value { color: #2F6B4F !important; }
.range-track { height: 4px; background: #E4DDD2; position: relative; margin: 0.35rem 0; }
.range-fill { height: 100%; background: #C9D5CC; }
.range-mark { position: absolute; top: 50%; width: 8px; height: 8px; border-radius: 50%; background: #2F6B4F; transform: translate(-50%, -50%); }
.fact-row { display: flex; justify-content: space-between; align-items: baseline; gap: 0.75rem; padding: 0.32rem 0; border-bottom: 1px solid #E4DDD2; font-size: 0.92rem; }
.fact-row span { color: #3F3A34; min-width: 0; }
.fact-row b { font-weight: 600; color: #1C1915; font-variant-numeric: tabular-nums; text-align: right; flex: 0 1 auto; max-width: 62%; }
.fact-row b.ready { color: #2F6B4F !important; }
[data-testid="stVerticalBlockBorderWrapper"] { background: #F7F4EE !important; color: #1C1915 !important; border-color: #E4DDD2 !important; box-shadow: none !important; }
[data-testid="stDataFrame"], [data-testid="stExpander"], [data-testid="stMetric"] { box-shadow: none !important; }
div[data-testid="stAlert"] { background: #EFEBE1 !important; color: #1C1915 !important; border: 0 !important; border-left: 3px solid #2F6B4F !important; border-radius: 0 !important; box-shadow: none !important; }
[data-testid="stMain"] [data-baseweb="select"] > div, [data-testid="stMain"] input { background: #F7F4EE !important; color: #1C1915 !important; border-color: #D9D1C5 !important; }
[data-testid="stSelectboxVirtualDropdown"],
[data-testid="stMultiSelectDropdown"],
[data-baseweb="menu"],
[data-baseweb="popover"] [role="listbox"] { background: #EFEBE1 !important; color: #1C1915 !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"],
[data-testid="stSelectboxVirtualDropdown"] [data-item-hl],
[data-testid="stMultiSelectDropdown"] [role="option"],
[data-testid="stMultiSelectDropdown"] [data-item-hl],
[data-baseweb="menu"] li,
[data-baseweb="popover"] [role="option"] { background: transparent !important; color: #1C1915 !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-hovered],
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused],
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
[data-testid="stMultiSelectDropdown"] [role="option"][data-hovered],
[data-testid="stMultiSelectDropdown"] [role="option"][data-focused],
[data-testid="stMultiSelectDropdown"] [role="option"][aria-selected="true"],
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] li[aria-selected="true"],
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"] { background: #2F6B4F !important; color: #F7F4EE !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-hovered] [data-item-hl],
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused] [data-item-hl],
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] [data-item-hl],
[data-testid="stMultiSelectDropdown"] [role="option"][data-hovered] [data-item-hl],
[data-testid="stMultiSelectDropdown"] [role="option"][data-focused] [data-item-hl],
[data-testid="stMultiSelectDropdown"] [role="option"][aria-selected="true"] [data-item-hl] { background: #2F6B4F !important; color: #F7F4EE !important; }
[data-testid="stDataFrameColumnMenu"],
[data-testid="stDataFrameColumnVisibilityMenu"],
[data-testid="stDataFrameStatisticsMenu"],
[data-testid="stDataFrameColumnFormattingMenu"],
[data-testid="stDataFrameButtonActionMenu"],
[data-testid="stDataFrameTooltipContent"],
[role="tooltip"]:has([data-testid="stDataFrameTooltipContent"]) { background: #EFEBE1 !important; color: #1C1915 !important; }
[data-testid="stDataFrameColumnMenu"] [role="menu"],
[data-testid="stDataFrameColumnVisibilityMenu"] [role="menu"],
[data-testid="stDataFrameColumnFormattingMenu"] [role="menu"],
[data-testid="stDataFrameButtonActionMenu"] [role="menu"] { background: transparent !important; color: #1C1915 !important; }
[data-testid="stDataFrameColumnMenu"] [role="menuitem"],
[data-testid="stDataFrameColumnVisibilityMenu"] [role="menuitem"],
[data-testid="stDataFrameColumnFormattingMenu"] [role="menuitem"],
[data-testid="stDataFrameStatisticsMenu"] [role="menuitem"],
[data-testid="stDataFrameButtonActionMenu"] [role="menuitem"],
[data-testid="stDataFrameColumnVisibilityMenu"] label { background: transparent !important; color: #1C1915 !important; }
[data-testid="stDataFrameColumnMenu"] [role="menuitem"]:hover,
[data-testid="stDataFrameColumnMenu"] [role="menuitem"]:focus,
[data-testid="stDataFrameColumnMenu"] [role="menuitem"]:focus-visible,
[data-testid="stDataFrameColumnMenu"] [role="menuitem"][aria-selected="true"],
[data-testid="stDataFrameColumnMenu"] [role="menuitem"][data-highlighted="true"],
[data-testid="stDataFrameColumnVisibilityMenu"] [role="menuitem"]:hover,
[data-testid="stDataFrameColumnVisibilityMenu"] [role="menuitem"]:focus,
[data-testid="stDataFrameColumnVisibilityMenu"] label:hover,
[data-testid="stDataFrameColumnFormattingMenu"] [role="menuitem"]:hover,
[data-testid="stDataFrameColumnFormattingMenu"] [role="menuitem"]:focus,
[data-testid="stDataFrameStatisticsMenu"] [role="menuitem"]:hover,
[data-testid="stDataFrameButtonActionMenu"] [role="menuitem"]:hover,
[data-testid="stDataFrameButtonActionMenu"] [role="menuitem"]:focus { background: #2F6B4F !important; color: #F7F4EE !important; }
[data-testid="stDataFrameColumnMenu"] [role="menuitem"]:hover *,
[data-testid="stDataFrameColumnMenu"] [role="menuitem"]:focus *,
[data-testid="stDataFrameColumnMenu"] [role="menuitem"]:focus-visible *,
[data-testid="stDataFrameColumnMenu"] [role="menuitem"][aria-selected="true"] *,
[data-testid="stDataFrameColumnMenu"] [role="menuitem"][data-highlighted="true"] *,
[data-testid="stDataFrameColumnVisibilityMenu"] [role="menuitem"]:hover *,
[data-testid="stDataFrameColumnVisibilityMenu"] label:hover *,
[data-testid="stDataFrameColumnFormattingMenu"] [role="menuitem"]:hover *,
[data-testid="stDataFrameStatisticsMenu"] [role="menuitem"]:hover *,
[data-testid="stDataFrameButtonActionMenu"] [role="menuitem"]:hover * { color: #F7F4EE !important; background: transparent !important; }
[data-testid="stMain"] button { border-radius: 2px !important; box-shadow: none !important; height: auto; white-space: normal; }
[data-testid="stMain"] button:not([data-testid="stBaseButton-elementToolbar"]) { min-height: 2.4rem; }
[data-testid="stMain"] button p { white-space: normal; overflow: visible; line-height: 1.25; }
[data-testid="stElementToolbar"] { background: transparent !important; }
[data-testid="stElementToolbarButtonContainer"] { background: #F7F4EE !important; color: #1C1915 !important; border: 1px solid #E4DDD2 !important; box-shadow: none !important; }
[data-testid="stElementToolbar"] button,
button[data-testid="stBaseButton-elementToolbar"] { background: #F7F4EE !important; color: #1C1915 !important; border: 1px solid transparent !important; min-height: 0 !important; height: auto !important; box-shadow: none !important; }
[data-testid="stElementToolbar"] button:hover,
[data-testid="stElementToolbar"] button:focus,
[data-testid="stElementToolbar"] button:focus-visible,
[data-testid="stElementToolbar"] button:active,
button[data-testid="stBaseButton-elementToolbar"]:hover,
button[data-testid="stBaseButton-elementToolbar"]:focus,
button[data-testid="stBaseButton-elementToolbar"]:focus-visible,
button[data-testid="stBaseButton-elementToolbar"]:active { background: #EFEBE1 !important; color: #2F6B4F !important; border-color: #C9C0B2 !important; }
[data-testid="stElementToolbar"] svg,
[data-testid="stElementToolbarButtonIcon"] { color: inherit !important; fill: currentColor !important; }
[data-testid="stFullScreenFrame"] { background: #F7F4EE !important; color: #1C1915 !important; }
[data-testid="stMain"] button[data-testid="stBaseButton-secondary"] { background: #F7F4EE !important; border: 1px solid #C9C0B2 !important; color: #1C1915 !important; }
[data-testid="stMain"] button[data-testid="stBaseButton-secondary"] p { color: #1C1915 !important; }
[data-testid="stMain"] button[data-testid="stBaseButton-primary"] { background: #2F6B4F !important; border: 1px solid #2F6B4F !important; color: #F7F4EE !important; }
[data-testid="stMain"] button[data-testid="stBaseButton-primary"] p,
[data-testid="stMain"] button[data-testid="stBaseButton-primary"] span { color: #F7F4EE !important; }
[data-testid="stMain"] [data-testid="stSliderThumbValue"] [data-testid="stMarkdownContainer"],
[data-testid="stMain"] [data-testid="stSliderThumbValue"] [data-testid="stMarkdownContainer"] p { color: #F7F4EE !important; }
[data-testid="stImage"] { max-width: 100%; }
[data-testid="stImage"] img { width: 100%; height: auto; }
@media (max-width: 820px) { .price-row, .desk-grid { grid-template-columns: 1fr !important; } }
"""


def current_theme() -> str:
    return "light"


def inject() -> None:
    st.markdown(f"<style>{DESK_CSS}</style>", unsafe_allow_html=True)


def page_header(kicker_text: str, title: str, blurb: str) -> None:
    del kicker_text
    st.markdown(
        f'<p class="page-title">{escape(title)}</p><p class="page-note">{escape(blurb)}</p>',
        unsafe_allow_html=True,
    )


def section_title(label: str, note: str = "") -> None:
    body = f'<p class="section-title">{escape(label)}</p>'
    if note:
        body += f'<p class="section-note">{escape(note)}</p>'
    st.markdown(body, unsafe_allow_html=True)


def card_grid(items: list[dict[str, Any]], columns: int = 4) -> None:
    cards = []
    for item in items:
        active = " active" if item.get("active") else ""
        body = f'<div class="desk-body">{item.get("body") or ""}</div>' if item.get("body") else ""
        rows = "".join(
            f'<div class="fact-row"><span>{escape(str(label))}</span><b>{escape(str(value))}</b></div>'
            for label, value in (item.get("rows") or [])
        )
        note = f'<p class="mist">{escape(str(item["note"]))}</p>' if item.get("note") else ""
        cards.append(
            f'<div class="desk-cell{active}"><div class="desk-title">{escape(str(item.get("title", "")))}</div>{body}{rows}{note}</div>'
        )
    st.markdown(
        f'<div class="desk-grid" style="grid-template-columns:repeat({columns},1fr)">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def fact(label: str, value: str, ready: bool = False) -> None:
    facts([(label, value)], ready=ready)


def facts(rows: list[tuple[str, str]], ready: bool = False) -> None:
    klass = "ready" if ready else ""
    body = "".join(
        f'<div class="fact-row"><span>{escape(str(label))}</span><b class="{klass}">{escape(str(value))}</b></div>'
        for label, value in rows
    )
    st.markdown(body, unsafe_allow_html=True)


def _b4_caption() -> str:
    try:
        from lib.scores import b4_coverage_sentence

        return b4_coverage_sentence()
    except Exception:
        return "The intended interval level is 90%."


def range_bar(low: float, mid: float, high: float) -> None:
    span = max(high - low, 1)
    pos = min(100.0, max(0.0, 100.0 * (mid - low) / span))
    st.markdown(
        f'<div class="range-track"><div class="range-fill"></div>'
        f'<div class="range-mark" style="left:{pos:.1f}%"></div></div>'
        f'<p class="mist">This interval is the runtime model, not Experiment B and not Experiment C. {escape(_b4_caption())}</p>',
        unsafe_allow_html=True,
    )


def chart_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        font=dict(family=CHART_FONT, color=SLATE, size=12),
        margin=dict(l=48, r=16, t=12, b=72),
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0, font=dict(color=SLATE, size=12)),
        hovermode="x unified",
        uirevision="htfe-studio",
    )
    axis = dict(gridcolor=LINE, zeroline=False, linecolor=LINE, automargin=True, tickfont=dict(color=SLATE, size=12), title_font=dict(color=SLATE, size=12))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def show_chart(fig: go.Figure, height: int = 360) -> None:
    st.plotly_chart(chart_layout(fig, height), width="stretch", config=CHART_CONFIG, theme=None)


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
