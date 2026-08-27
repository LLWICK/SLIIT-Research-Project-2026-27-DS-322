"""Theme helpers. Cards use native Streamlit so the layout cannot split."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

HARVEST = "#c6f31a"
MIST = "#9bb0a6"
CREAM = "#f4f7f2"
LINE = "#1d3a30"
INK = "#07110e"
INK2 = "#0c1a16"
PANEL = "#10211b"
SKY = "#7dd3c0"
AMBER = "#f5b942"

CHART_CONFIG = {"displayModeBar": False, "responsive": True}


def inject() -> None:
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Outfit:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: "Outfit", ui-sans-serif, system-ui, sans-serif; }
    h1, h2, h3, [data-testid="stSidebar"] h1 {
        font-family: "Fraunces", ui-serif, Georgia, serif !important;
        letter-spacing: -0.02em;
    }
    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(198,243,26,0.08), transparent 50%),
            radial-gradient(900px 500px at 110% 10%, rgba(245,185,66,0.07), transparent 45%),
            #07110e;
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(234,243,236,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(234,243,236,0.035) 1px, transparent 1px);
        background-size: 28px 28px;
        z-index: 0;
    }
    #MainMenu, footer, [data-testid="stDecoration"],
    [data-testid="stAppDeployButton"] {
        visibility: hidden;
        height: 0;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 1.4rem;
        padding-bottom: 2.6rem;
        max-width: 1180px;
    }
    [data-testid="stSidebar"] {
        background: rgba(12, 25, 20, 0.92);
        border-right: 1px solid #1d3a30;
    }
    [data-testid="stVerticalBlock"] { gap: 0.75rem !important; }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 0.95rem 1rem 1.05rem;
        background: #10211b;
        border-color: #1d3a30 !important;
    }
    [data-testid="stVerticalBlock"]:has(#dash-trio) [data-testid="stHorizontalBlock"]:has([data-testid="stSlider"]) {
        align-items: stretch !important;
        gap: 0.85rem !important;
    }
    [data-testid="stVerticalBlock"]:has(#dash-trio) [data-testid="stHorizontalBlock"]:has([data-testid="stSlider"]) > div {
        display: flex !important;
        flex-direction: column !important;
        background: linear-gradient(180deg, rgba(198,243,26,0.04), transparent 36%), #10211b;
        border: 1px solid #1d3a30;
        border-radius: 20px;
        padding: 1.05rem 1.1rem 1.15rem;
        min-height: 640px;
        box-sizing: border-box;
    }
    .slider-ends {
        display: flex;
        justify-content: space-between;
        color: #8aa396;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        margin-top: -0.25rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        align-items: center;
        border: 0;
        border-radius: 12px;
        padding: 0.45rem 0.75rem !important;
        margin-bottom: 0.12rem;
        color: rgba(234,243,236,0.8);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: #c6f31a;
        color: #07110e;
        font-weight: 600;
    }
    [data-testid="stSlider"] [role="slider"] { background-color: #c6f31a !important; }
    .stCaption, [data-testid="stCaption"] { letter-spacing: 0.04em; color: #8aa396; }
    .stage-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.65rem;
        margin: 0.2rem 0 0.35rem;
    }
    .stage-card {
        border: 1px solid #1d3a30;
        background: #10211b;
        border-radius: 16px;
        padding: 0.85rem 0.9rem 1rem;
    }
    .stage-card.active { border-color: #c6f31a; }
    .stage-role {
        color: #8aa396;
        font-size: 0.7rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .stage-title {
        font-family: "Fraunces", ui-serif, Georgia, serif;
        font-size: 1.15rem;
        margin: 0.28rem 0 0.2rem;
        color: #eaf3ec;
    }
    .stage-body { color: #8aa396; font-size: 0.88rem; }
    .price-row {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0.55rem;
        margin: 0.7rem 0 0.75rem;
    }
    .price-tile {
        border: 1px solid #1d3a30;
        background: #0c1914;
        border-radius: 16px;
        padding: 0.75rem 0.7rem 0.85rem;
    }
    .price-tile.accent {
        border-color: #c6f31a;
        background: rgba(198,243,26,0.10);
    }
    .price-label {
        color: #8aa396;
        font-size: 0.68rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .price-value {
        font-family: "Fraunces", ui-serif, Georgia, serif;
        font-size: 1.35rem;
        font-variant-numeric: tabular-nums;
        color: #eaf3ec;
        line-height: 1.15;
    }
    .price-tile.accent .price-value { color: #c6f31a; }
    .range-track {
        height: 7px;
        border-radius: 99px;
        background: #07110e;
        border: 1px solid #1d3a30;
        position: relative;
        margin: 0.35rem 0 0.45rem;
    }
    .range-fill {
        height: 100%;
        border-radius: 99px;
        background: rgba(198,243,26,0.35);
        width: 100%;
    }
    .range-mark {
        position: absolute;
        top: 50%;
        width: 11px;
        height: 11px;
        border-radius: 50%;
        background: #c6f31a;
        transform: translate(-50%, -50%);
    }
    .fact-row {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.38rem 0;
        border-bottom: 1px solid #1d3a30;
        font-size: 0.92rem;
    }
    .fact-row span { color: #8aa396; }
    .fact-row b { color: #eaf3ec; font-weight: 500; }
    .fact-row b.ready { color: #c6f31a; }
    .mist { color: #8aa396; font-size: 0.9rem; line-height: 1.45; }
</style>
        """,
        unsafe_allow_html=True,
    )
    components.html(
        """
<script>
(function () {
  const win = window.parent;
  const doc = win.document;
  try {
    Object.keys(win.localStorage).forEach(function (key) {
      if (/sidebar/i.test(key)) win.localStorage.removeItem(key);
    });
  } catch (err) {}
  function expand() {
    const nodes = doc.querySelectorAll(
      '[data-testid="stSidebarCollapsedControl"] button, [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] button, [data-testid="collapsedControl"]'
    );
    if (!nodes.length) return;
    nodes[0].click();
  }
  expand();
  [150, 400, 900].forEach(function (ms) { setTimeout(expand, ms); });
})();
</script>
        """,
        height=0,
        width=0,
    )


def page_header(kicker_text: str, title: str, blurb: str) -> None:
    st.caption(kicker_text.upper())
    st.title(title)
    st.write(blurb)


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
    cols = st.columns(columns, gap="small")
    for col, item in zip(cols, items):
        with col:
            with st.container(border=True):
                if item.get("role"):
                    st.caption(str(item["role"]).upper())
                st.markdown(f"**{item.get('title', '')}**")
                body = item.get("body") or ""
                if body:
                    st.caption(str(body).replace("<br>", " | "))
                for label, value in item.get("rows") or []:
                    st.markdown(f"{label} · **{value}**")
                if item.get("note"):
                    st.caption(item["note"])


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
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,14,0.35)",
        font=dict(color=MIST, size=12),
        margin=dict(l=40, r=16, t=18, b=78),
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),
        hovermode="x unified",
        uirevision="htfe-studio",
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
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
