"""
KPI Metric Cards and Executive Header Widgets.
"""

from typing import Dict, Any, Optional
import streamlit as st


def render_kpi_card_html(
    title: str,
    value: str,
    sub_text: str,
    border_color: str = "#1E5631",
    badge_text: Optional[str] = None,
    badge_class: str = "badge-real",
) -> str:
    """
    Generate HTML snippet for a modern executive metric card with provenance tag.

    Args:
        title: Uppercase metric label.
        value: Main numeric value or status.
        sub_text: Subtitle or delta indicator.
        border_color: Left border accent color.
        badge_text: Optional text for data provenance badge.
        badge_class: CSS class for badge styling.

    Returns:
        HTML string.
    """
    badge_html = (
        f'<span class="{badge_class}" style="font-size: 0.68rem; padding: 0.15rem 0.45rem; flex-shrink: 0;">{badge_text}</span>'
        if badge_text
        else ""
    )

    return f"""
    <div class="metric-card" style="border-left-color: {border_color};">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 4px; margin-bottom: 0.2rem;">
            <div class="metric-title" style="flex-grow: 1; margin-bottom: 0;">{title}</div>
            {badge_html}
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub_text}</div>
    </div>
    """


def render_executive_overview_cards(
    latest_price: float,
    current_msrs: float,
    msrs_tier: str,
    msrs_color: str,
    dcvs_score: float,
    dcvs_tier: str,
    dcvs_color: str,
    dri_score: float,
    dri_tier: str,
    dri_color: str,
    location_label: str = "National Aggregate",
    dri_provenance: str = "VERIFIED AUDIT LEDGER",
    committed_ha: Optional[float] = None,
) -> None:
    """
    Render the 4 executive overview KPI cards across 4 Streamlit columns.
    Ensures 100% numerical consistency with the active spatial filter.
    """
    col1, col2, col3, col4 = st.columns(4)

    is_national = "National" in location_label
    msrs_title = f"Saturation MSRS ({location_label})"
    msrs_sub = f"Status: {msrs_tier}" + (f" ({committed_ha:.0f} ha)" if committed_ha is not None else "")

    with col1:
        st.markdown(
            render_kpi_card_html(
                title=f"Market Price ({location_label})",
                value=f"Rs. {latest_price:.1f} / kg",
                sub_text="Latest weekly quoted price",
                border_color="#1E5631",
                badge_text="REAL HARTI/CBSL",
                badge_class="badge-real",
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            render_kpi_card_html(
                title=msrs_title,
                value=f"{current_msrs:.2f}",
                sub_text=msrs_sub,
                border_color=msrs_color,
                badge_text="SYNTHETIC PROXY",
                badge_class="badge-proxy",
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            render_kpi_card_html(
                title=f"Viability DCVS ({location_label})",
                value=f"{dcvs_score:.1f}%",
                sub_text=f"Viability: {dcvs_tier}",
                border_color=dcvs_color,
                badge_text="EVIDENCE SYNTHESIS",
                badge_class="badge-proxy",
            ),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            render_kpi_card_html(
                title="Source Reliability (DRI)",
                value=f"{dri_score:.2f}%",
                sub_text=f"HARTI Benchmark ({dri_tier} Tier)",
                border_color=dri_color,
                badge_text="HARTI AUDIT BENCHMARK",
                badge_class="badge-real",
            ),
            unsafe_allow_html=True,
        )
