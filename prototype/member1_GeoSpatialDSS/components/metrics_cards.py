"""
KPI Metric Cards and Executive Header Widgets.
Presents 5 standardized overview KPI cards without scattered badges:
  1. Market Price
  2. Saturation (MSRS)
  3. Viability (DCVS)
  4. Reliability (DRI)
  5. Confidence
Standardized to 0.0 - 100.0 scale with 1 decimal place.
"""

from typing import Dict, Any, Optional
import numpy as np
import streamlit as st


def render_kpi_card_html(
    title: str,
    value: str,
    sub_text: str,
    scope_text: str = "",
    border_color: str = "#1E5631",
    badge_text: Optional[str] = None,
    badge_class: str = "badge-real",
) -> str:
    """
    Generate clean, unindented HTML snippet for an executive metric card.
    Guarantees zero leading indentation so CommonMark never treats it as a code block.
    """
    badge_html = (
        f'<span class="{badge_class}" style="font-size: 0.68rem; padding: 0.15rem 0.45rem; flex-shrink: 0;">{badge_text}</span>'
        if badge_text
        else ""
    )
    scope_html = (
        f'<div style="font-size: 0.70rem; color: #A3B899; font-weight: 600; margin-top: 2px;">Scope: {scope_text}</div>'
        if scope_text
        else ""
    )

    # Sanitize value display (fallback to 'Unavailable' if None or NaN)
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip().lower() in ("", "none", "nan"):
        display_val = "Unavailable"
    else:
        display_val = str(value)

    # Completely unindented HTML to prevent Streamlit/Markdown code block escaping
    return (
        f'<div class="metric-card" style="border-left-color: {border_color};">'
        f'<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 4px; margin-bottom: 0.2rem;">'
        f'<div class="metric-title" style="flex-grow: 1; margin-bottom: 0;">{title}</div>'
        f'{badge_html}'
        f'</div>'
        f'<div class="metric-value">{display_val}</div>'
        f'<div class="metric-sub">{sub_text}</div>'
        f'{scope_html}'
        f'</div>'
    )


def render_executive_overview_cards(
    latest_price: Optional[float],
    current_msrs: Optional[float],
    msrs_tier: str,
    msrs_color: str,
    dcvs_score: Optional[float],
    dcvs_tier: str,
    dcvs_color: str,
    dri_score: Optional[float],
    dri_tier: str,
    dri_color: str,
    confidence_label: str = "High confidence",
    confidence_color: str = "#2E7D32",
    location_label: str = "National Aggregate",
    committed_ha: Optional[float] = None,
    target_ha: Optional[float] = None,
    # Optional backward compatibility arguments
    dri_provenance: Optional[str] = None,
) -> None:
    """
    Render 5 executive overview KPI cards across 5 Streamlit columns:
      1. Market Price: Rs. {latest_price:.1f}/kg or Unavailable
      2. Saturation (MSRS): {current_msrs:.1f} (0-100 scale)
      3. Viability (DCVS): {dcvs_score:.1f} (0-100 scale)
      4. Reliability (DRI): {dri_score:.1f} (0-100 scale)
      5. Advisory Confidence: {confidence_label}
    Displays value, qualitative status, and selected scope.
    """
    # Safe value formatting with 'Unavailable' state
    val_price = (
        f"Rs. {latest_price:.1f}/kg"
        if (latest_price is not None and not np.isnan(latest_price))
        else "Unavailable"
    )
    val_msrs = (
        f"{current_msrs:.1f}"
        if (current_msrs is not None and not np.isnan(current_msrs))
        else "Unavailable"
    )
    val_dcvs = (
        f"{dcvs_score:.1f}"
        if (dcvs_score is not None and not np.isnan(dcvs_score))
        else "Unavailable"
    )
    val_dri = (
        f"{dri_score:.1f}"
        if (dri_score is not None and not np.isnan(dri_score))
        else "Unavailable"
    )
    val_conf = (
        confidence_label.split("—")[0].strip()
        if confidence_label
        else "Unavailable"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            render_kpi_card_html(
                title="Market Price",
                value=val_price,
                sub_text="Latest weekly quote",
                scope_text=location_label,
                border_color="#1E5631",
            ),
            unsafe_allow_html=True,
        )

    with col2:
        msrs_sub = f"Status: {msrs_tier}" + (
            f" ({committed_ha:.0f} ha)" if (committed_ha is not None and not np.isnan(committed_ha)) else ""
        )
        st.markdown(
            render_kpi_card_html(
                title="Market Saturation (MSRS)",
                value=val_msrs,
                sub_text=msrs_sub,
                scope_text=location_label,
                border_color=msrs_color or "#1E5631",
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            render_kpi_card_html(
                title="Crop Viability (DCVS)",
                value=val_dcvs,
                sub_text=f"Viability: {dcvs_tier}",
                scope_text=location_label,
                border_color=dcvs_color or "#1E5631",
            ),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            render_kpi_card_html(
                title="Data Reliability (DRI)",
                value=val_dri,
                sub_text=f"Health: {dri_tier} Reliability",
                scope_text=location_label,
                border_color=dri_color or "#1E5631",
            ),
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            render_kpi_card_html(
                title="Confidence",
                value=val_conf,
                sub_text=confidence_label,
                scope_text=location_label,
                border_color=confidence_color or "#1E5631",
            ),
            unsafe_allow_html=True,
        )
