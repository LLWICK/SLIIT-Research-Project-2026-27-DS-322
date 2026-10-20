"""
Actionable Farmer Advisory Presentation Components.
Presents transparent, evidence-based recommendations, structured rationale,
and alternative crop options with dark forest-green styling.
"""

from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
from engine.advisor import Recommendation


def render_advisory_card(rec: Recommendation) -> None:
    """
    Render executive recommendation banner and transparent evidence breakdown.
    No scattered badges; standardized 0–100 scales with 1 decimal place.
    """
    # 1. Actionable Guidance Banner
    st.markdown(
        f'<div class="{rec.banner_class}">'
        f'<div style="font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; color: #A3B899; margin-bottom: 0.25rem;">'
        f'Operational Guidance — {rec.district} ({rec.season_label})'
        f'</div>'
        f'<div style="font-size: 1.25rem; font-weight: 800; color: {rec.action_color}; margin-bottom: 0.35rem;">'
        f'{rec.action}'
        f'</div>'
        f'<div style="font-size: 0.95rem; color: #E8EFEA; font-weight: 500;">'
        f'{rec.headline}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 2. Transparent Evidence Rationale (Collapsible Expander)
    with st.expander("🔍 Transparent Evidence Rationale & Indicator Breakdown", expanded=True):
        st.markdown("#### Indicator Evaluation Summary")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                label="Market Saturation (MSRS)",
                value=f"{rec.msrs_score:.1f}",
                delta=rec.msrs_tier,
                delta_color="inverse",
            )
        with m2:
            st.metric(
                label="Crop Viability (DCVS)",
                value=f"{rec.dcvs_score:.1f}",
                delta=rec.dcvs_tier,
                delta_color="normal",
            )
        with m3:
            st.metric(
                label="Data Reliability (DRI)",
                value=f"{rec.dri_score:.1f}",
                delta=rec.dri_tier,
                delta_color="normal",
            )
        with m4:
            st.metric(
                label="Advisory Confidence",
                value=rec.confidence.split("—")[0].strip(),
                delta=rec.confidence,
                delta_color="off",
            )

        st.markdown("---")
        st.markdown("#### Comprehensive Decision Rationale")
        for pt in rec.rationale_points:
            st.markdown(f"- {pt}")

    # 3. Alternative Crop Recommendations (Displayed only when crop is High Risk / Diversify)
    if rec.suggested_alternatives:
        st.markdown("#### Recommended Diversification Alternatives")
        st.caption(
            "Evidence-based crop routing based on regional saturation (MSRS) and dynamic viability (DCVS):"
        )

        cols = st.columns(len(rec.suggested_alternatives), gap="medium")
        for i, alt in enumerate(rec.suggested_alternatives):
            status_color = "#2E7D32" if alt["msrs"] <= 20.0 else ("#D97706" if alt["msrs"] <= 60.0 else "#C53030")
            expl_text = alt.get("explanation", "Alternative crop candidate.")
            with cols[i]:
                # Completely dark-green themed alternative cards (no harsh white surfaces)
                st.markdown(
                    f'<div style="background: #112417; border: 1px solid #1E3D27; border-radius: 8px; padding: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.35); height: 100%; box-sizing: border-box; display: flex; flex-direction: column; justify-content: space-between;">'
                    f'<div>'
                    f'<div style="font-weight: 800; font-size: 1.05rem; color: #4ADE80; margin-bottom: 4px;">{alt["crop"]}</div>'
                    f'<div style="font-size: 0.82rem; color: {status_color}; font-weight: 700; margin-bottom: 2px;">MSRS: {alt["msrs"]:.1f} — {alt["status"]}</div>'
                    f'<div style="font-size: 0.82rem; color: #E8EFEA; font-weight: 600; margin-bottom: 2px;">Viability (DCVS): <strong>{alt["dcvs"]:.1f}</strong></div>'
                    f'<div style="font-size: 0.80rem; color: #A3B899; margin-bottom: 6px;">Quoted Price: Rs. {alt["current_price"]:.1f}/kg</div>'
                    f'</div>'
                    f'<div style="font-size: 0.76rem; color: #C8D8CC; border-top: 1px dashed #1E3D27; padding-top: 6px; margin-top: 4px;">{expl_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def render_crop_recommendations_table(df_recs: pd.DataFrame) -> None:
    """
    Render clear, sortable recommendation table:
    Crop | MSRS | DCVS | DRI | Confidence | Recommendation | Key Reason
    Formatted to 1 decimal place with no applymap deprecation warnings.
    """
    st.markdown("### Multi-Crop Planning Recommendations")
    st.caption("Comparison across all 5 research target crops for the selected planning scope.")

    # Color styling function for Recommendation column
    def color_rec(val):
        if val == "Recommended":
            return "color: #4ADE80; font-weight: bold;"
        elif val == "Consider":
            return "color: #FBBF24; font-weight: bold;"
        elif val == "High Risk/Avoid":
            return "color: #F87171; font-weight: bold;"
        return ""

    # Explicit format string for numeric columns to prevent unrounded numbers (e.g., 48.600000)
    format_dict = {}
    for col in ["MSRS", "DCVS", "DRI"]:
        if col in df_recs.columns:
            format_dict[col] = "{:.1f}"

    styled_df = df_recs.style.format(format_dict)

    # Use Styler.map if available (Pandas >= 2.1), fallback to Styler.applymap (Pandas < 2.1)
    map_func = getattr(styled_df, "map", getattr(styled_df, "applymap", None))
    if map_func:
        styled_df = map_func(color_rec, subset=["Recommendation"])

    st.dataframe(styled_df, hide_index=True, use_container_width=True)


def render_district_drilldown_panel(summary: Dict[str, Any]) -> None:
    """
    Render District Drill-Down Panel showing:
    - Price trends & moving averages
    - Committed vs. target hectares
    - Agro-weather summary
    - DRI component breakdown
    """
    st.markdown(f"### District Drill-Down: {summary['district']} ({summary['crop']})")
    st.caption(f"Detailed agronomic and market parameters for {summary['season']}.")

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("Latest Quoted Price", f"Rs. {summary['latest_price']:.1f}/kg", delta=f"8W MA: Rs. {summary['eight_week_ma']:.1f}")
    c2.metric("Cultivation Quota", f"{summary['committed_ha']:.1f} ha", delta=f"{summary['overshoot_pct']:+.1f}% vs Target ({summary['target_ha']:.1f} ha)")
    c3.metric("Rainfall Accumulation", f"{summary['rainfall_avg_mm']:.1f} mm/day", f"Temp: {summary['temp_avg_c']:.1f}°C")
    c4.metric("Data Reliability (DRI)", f"{summary['dri_score']:.1f}", summary['confidence'])
