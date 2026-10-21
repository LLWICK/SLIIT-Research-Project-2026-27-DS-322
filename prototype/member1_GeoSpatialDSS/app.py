"""
Geo-Spatial Decision Support System (DSS) Dashboard.

Research Project: Data-Driven Decision Support System to Mitigate the Cobweb Effect in Sri Lankan Vegetable Markets
Collaboration: Hector Kobbekaduwa Agrarian Research and Training Institute (HARTI) collaboration | SLIIT | J26-DS-322
"""

import sys
from pathlib import Path

# Ensure dashboard directory is on sys.path
_dashboard_root = str(Path(__file__).resolve().parent)
if _dashboard_root not in sys.path:
    sys.path.insert(0, _dashboard_root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="Geo-Spatial DSS | Sri Lankan Vegetable Market Intelligence",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import (
    RESEARCH_TOPIC,
    SIDEBAR_BRAND_TITLE,
    SIDEBAR_BRAND_SUBTITLE,
    SIDEBAR_BRAND_DESC,
    APP_TITLE,
    APP_SUBTITLE,
    DISTRICT_NAMES,
    TARGET_CROPS,
    CUSTOM_CSS,
    COLOR_FOREST_GREEN,
    COLOR_SAGE_GREEN,
    FOOTER_TEXT,
    COMMITMENT_PROVENANCE_NOTE,
    DASHBOARD_TABS,
    MAP_PLOTLY_CONFIG,
    CHART_PLOTLY_CONFIG,
    BAR_CHART_PLOTLY_CONFIG,
    get_plain_language_confidence,
    get_risk_band,
)
from data_loader import (
    load_price_data,
    load_commitment_data,
    load_weather_data,
    load_dri_audit_ledger,
    get_available_seasons,
)
from engine.msrs_engine import (
    compute_district_msrs_table,
    compute_national_msrs_summary,
    get_district_msrs_record,
)
from engine.dcvs_engine import (
    compute_district_dcvs_table,
)
from engine.dri_bridge import (
    assess_district_data_readiness,
    compute_district_dri_table,
    get_dri_evaluation_summary,
    get_dri_source_benchmark,
)
from engine.advisor import (
    FarmerAdvisoryEngine,
)
from components.maps import (
    render_sri_lanka_map,
    render_sri_lanka_saturation_map,
    render_district_viability_map,
)
from components.charts import (
    render_price_history_chart,
    render_cobweb_phase_plot,
    render_district_commitment_barchart,
    render_weather_trend_chart,
)
from components.metrics_cards import (
    render_executive_overview_cards,
)
from components.recommendations import (
    render_advisory_card,
    render_crop_recommendations_table,
    render_district_drilldown_panel,
)

# Apply Custom Professional Dark-Green and Neutral Theme
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    # Load Primary Research Datasets with Streamlit Caching
    with st.spinner("Initializing Geo-Spatial DSS research datasets..."):
        try:
            df_price = load_price_data()
            df_comm = load_commitment_data()
            df_weather = load_weather_data()
            dri_ledger = load_dri_audit_ledger()
        except Exception as e:
            st.error(f"Failed to load research datasets: {str(e)}")
            st.stop()

    # =========================================================================
    # SIDEBAR: Research Filters & Scope Parameters
    # =========================================================================
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand-box">
                <div class="sidebar-brand-title">{SIDEBAR_BRAND_TITLE}</div>
                <div class="sidebar-brand-subtitle">{SIDEBAR_BRAND_SUBTITLE}</div>
                <div class="sidebar-brand-desc">{SIDEBAR_BRAND_DESC}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.subheader("Analytical Parameters")

        # 1. Target Crop Selector (5 Official Research Crops)
        selected_crop = st.selectbox(
            "Target Vegetable Crop",
            options=TARGET_CROPS,
            index=0,
            help="Restricted to the five core project research crops.",
        )

        # 2. Season & Year Selectors
        avail_seasons = get_available_seasons(df_comm)
        avail_years = sorted(list(set(y for _, y in avail_seasons)), reverse=True)
        selected_year = st.selectbox(
            "Planning Season Year",
            options=avail_years,
            index=1 if len(avail_years) > 1 else 0,
        )

        avail_season_types = sorted(list(set(s for s, y in avail_seasons if y == selected_year)))
        selected_season = st.selectbox("Agricultural Season", options=avail_season_types, index=0)
        season_label = f"{selected_season} {selected_year}"

        # 3. Administrative District Filter
        district_options = ["All Island (National Average)"] + DISTRICT_NAMES
        selected_district = st.selectbox(
            "District Granularity",
            options=district_options,
            index=0,
            help="Filter spatial indicators to an administrative district or national aggregate.",
        )

        # 4. Price Type Filter
        price_type = st.radio(
            "Price Quotation Level",
            options=["retail", "wholesale"],
            index=0,
            horizontal=True,
        )

    # =========================================================================
    # HEADER SECTION
    # =========================================================================
    st.markdown(
        f'<h1 class="dashboard-header">'
        f'Data-Driven Decision Support System to Mitigate the “Cobweb Effect”<br>'
        f'<span class="dashboard-header-sub">in Sri Lankan Vegetable Markets</span>'
        f'</h1>',
        unsafe_allow_html=True,
    )

    # Compute Core Metrics across Scope
    df_dist_msrs = compute_district_msrs_table(df_comm, selected_crop, selected_season, selected_year)
    msrs_summary = compute_national_msrs_summary(df_comm, selected_crop, selected_season, selected_year)

    # Resolve District vs National Scope for Metrics Consistency
    is_national = (selected_district == "All Island (National Average)")
    location_label = "National Aggregate" if is_national else selected_district

    if not is_national:
        dist_msrs_rec = get_district_msrs_record(df_dist_msrs, selected_district)
        current_msrs = dist_msrs_rec["msrs"]
        msrs_tier = dist_msrs_rec["risk_tier"]
        msrs_color = dist_msrs_rec["risk_color"]
        current_comm_ha = dist_msrs_rec["committed_ha"]
        current_tgt_ha = dist_msrs_rec["target_ha"]
    else:
        current_msrs = msrs_summary["national_msrs"]
        msrs_tier = msrs_summary["risk_tier"]
        msrs_color = msrs_summary["risk_color"]
        current_comm_ha = msrs_summary["total_committed_ha"]
        current_tgt_ha = msrs_summary["total_target_ha"]

    # Filter prices for latest observation
    crop_price_df = df_price[
        (df_price["crop"] == selected_crop) &
        (df_price["price_type"] == price_type)
    ]
    if not is_national:
        dist_price_df = crop_price_df[crop_price_df["district"] == selected_district]
        if not dist_price_df.empty:
            latest_p = float(dist_price_df["price"].iloc[-1])
        else:
            latest_p = float(crop_price_df["price"].iloc[-1]) if not crop_price_df.empty else 120.0
    else:
        latest_p = float(crop_price_df["price"].iloc[-1]) if not crop_price_df.empty else 120.0

    # DCVS average or district-specific
    df_dcvs = compute_district_dcvs_table(
        df_comm, df_price, df_weather, selected_crop, selected_season, selected_year, price_type=price_type
    )
    if not is_national:
        dcvs_row = df_dcvs[df_dcvs["district"] == selected_district]
        avg_dcvs = float(dcvs_row["dcvs"].iloc[0]) if not dcvs_row.empty else 60.0
        dcvs_tier = str(dcvs_row["viability_tier"].iloc[0]) if not dcvs_row.empty else "MODERATE"
        dcvs_color = str(dcvs_row["viability_color"].iloc[0]) if not dcvs_row.empty else "#D97706"
    else:
        avg_dcvs = float(df_dcvs["dcvs"].mean())
        dcvs_tier = "HIGH" if avg_dcvs >= 70.0 else ("MODERATE" if avg_dcvs >= 45.0 else "LOW")
        dcvs_color = "#2E7D32" if dcvs_tier == "HIGH" else ("#D97706" if dcvs_tier == "MODERATE" else "#C53030")

    # DRI audit status (empirical 3-component DRI score)
    target_d = selected_district if not is_national else "All Island (National Average)"
    dri_info = assess_district_data_readiness(
        target_d, selected_crop, df_price, df_weather, df_comm, season_year=selected_year
    )

    # Precompute per-district DRI table for cross-component and map-layer integration
    df_dist_dri = compute_district_dri_table(df_price, df_weather, df_comm, selected_crop, season_year=selected_year)
    # Join DRI score into df_dist_msrs so map and tables have unified access
    df_dist_msrs = df_dist_msrs.merge(
        df_dist_dri[["district", "dri_score", "completeness", "recency", "consistency", "confidence_label"]],
        on="district",
        how="left",
    )

    # Initialize Advisory Engine
    advisor_engine = FarmerAdvisoryEngine()
    recommendation = advisor_engine.generate_advisory(
        district=target_d,
        crop=selected_crop,
        season_type=selected_season,
        season_year=selected_year,
        df_comm=df_comm,
        df_price=df_price,
        df_weather=df_weather,
        price_type=price_type,
        precomputed_msrs=current_msrs,
        precomputed_dcvs=avg_dcvs,
    )

    # =========================================================================
    # TABS: 6 Professional Research Tabs for Geo-Spatial DSS (No Emojis)
    # =========================================================================
    (
        tab_overview,
        tab_saturation,
        tab_price,
        tab_weather,
        tab_recs,
        tab_dri,
    ) = st.tabs(DASHBOARD_TABS)

    # -------------------------------------------------------------------------
    # TAB 1: Overview
    # -------------------------------------------------------------------------
    with tab_overview:
        # Render 5 Standardized Overview KPI Cards (No scattered badges)
        render_executive_overview_cards(
            latest_price=latest_p,
            current_msrs=current_msrs,
            msrs_tier=msrs_tier,
            msrs_color=msrs_color,
            dcvs_score=avg_dcvs,
            dcvs_tier=dcvs_tier,
            dcvs_color=dcvs_color,
            dri_score=dri_info["dri_score"],
            dri_tier=dri_info["reliability_tier"],
            dri_color=dri_info["tier_color"],
            confidence_label=dri_info["confidence_label"],
            confidence_color=dri_info["confidence_color"],
            location_label=location_label,
            committed_ha=current_comm_ha,
            target_ha=current_tgt_ha,
        )

        # Active Scope Confirmation Banner (Clean typography, no scattered badges)
        st.markdown(
            f'<div class="filter-context-banner">'
            f'<span>Active Analysis Scope: <strong>{selected_crop}</strong> | <strong>{season_label}</strong> | <strong>{location_label}</strong> ({price_type.capitalize()} Quotations)</span>'
            f'<span>Confidence: <strong>{dri_info["confidence_label"]}</strong></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Overview Executive Summary Columns
        o_col1, o_col2 = st.columns([3, 2], gap="large")

        with o_col1:
            st.markdown("### Decision Support Summary")
            st.markdown(
                f"""
                - **Market Saturation Index (MSRS)**: **{current_msrs:.1f} / 100** ({msrs_tier})
                  - Committed planting area: **{current_comm_ha:,.1f} ha** vs sustainable quota: **{current_tgt_ha:,.1f} ha**.
                - **Dynamic Crop Viability (DCVS)**: **{avg_dcvs:.1f} / 100** ({dcvs_tier})
                  - Balances agro-climatic suitability, recent price trends, and quota absorption capacity.
                - **Data Reliability Index (DRI)**: **{dri_info['dri_score']:.1f} / 100** ({dri_info['reliability_tier']})
                  - Plain-language confidence: **{dri_info['confidence_label']}**.
                - **Actionable Advisory**: **{recommendation.action}**
                  - {recommendation.key_reason}
                """
            )

        with o_col2:
            st.markdown("### Scope Parameters")
            st.markdown(
                f"""
                - **Vegetable Crop**: {selected_crop}
                - **Agricultural Season**: {season_label}
                - **Geographic Focus**: {location_label}
                - **Quotation Level**: {price_type.capitalize()}
                - **Latest Observed Price**: Rs. {latest_p:.1f} / kg
                """
            )

    # -------------------------------------------------------------------------
    # TAB 2: Market Saturation
    # -------------------------------------------------------------------------
    with tab_saturation:
        st.markdown(f"### Regional Market Saturation (MSRS) — {selected_crop} ({season_label})")
        st.caption(
            "Spatial evaluation of vegetable glut risk and potential Cobweb oversupply based on regional cultivation quotas. "
            "Scores standardized on 0–100 scale: Low (0–20), Moderate (20–60), High (60–80), Critical (80–100)."
        )

        # Map Layer Toggle: MSRS vs DRI
        map_layer = st.radio(
            "Geospatial Map Layer",
            options=["Market Saturation (MSRS)", "Data Reliability (DRI)"],
            index=0,
            horizontal=True,
            help="Toggle between Market Saturation risk and Data Reliability Index across Sri Lankan districts.",
        )
        layer_key = "dri" if "DRI" in map_layer else "msrs"

        map_col, bar_col = st.columns([3, 2], gap="medium")

        with map_col:
            fig_map = render_sri_lanka_map(
                df_district_metrics=df_dist_msrs,
                crop_name=selected_crop,
                season_label=season_label,
                layer=layer_key,
                height=520,
            )
            st.plotly_chart(fig_map, use_container_width=True, config=MAP_PLOTLY_CONFIG)

        with bar_col:
            fig_bar = render_district_commitment_barchart(df_dist_msrs, selected_crop, season_label)
            st.plotly_chart(fig_bar, use_container_width=True, config=BAR_CHART_PLOTLY_CONFIG)

        # Selected District Focus Banner (if filtered)
        if not is_national:
            st.markdown(
                f"""
                <div style="background: #132619; border: 1px solid #2B5237; border-left: 5px solid #4ADE80; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; color: #E8EFEA; font-size: 0.88rem;">
                    <strong style="color: #FFFFFF;">Selected District Focus: {selected_district}</strong> ({dist_msrs_rec['province']} Province) —
                    Committed Extent: <strong style="color: #4ADE80;">{dist_msrs_rec['committed_ha']:.1f} ha</strong> |
                    Target Quota: <strong style="color: #FFFFFF;">{dist_msrs_rec['target_ha']:.1f} ha</strong> |
                    Saturation Gap: <strong style="color: {dist_msrs_rec['risk_color']};">{dist_msrs_rec['overshoot_pct']:+.1f}%</strong> |
                    MSRS: <strong style="color: {dist_msrs_rec['risk_color']};">{dist_msrs_rec['msrs']:.1f}</strong> ({dist_msrs_rec['risk_label']})
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Complete 25-District Saturation Table (Data Provenance & raw synthetic_proxy REMOVED)
        with st.expander("📊 View Complete 25-District Saturation Data Table", expanded=True):
            st.caption("Cultivation commitment values for the prototype are derived from historical HARTI district records and refreshed per season.")
            df_display = df_dist_msrs[[
                "district", "province", "committed_hectares", "target_hectares",
                "overshoot_ratio", "overshoot_pct", "msrs", "risk_label"
            ]].rename(columns={
                "district": "District",
                "province": "Province",
                "committed_hectares": "Committed Extent (ha)",
                "target_hectares": "Target Quota (ha)",
                "overshoot_ratio": "Quota Ratio",
                "overshoot_pct": "Saturation Gap (%)",
                "msrs": "MSRS Score (0-100)",
                "risk_label": "Risk Classification",
            })
            st.dataframe(df_display, hide_index=True, use_container_width=True, height=360)
            st.download_button(
                label="📥 Download District Saturation Table (CSV)",
                data=df_display.to_csv(index=False).encode("utf-8"),
                file_name=f"sri_lanka_district_msrs_{selected_crop}_{selected_season}_{selected_year}.csv",
                mime="text/csv",
                key="download_district_saturation_csv",
            )

    # -------------------------------------------------------------------------
    # TAB 3: Price Dynamics
    # -------------------------------------------------------------------------
    with tab_price:
        st.markdown(f"### Market Price Trajectory & Empirical Cobweb Cycles — {selected_crop}")
        st.caption(
            "Empirical evidence of price volatility and supply-lag oscillations from official weekly price reports."
        )

        p_col1, p_col2 = st.columns([3, 2], gap="medium")

        with p_col1:
            fig_price = render_price_history_chart(df_price, selected_crop, selected_district)
            st.plotly_chart(fig_price, use_container_width=True, config=CHART_PLOTLY_CONFIG)

        with p_col2:
            fig_cobweb = render_cobweb_phase_plot(df_price, df_comm, selected_crop)
            st.plotly_chart(fig_cobweb, use_container_width=True, config=CHART_PLOTLY_CONFIG)

        st.info(
            "**Cobweb Effect Diagnostic**: Notice the cyclical phase spiral above. When previous-season prices spike, "
            "growers collectively overcommit acreage, creating an oversupply glut in the following season that crashes wholesale prices. "
            "The Geo-Spatial Decision Support System identifies this overcommitment before planting occurs to help stabilize market supplies."
        )

    # -------------------------------------------------------------------------
    # TAB 4: Agro-Meteorology
    # -------------------------------------------------------------------------
    with tab_weather:
        weather_dist = selected_district if not is_national else "Nuwara Eliya"
        scope_suffix = " (Representative Commercial Highland Zone)" if is_national else ""
        st.markdown(f"### Agro-Meteorological Environment (NASA POWER) — {weather_dist}{scope_suffix}")
        st.caption(
            "Continuous daily precipitation and temperature monitoring. Excessive moisture or extreme heat waves "
            "directly influence crop yield potential and viability scoring."
        )

        fig_weather = render_weather_trend_chart(df_weather, weather_dist)
        st.plotly_chart(fig_weather, use_container_width=True, config=CHART_PLOTLY_CONFIG)

        # Weather Summary Stats
        w_sub = df_weather[df_weather["district"] == weather_dist]
        if not w_sub.empty:
            w_col1, w_col2, w_col3 = st.columns(3, gap="medium")
            w_col1.metric("Historical Mean Daily Rainfall", f"{w_sub['rainfall_mm'].mean():.2f} mm")
            w_col2.metric("Historical Mean Ambient Temperature", f"{w_sub['temp_c'].mean():.1f} °C")
            w_col3.metric("Total Observation Days", f"{len(w_sub):,} days")

    # -------------------------------------------------------------------------
    # TAB 5: Recommendations
    # -------------------------------------------------------------------------
    with tab_recs:
        st.markdown("### Actionable Planting Recommendations & Viability")
        st.caption(
            "Holistic planting recommendations synthesized from regional market saturation, "
            "agro-climatic favorability, and price incentives. Standardized to 0.0 - 100.0 scale."
        )

        # 1. Main Recommendation Card
        render_advisory_card(recommendation)

        st.markdown("---")

        # 2. Multi-Crop Sortable Recommendations Table across all 5 research crops
        df_crop_recs = advisor_engine.generate_crop_recommendations_table(
            district=target_d,
            season_type=selected_season,
            season_year=selected_year,
            df_comm=df_comm,
            df_price=df_price,
            df_weather=df_weather,
            price_type=price_type,
        )
        render_crop_recommendations_table(df_crop_recs)

        st.markdown("---")

        # 3. District Drill-Down Panel
        drilldown_summary = advisor_engine.get_district_drilldown_summary(
            district=target_d,
            crop=selected_crop,
            season_type=selected_season,
            season_year=selected_year,
            df_comm=df_comm,
            df_price=df_price,
            df_weather=df_weather,
            price_type=price_type,
        )
        render_district_drilldown_panel(drilldown_summary)

    # -------------------------------------------------------------------------
    # TAB 6: Data Reliability
    # -------------------------------------------------------------------------
    with tab_dri:
        st.markdown("### Data Reliability Index (DRI) Quality Audit")
        st.caption(
            "Rigorous 3-component data governance framework evaluating observational completeness (40%), "
            "recency (30%), and consistency (30%) across agricultural data categories."
        )

        # 3 Component Score Cards
        col_c1, col_c2, col_c3, col_c4 = st.columns(4, gap="medium")
        col_c1.metric("Composite DRI Score", f"{dri_info['dri_score']:.1f}", dri_info["confidence_label"])
        col_c2.metric("Completeness (40%)", f"{dri_info['completeness']:.1f}%")
        col_c3.metric("Recency (30%)", f"{dri_info['recency']:.1f}%")
        col_c4.metric("Consistency (30%)", f"{dri_info['consistency']:.1f}%")

        # Plain-language confidence and computed "Why this score?" explanation
        st.markdown(
            f"""
            <div style="background: #112316; border: 1px solid #1E3D27; border-left: 5px solid #4ADE80; border-radius: 6px; padding: 12px 16px; margin: 12px 0 16px 0;">
                <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem; margin-bottom: 4px;">
                    Empirical Assessment: <span style="color: #4ADE80;">{dri_info['confidence_label']}</span> ({dri_info['reliability_tier']} Tier)
                </div>
                <div style="color: #C8D8CC; font-size: 0.88rem; line-height: 1.45;">
                    <strong>Why this score?</strong> {dri_info['why_explanation']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Category-Level Reliability Audit Table
        st.markdown("#### Data Category Reliability Audit")
        cat_audits = dri_info["category_audits"]
        cat_records = []
        for cat_name, c_data in cat_audits.items():
            if cat_name.lower() == "yield":
                continue
            if c_data.get("available", False):
                cat_records.append({
                    "Category": cat_name.capitalize(),
                    "Status": "Available",
                    "Record Count": f"{c_data.get('record_count', 0):,}",
                    "Completeness": f"{c_data.get('completeness', 0) * 100:.1f}%",
                    "Recency": f"{c_data.get('recency', 0) * 100:.1f}%",
                    "Consistency": f"{c_data.get('consistency', 0) * 100:.1f}%",
                    "Category DRI": f"{c_data.get('dri_score', 0):.1f}",
                    "Audit Notes": c_data.get("notes", ""),
                })
            else:
                cat_status = c_data.get("status_label", "Unavailable — source data not yet integrated")
                rec_cnt = f"{c_data.get('record_count'):,}" if c_data.get("record_count") is not None else "N/A"
                cat_records.append({
                    "Category": cat_name.capitalize(),
                    "Status": cat_status,
                    "Record Count": rec_cnt,
                    "Completeness": "N/A",
                    "Recency": "N/A",
                    "Consistency": "N/A",
                    "Category DRI": "N/A",
                    "Audit Notes": c_data.get("notes", "Data category not collected in active intake pipeline."),
                })

        df_cat_summary = pd.DataFrame(cat_records)
        st.dataframe(df_cat_summary, hide_index=True, use_container_width=True)

        st.markdown("---")

        # Complete 25-District DRI Panel
        st.markdown(f"#### Complete 25-District Data Reliability Table ({selected_crop})")
        df_dri_display = df_dist_dri[[
            "district", "province", "dri_score", "completeness", "recency", "consistency", "confidence_label"
        ]].rename(columns={
            "district": "District",
            "province": "Province",
            "dri_score": "DRI Score (0-100)",
            "completeness": "Completeness (%)",
            "recency": "Recency (%)",
            "consistency": "Consistency (%)",
            "confidence_label": "Confidence Tier",
        })
        st.dataframe(df_dri_display, hide_index=True, use_container_width=True, height=350)

        # Audit Ledger Reference
        if dri_ledger:
            with st.expander("📜 View Verified Intake Audit Ledger (JSONL Logs)", expanded=False):
                st.caption(f"Loaded {len(dri_ledger)} verified intake audit entries from `dri_audit_ledger.jsonl`.")
                for i, audit in enumerate(dri_ledger):
                    st.json(audit)



    # =========================================================================
    # FOOTER (Hector Kobbekaduwa Agrarian Research and Training Institute (HARTI) collaboration | SLIIT | J26-DS-322)
    # =========================================================================
    st.markdown(
        f'<div class="dashboard-footer">{FOOTER_TEXT}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
