"""
Geo-Spatial Decision Support System (DSS) Dashboard.

Research Project: Data-Driven Decision Support System to Mitigate the Cobweb Effect in Sri Lankan Vegetable Markets
Institution: Sri Lanka Institute of Information Technology (SLIIT)
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
    page_title="Geo-Spatial DSS | Vegetable Market Intelligence",
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
    get_dri_evaluation_summary,
)
from engine.forecast_bridge import (
    ForecastAdapter,
)
from engine.advisor import (
    FarmerAdvisoryEngine,
)
from components.maps import (
    render_sri_lanka_saturation_map,
    render_district_viability_map,
)
from components.charts import (
    render_price_history_chart,
    render_cobweb_phase_plot,
    render_district_commitment_barchart,
    render_weather_trend_chart,
    render_forecast_trajectory_chart,
)
from components.metrics_cards import (
    render_executive_overview_cards,
)
from components.recommendations import (
    render_advisory_card,
)

# Apply Custom Professional Green Theme
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    # Load Primary Research Datasets
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
    # SIDEBAR: Research Filters & Navigation
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
            help="Strictly restricted to the 5 primary research crops.",
        )

        # 2. Season & Year Selectors
        avail_seasons = get_available_seasons(df_comm)
        avail_years = sorted(list(set(y for _, y in avail_seasons)), reverse=True)
        selected_year = st.selectbox(
            "Planning Season Year",
            options=avail_years,
            index=1 if len(avail_years) > 1 else 0
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
            help="Filter spatial indicators to a specific district or evaluate national aggregates.",
        )

        # 4. Price Type Filter
        price_type = st.radio(
            "Price Quotation Level",
            options=["retail", "wholesale"],
            index=0,
            horizontal=True
        )

        st.markdown("---")
        st.subheader("Price Forecast Simulation")

        # Demo Forecast Simulation Toggle (strictly labeled)
        enable_demo_forecast = st.toggle(
            "Enable 12-Week Forecast Projection",
            value=False,
            help="Displays forward 12-week price projections and uncertainty corridors for demonstration.",
        )

        st.caption("Data Source Provenance:")
        st.markdown("- **Vegetable Prices**: <span class='badge-real'>HARTI / CBSL (Real Data)</span>", unsafe_allow_html=True)
        st.markdown("- **Agro-Weather**: <span class='badge-real'>NASA POWER (Real Data)</span>", unsafe_allow_html=True)
        st.markdown("- **Cultivation Commitment**: <span class='badge-proxy'>DOA Baseline (Synthetic Proxy)</span>", unsafe_allow_html=True)
        if enable_demo_forecast:
            st.markdown("- **Forward Projection**: <span class='badge-demo'>Demonstration Forecast</span>", unsafe_allow_html=True)

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
    st.markdown(
        f'<div class="dashboard-caption">'
        f'<span class="badge-real">HARTI Prices — Real Data</span> '
        f'<span class="badge-real">NASA POWER Weather — Real Data</span> '
        f'<span class="badge-proxy">DOA Commitment — Synthetic Proxy</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Compute Core Metrics
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
    else:
        current_msrs = msrs_summary["national_msrs"]
        msrs_tier = msrs_summary["risk_tier"]
        msrs_color = msrs_summary["risk_color"]
        current_comm_ha = msrs_summary["total_committed_ha"]

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
        dcvs_tier = "HIGH" if avg_dcvs >= 70 else ("MODERATE" if avg_dcvs >= 45 else "LOW")
        dcvs_color = "#2E7D32" if dcvs_tier == "HIGH" else ("#D97706" if dcvs_tier == "MODERATE" else "#C53030")

    # DRI audit status (HARTI source-level intake audit benchmark)
    target_d = selected_district if not is_national else "All Island (National Average)"
    dri_info = assess_district_data_readiness(target_d, selected_crop, df_price, df_weather, df_comm)

    # Render Executive KPI Cards
    render_executive_overview_cards(
        latest_price=latest_p,
        current_msrs=current_msrs,
        msrs_tier=msrs_tier,
        msrs_color=msrs_color,
        dcvs_score=avg_dcvs,
        dcvs_tier=dcvs_tier,
        dcvs_color=dcvs_color,
        dri_score=dri_info["benchmark_dri_score"],
        dri_tier=dri_info["reliability_tier"],
        dri_color=dri_info["tier_color"],
        location_label=location_label,
        dri_provenance=dri_info["provenance_status"],
        committed_ha=current_comm_ha,
    )

    # Active Filter Context Confirmation Banner
    st.markdown(
        f'<div class="filter-context-banner">'
        f'<span>Viewing Scope: <strong>{selected_crop}</strong> | <strong>{season_label}</strong> | <strong>{location_label}</strong> ({price_type.capitalize()} Quotes)</span>'
        f'<span><span class="badge-real">REAL PRICE DATA</span> <span class="badge-proxy">SYNTHETIC COMMITMENT</span> <span class="badge-real">REAL WEATHER</span></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # =========================================================================
    # TABS: Multi-View Decision Support Panels
    # =========================================================================
    tab_geo, tab_price, tab_weather, tab_advisor, tab_dri, tab_forecast = st.tabs([
        "🗺️ Market Saturation",
        "📈 Price Dynamics & Cobweb",
        "🌦️ Agro-Meteorology",
        "👨‍🌾 Farmer Advisory",
        "🛡️ Data Reliability",
        "🔮 Price Forecast",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Geo-Spatial Saturation Map (MSRS)
    # -------------------------------------------------------------------------
    with tab_geo:
        st.markdown(f"### Regional Market Saturation (MSRS) — {selected_crop} ({season_label})")
        st.caption(
            "Spatial evaluation of vegetable glut risk and potential Cobweb oversupply based on regional cultivation quotas. "
            "Red markers indicate critical saturation risk (MSRS ≥ 0.70)."
        )

        map_col, bar_col = st.columns([3, 2], gap="medium")

        with map_col:
            fig_map = render_sri_lanka_saturation_map(df_dist_msrs, selected_crop, season_label)
            st.plotly_chart(fig_map)

        with bar_col:
            fig_bar = render_district_commitment_barchart(df_dist_msrs, selected_crop, season_label)
            st.plotly_chart(fig_bar)

        # Highlight selected district if filtered
        if not is_national:
            st.markdown(
                f"""
                <div style="background: #132619; border: 1px solid #2B5237; border-left: 5px solid #4ADE80; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; color: #E8EFEA; font-size: 0.88rem;">
                    <strong style="color: #FFFFFF;">Selected District Focus: {selected_district}</strong> ({dist_msrs_rec['province']} Province) — 
                    Committed Area: <strong style="color: #4ADE80;">{dist_msrs_rec['committed_ha']:.1f} ha</strong> | 
                    Target Quota: <strong style="color: #FFFFFF;">{dist_msrs_rec['target_ha']:.1f} ha</strong> | 
                    Saturation Gap: <strong style="color: {dist_msrs_rec['risk_color']};">{dist_msrs_rec['overshoot_pct']:+.1f}%</strong> | 
                    MSRS: <strong style="color: {dist_msrs_rec['risk_color']};">{dist_msrs_rec['msrs']:.2f}</strong> ({dist_msrs_rec['risk_label']})
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("📊 View Complete 25-District Saturation Data Table", expanded=True):
            df_display = df_dist_msrs[[
                "district", "province", "committed_hectares", "target_hectares",
                "overshoot_ratio", "overshoot_pct", "msrs", "risk_label", "data_source"
            ]].rename(columns={
                "district": "District",
                "province": "Province",
                "committed_hectares": "Committed Area (ha)",
                "target_hectares": "Target Quota (ha)",
                "overshoot_ratio": "Quota Ratio",
                "overshoot_pct": "Saturation Gap (%)",
                "msrs": "MSRS Score",
                "risk_label": "Risk Classification",
                "data_source": "Data Provenance",
            })
            st.dataframe(df_display, hide_index=True, height=350)

    # -------------------------------------------------------------------------
    # TAB 2: Market Price Dynamics & Cobweb Cycles
    # -------------------------------------------------------------------------
    with tab_price:
        st.markdown(f"### Market Price Trajectory & Empirical Cobweb Cycles — {selected_crop}")
        st.caption(
            "Empirical evidence of price volatility and supply-lag oscillations from official weekly price reports."
        )

        p_col1, p_col2 = st.columns([3, 2], gap="medium")

        with p_col1:
            fig_price = render_price_history_chart(df_price, selected_crop, selected_district)
            st.plotly_chart(fig_price)

        with p_col2:
            fig_cobweb = render_cobweb_phase_plot(df_price, df_comm, selected_crop)
            st.plotly_chart(fig_cobweb)

        st.info(
            "**Cobweb Effect Diagnostic**: Notice the cyclical phase spiral above. When previous-season prices spike, "
            "growers collectively overcommit acreage, creating an oversupply glut in the following season that crashes wholesale prices. "
            "The Geo-Spatial Decision Support System identifies this overcommitment before planting occurs to help stabilize market supplies."
        )

    # -------------------------------------------------------------------------
    # TAB 3: Agro-Meteorological Environment (NASA POWER)
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
        st.plotly_chart(fig_weather)

        # Weather Summary Stats
        w_sub = df_weather[df_weather["district"] == weather_dist]
        if not w_sub.empty:
            w_col1, w_col2, w_col3 = st.columns(3, gap="medium")
            w_col1.metric("Historical Mean Rainfall", f"{w_sub['rainfall_mm'].mean():.2f} mm/day")
            w_col2.metric("Historical Mean Temperature", f"{w_sub['temp_c'].mean():.1f} °C")
            w_col3.metric("Total Observation Days", f"{len(w_sub):,} days")

    # -------------------------------------------------------------------------
    # TAB 4: Farmer Advisory & Dynamic Viability (DCVS)
    # -------------------------------------------------------------------------
    with tab_advisor:
        st.markdown("### Proactive Farmer Decision Support & Viability (DCVS)")
        st.caption(
            "Holistic planting recommendations synthesized from regional market saturation, "
            "agro-climatic favorability, and price incentives."
        )

        adv_dist = selected_district if not is_national else "All Island (National Average)"

        advisor_engine = FarmerAdvisoryEngine()
        recommendation = advisor_engine.generate_advisory(
            district=adv_dist,
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

        # Render Recommendation Card
        render_advisory_card(recommendation)

        if is_national:
            st.info(
                "💡 **District-Level Granularity**: Currently viewing **National Aggregate** market capacity guidance. "
                "To inspect localized planting advisories for specific high-production farming districts "
                "(e.g., Nuwara Eliya, Badulla, or Matale), choose a district from the sidebar."
            )

        st.markdown("---")
        st.markdown(f"#### Spatial Crop Viability Map (DCVS) — {selected_crop} ({season_label})")
        fig_dcvs_map = render_district_viability_map(df_dcvs, selected_crop, season_label)
        st.plotly_chart(fig_dcvs_map)

    # -------------------------------------------------------------------------
    # TAB 5: Data Readiness Index (DRI) Quality Audit
    # -------------------------------------------------------------------------
    with tab_dri:
        st.markdown("### Data Reliability & Quality Audit (DRI)")
        st.caption(
            "Data governance framework assessing observational completeness, timeliness, "
            "and consistency across vegetable price and meteorological sources."
        )

        if dri_ledger:
            st.success(f"Loaded {len(dri_ledger)} verified empirical audit evaluations from `dri_audit_ledger.jsonl`.")
            for i, audit in enumerate(dri_ledger):
                with st.expander(f"Audit Record #{i+1}: {audit.get('source_agency')} ({audit.get('reliability_tier')} Tier — DRI: {audit.get('calculated_dri')}%)"):
                    st.json(audit)
        else:
            st.info("Showing live stream evaluation from active observation ledger:")

        # Show Live Stream Readiness Evaluation
        st.markdown("#### Live Stream Observation Health Audit")
        col_d1, col_d2, col_d3 = st.columns(3, gap="medium")
        col_d1.metric("Overall Completeness Score", f"{dri_info['overall_completeness_score']:.1f}%")
        col_d2.metric("Benchmark DRI Score", f"{dri_info['benchmark_dri_score']:.2f}%")
        col_d3.metric("Reliability Classification", f"{dri_info['reliability_tier']} ({dri_info['source_agency']})")

        st.markdown(
            f"**Data Intake Benchmark**: <span class='badge-real'>{dri_info['tier_label']}</span><br>"
            f"**Audit Scope**: <em>{dri_info['dri_scope']}</em> — {dri_info['scope_description']}<br>"
            f"**Observational Health**: Audited across {dri_info['price_records_count']} weekly price observations and "
            f"{dri_info['weather_records_count']} daily meteorological readings for <strong>{selected_crop}</strong> in <strong>{location_label}</strong>.",
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # TAB 6: Price Forecasting & Projection
    # -------------------------------------------------------------------------
    with tab_forecast:
        st.markdown("### Price Forecasting & Projection")
        st.caption(
            "Forward price projection model providing 12-week expected trajectories and "
            "uncertainty bands to anticipate upcoming market movements."
        )

        forecast_adapter = ForecastAdapter()
        forecast_dist = selected_district if not is_national else "Colombo"

        res = forecast_adapter.get_forecast(
            crop=selected_crop,
            district=forecast_dist,
            historical_price_df=df_price,
            enable_demo_simulation=enable_demo_forecast,
            weeks_ahead=12,
        )

        badge_cls = "badge-demo" if enable_demo_forecast else "badge-proxy"
        st.markdown(
            f"**Forecast Status**: <span class='{badge_cls}'>{res.status_label}</span> | "
            f"**Model Specification**: `{res.model_name}` | "
            f"**Confidence Band**: `{res.confidence_level * 100:.0f}%`",
            unsafe_allow_html=True,
        )

        # Render Integrated Chart (Historical Baseline + Simulated Forward Corridors)
        fig_fc = render_forecast_trajectory_chart(
            forecast_df=res.forecast_df,
            historical_df=res.historical_df,
            crop_name=selected_crop,
            district=forecast_dist,
            height=380,
        )
        st.plotly_chart(fig_fc)

        if not res.forecast_df.empty:
            mean_fc_price = float(res.forecast_df["predicted_price"].mean())
            min_fc_price = float(res.forecast_df["lower_bound_95"].min())
            max_fc_price = float(res.forecast_df["upper_bound_95"].max())

            fc_c1, fc_c2, fc_c3 = st.columns(3, gap="medium")
            fc_c1.metric("12-Week Mean Forecast", f"Rs. {mean_fc_price:.1f} / kg")
            fc_c2.metric("95% Lower Corridor", f"Rs. {min_fc_price:.1f} / kg")
            fc_c3.metric("95% Upper Corridor", f"Rs. {max_fc_price:.1f} / kg")

            st.markdown(
                f"""
                <div class="banner-warning">
                    <strong>Demo Simulation Notice:</strong> {res.narrative_summary}
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("📄 View Forecast Projection Table (Demonstration Data)"):
                st.dataframe(res.forecast_df, hide_index=True, height=280)
        else:
            st.markdown(
                f"""
                <div class="banner-safe">
                    <strong>Forecast Baseline Ready:</strong> {res.narrative_summary}
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
