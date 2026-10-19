"""
Automated unit and regression tests for Geo-Spatial DSS Dashboard engines,
presentation components, and Member 2 forecast adapters.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure geospatial-dss-dashboard directory is on sys.path
dashboard_root = str(Path(__file__).resolve().parent.parent)
if dashboard_root not in sys.path:
    sys.path.insert(0, dashboard_root)

from config import (
    DISTRICT_COORDS,
    DISTRICT_NAMES,
    TARGET_CROPS,
    COLOR_FOREST_GREEN,
    COLOR_SAGE_GREEN,
    COLOR_SAFE_GREEN,
    COLOR_WARNING_AMBER,
    COLOR_CRITICAL_RED,
    CUSTOM_CSS,
)
from data_loader import (
    load_price_data,
    load_commitment_data,
    load_weather_data,
    load_dri_audit_ledger,
    get_available_seasons,
)
from engine.msrs_engine import (
    calculate_msrs,
    evaluate_saturation_risk,
    compute_district_msrs_table,
    compute_national_msrs_summary,
    get_district_msrs_record,
)
from engine.dcvs_engine import (
    calculate_dcvs,
    evaluate_viability_tier,
    compute_district_dcvs_table,
)
from engine.dri_bridge import (
    get_dri_evaluation_summary,
    get_dri_source_benchmark,
    assess_district_data_readiness,
)
from engine.forecast_bridge import (
    ForecastAdapter,
)
from engine.advisor import (
    FarmerAdvisoryEngine,
)
from components.charts import (
    render_forecast_trajectory_chart,
    render_cobweb_phase_plot,
)


# =====================================================================
# 1. Data Loader & Schema Verification Tests
# =====================================================================

def test_price_data_loader_and_schema():
    """Verify analysis_ready_price.csv loads with required columns and types."""
    df_price = load_price_data()
    assert not df_price.empty
    expected_cols = ["price_date", "district", "crop", "price_type", "granularity", "price", "source"]
    for col in expected_cols:
        assert col in df_price.columns

    # Verify 5 research crops
    crops = set(df_price["crop"].unique())
    assert "Carrot" in crops
    assert "Big Onion (Local)" in crops
    assert len(df_price["district"].unique()) == 25


def test_commitment_data_loader_and_schema():
    """Verify analysis_ready_commitment.csv loads with required columns."""
    df_comm = load_commitment_data()
    assert not df_comm.empty
    expected_cols = ["district", "crop_name", "season_type", "season_year", "target_hectares", "committed_hectares"]
    for col in expected_cols:
        assert col in df_comm.columns

    assert set(df_comm["season_type"].unique()) == {"Maha", "Yala"}
    assert len(df_comm["district"].unique()) == 25


def test_weather_data_loader_and_schema():
    """Verify weather_data.json loads correctly and maps district names."""
    df_weather = load_weather_data()
    assert not df_weather.empty
    assert "district" in df_weather.columns
    assert "obs_date" in df_weather.columns
    assert "rainfall_mm" in df_weather.columns
    assert "temp_c" in df_weather.columns
    assert len(df_weather["district"].unique()) == 25


def test_dri_audit_ledger_loader():
    """Verify dri_audit_ledger.jsonl can be loaded without error."""
    records = load_dri_audit_ledger()
    assert isinstance(records, list)
    if records:
        first = records[0]
        assert "calculated_dri" in first
        assert "reliability_tier" in first


# =====================================================================
# 2. MSRS Calculation Engine Tests
# =====================================================================

def test_msrs_calculation_formula():
    """Verify MSRS boundary conditions and linear scaling."""
    # Safe zone (<= 80% capacity) -> 0.0
    assert calculate_msrs(committed_area=70.0, market_demand_capacity=100.0) == 0.0
    assert calculate_msrs(committed_area=80.0, market_demand_capacity=100.0) == 0.0

    # Critical zone (>= 120% capacity) -> 1.0
    assert calculate_msrs(committed_area=120.0, market_demand_capacity=100.0) == 1.0
    assert calculate_msrs(committed_area=150.0, market_demand_capacity=100.0) == 1.0

    # Midpoint (100% capacity) -> 0.5
    assert calculate_msrs(committed_area=100.0, market_demand_capacity=100.0) == 0.5

    # Edge cases
    assert calculate_msrs(committed_area=0.0, market_demand_capacity=100.0) == 0.0
    assert calculate_msrs(committed_area=50.0, market_demand_capacity=0.0) == 1.0


def test_msrs_risk_evaluation():
    """Verify restrained risk classification and color assignments."""
    tier, label, color = evaluate_saturation_risk(0.15)
    assert tier == "SAFE"
    assert color == COLOR_SAFE_GREEN

    tier, label, color = evaluate_saturation_risk(0.50)
    assert tier == "WARNING"
    assert color == COLOR_WARNING_AMBER

    tier, label, color = evaluate_saturation_risk(0.85)
    assert tier == "CRITICAL"
    assert color == COLOR_CRITICAL_RED


def test_district_msrs_panel_generation():
    """Verify district MSRS panel covers all 25 districts with valid coordinates."""
    df_comm = load_commitment_data()
    panel = compute_district_msrs_table(df_comm, "Carrot", "Maha", 2024)

    assert len(panel) == 25
    assert set(panel["district"]) == set(DISTRICT_NAMES)
    assert panel["msrs"].between(0.0, 1.0).all()
    assert panel["latitude"].notna().all()
    assert panel["longitude"].notna().all()


def test_district_and_national_metric_consistency():
    """Verify that district record extraction matches the computed panel and national summary."""
    df_comm = load_commitment_data()
    panel = compute_district_msrs_table(df_comm, "Leek", "Yala", 2024)
    summary = compute_national_msrs_summary(df_comm, "Leek", "Yala", 2024)

    # Check that individual district record matches table exactly
    rec = get_district_msrs_record(panel, "Nuwara Eliya")
    row = panel[panel["district"] == "Nuwara Eliya"].iloc[0]
    assert rec["msrs"] == float(row["msrs"])
    assert rec["committed_ha"] == float(row["committed_hectares"])
    assert rec["target_ha"] == float(row["target_hectares"])

    # National totals match sum of districts
    assert round(summary["total_committed_ha"], 1) == round(float(panel["committed_hectares"].sum()), 1)
    assert round(summary["total_target_ha"], 1) == round(float(panel["target_hectares"].sum()), 1)


# =====================================================================
# 3. DCVS Calculation Engine Tests
# =====================================================================

def test_dcvs_calculation_formula():
    """Verify DCVS weighted formula: 40% weather, 30% price, 30% (1-msrs)."""
    # Ideal conditions: weather=1.0, price=1.0, msrs=0.0 (saturation=1.0) -> 100%
    assert calculate_dcvs(msrs=0.0, weather_index=1.0, price_forecast_index=1.0) == 100.0

    # Terrible conditions: weather=0.0, price=0.0, msrs=1.0 (saturation=0.0) -> 0%
    assert calculate_dcvs(msrs=1.0, weather_index=0.0, price_forecast_index=0.0) == 0.0

    # Balanced midpoint: weather=0.5, price=0.5, msrs=0.5 (saturation=0.5) -> 50%
    assert calculate_dcvs(msrs=0.5, weather_index=0.5, price_forecast_index=0.5) == 50.0


def test_district_dcvs_table_generation():
    """Verify district DCVS table merges price, weather, and commitment streams."""
    df_comm = load_commitment_data()
    df_price = load_price_data()
    df_weather = load_weather_data()

    dcvs_df = compute_district_dcvs_table(df_comm, df_price, df_weather, "Tomato", "Yala", 2024)

    assert len(dcvs_df) == 25
    assert dcvs_df["dcvs"].between(0.0, 100.0).all()
    assert "viability_tier" in dcvs_df.columns
    assert "current_price" in dcvs_df.columns


# =====================================================================
# 4. Member 2 Forecasting Adapter & Plotly Chart Tests
# =====================================================================

def test_forecast_adapter_standby_mode():
    """Verify that when demo simulation is disabled, baseline historical prices and clean status are returned."""
    df_price = load_price_data()
    adapter = ForecastAdapter()

    res = adapter.get_forecast("Carrot", "Nuwara Eliya", df_price, enable_demo_simulation=False)
    assert not res.is_live
    assert "HISTORICAL BASELINE" in res.status_label
    assert res.forecast_df.empty
    assert not res.historical_df.empty
    assert len(res.narrative_summary) > 0


def test_forecast_adapter_demo_simulation_mode():
    """Verify that demo forecast simulation is strictly watermarked with 'DEMO DATA'."""
    df_price = load_price_data()
    adapter = ForecastAdapter()

    res = adapter.get_forecast("Carrot", "Nuwara Eliya", df_price, enable_demo_simulation=True, weeks_ahead=12)
    assert not res.is_live
    assert "DEMO" in res.status_label
    assert len(res.forecast_df) == 12
    assert "DEMO DATA" in res.forecast_df["data_tag"].values
    assert not res.historical_df.empty
    assert len(res.narrative_summary) > 0


def test_forecast_toggle_repeated_executions():
    """
    Regression Test for NameError: name 'go' is not defined.
    Verifies switching the demo forecast toggle on and off repeatedly executes
    cleanly and produces valid Plotly Figure instances.
    """
    df_price = load_price_data()
    adapter = ForecastAdapter()

    for toggle_val in [False, True, False, True]:
        res = adapter.get_forecast("Cabbage", "Kandy", df_price, enable_demo_simulation=toggle_val, weeks_ahead=12)
        fig = render_forecast_trajectory_chart(
            forecast_df=res.forecast_df,
            historical_df=res.historical_df,
            crop_name="Cabbage",
            district="Kandy",
        )
        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) >= 1  # Contains at least historical price series


# =====================================================================
# 5. Farmer Advisory Engine Tests
# =====================================================================

def test_farmer_advisory_generation():
    """Verify that the advisory engine produces transparent recommendations with alternative crops."""
    df_comm = load_commitment_data()
    df_price = load_price_data()
    df_weather = load_weather_data()

    advisor = FarmerAdvisoryEngine()
    rec = advisor.generate_advisory(
        district="Badulla",
        crop="Carrot",
        season_type="Maha",
        season_year=2024,
        df_comm=df_comm,
        df_price=df_price,
        df_weather=df_weather,
    )

    assert rec.district == "Badulla"
    assert rec.crop == "Carrot"
    assert rec.action in ["FAVORABLE FOR PLANTING", "PROCEED WITH CAUTION", "AVOID EXPANSION / DIVERSIFY"]
    assert len(rec.rationale_points) >= 3
    assert rec.dri_score == 86.86
    assert rec.dri_source == "HARTI"
    assert "HARTI" in rec.confidence_limitation
    assert len(rec.suggested_alternatives) >= 1


def test_farmer_advisory_alternatives_robustness():
    """Verify that every district receives robust alternative crop recommendations with transparent reasons."""
    df_comm = load_commitment_data()
    df_price = load_price_data()
    df_weather = load_weather_data()

    advisor = FarmerAdvisoryEngine()
    for sample_district in ["Colombo", "Jaffna", "Nuwara Eliya", "Matale"]:
        rec = advisor.generate_advisory(
            district=sample_district,
            crop="Carrot",
            season_type="Yala",
            season_year=2024,
            df_comm=df_comm,
            df_price=df_price,
            df_weather=df_weather,
        )
        assert len(rec.suggested_alternatives) > 0
        for alt in rec.suggested_alternatives:
            assert alt["crop"] != "Carrot"
            assert "msrs" in alt
            assert "dcvs" in alt
            assert "explanation" in alt


def test_dri_source_selection_and_rounding():
    """Verify DRI source selection, rounding, and audit ledger integrity."""
    harti_bm = get_dri_source_benchmark("HARTI")
    assert harti_bm["source_agency"] == "HARTI"
    assert harti_bm["calculated_dri"] == 86.86
    assert harti_bm["calculated_dri_rounded"] == 86.9
    assert harti_bm["reliability_tier"] == "HIGH"
    assert "National Source Benchmark" in harti_bm["scope"]

    cbsl_bm = get_dri_source_benchmark("CBSL")
    assert cbsl_bm["source_agency"] == "CBSL"
    assert cbsl_bm["calculated_dri"] == 98.4
    assert cbsl_bm["reliability_tier"] == "HIGH"


def test_source_level_vs_district_level_reliability_labeling():
    """Verify that DRI is explicitly labeled as a source-level benchmark and not district-specific."""
    df_price = load_price_data()
    df_weather = load_weather_data()
    df_comm = load_commitment_data()

    dri_info = assess_district_data_readiness("Badulla", "Carrot", df_price, df_weather, df_comm)
    assert dri_info["benchmark_dri_score"] == 86.86
    assert dri_info["is_district_specific"] is False
    assert "HARTI" in dri_info["source_agency"]
    assert "National Source Benchmark" in dri_info["dri_scope"]
    assert "not independently establish district-specific data quality" in dri_info["confidence_limitation"]


def test_missing_input_data_handling():
    """Verify that missing inputs are transparently flagged in evidence sufficiency."""
    df_weather = load_weather_data()
    empty_price = pd.DataFrame(columns=["district", "crop", "price", "price_type", "price_date"])

    dri_info = assess_district_data_readiness("Badulla", "Carrot", empty_price, df_weather)
    assert dri_info["is_evidence_adequate"] is False
    assert "Missing Price Observations" in dri_info["evidence_sufficiency"]


def test_separation_of_dri_msrs_dcvs_and_recommendation():
    """Verify that high DRI does not override critical market saturation (MSRS >= 0.70)."""
    df_comm = load_commitment_data()
    df_price = load_price_data()
    df_weather = load_weather_data()

    advisor = FarmerAdvisoryEngine()
    # Force precomputed_msrs to 0.85 (Critical saturation) despite High DRI (86.86%)
    rec = advisor.generate_advisory(
        district="Badulla",
        crop="Carrot",
        season_type="Maha",
        season_year=2024,
        df_comm=df_comm,
        df_price=df_price,
        df_weather=df_weather,
        precomputed_msrs=0.85,
    )

    # Must advise AVOID EXPANSION despite high source data readiness
    assert rec.action == "AVOID EXPANSION / DIVERSIFY"
    assert rec.msrs_score == 0.85
    assert rec.dri_score == 86.86
    assert rec.dri_tier == "HIGH"


def test_dashboard_metrics_consistency_across_views():
    """Verify 100% numerical consistency between KPI, map, table, and advisory for both National and District scopes."""
    df_comm = load_commitment_data()
    df_price = load_price_data()
    df_weather = load_weather_data()

    # 1. District Scope: Nuwara Eliya
    dist_msrs_df = compute_district_msrs_table(df_comm, "Carrot", "Maha", 2024)
    dist_dcvs_df = compute_district_dcvs_table(df_comm, df_price, df_weather, "Carrot", "Maha", 2024, price_type="retail")
    dist_rec_msrs = get_district_msrs_record(dist_msrs_df, "Nuwara Eliya")
    dist_dcvs_val = float(dist_dcvs_df[dist_dcvs_df["district"] == "Nuwara Eliya"]["dcvs"].iloc[0])

    advisor = FarmerAdvisoryEngine()
    adv_dist = advisor.generate_advisory(
        district="Nuwara Eliya",
        crop="Carrot",
        season_type="Maha",
        season_year=2024,
        df_comm=df_comm,
        df_price=df_price,
        df_weather=df_weather,
        price_type="retail",
        precomputed_msrs=dist_rec_msrs["msrs"],
        precomputed_dcvs=dist_dcvs_val,
    )

    assert adv_dist.msrs_score == dist_rec_msrs["msrs"]
    assert adv_dist.dcvs_score == dist_dcvs_val

    # 2. National Scope
    nat_msrs_summary = compute_national_msrs_summary(df_comm, "Carrot", "Maha", 2024)
    nat_mean_dcvs = float(dist_dcvs_df["dcvs"].mean())

    adv_nat = advisor.generate_advisory(
        district="All Island (National Average)",
        crop="Carrot",
        season_type="Maha",
        season_year=2024,
        df_comm=df_comm,
        df_price=df_price,
        df_weather=df_weather,
        price_type="retail",
        precomputed_msrs=nat_msrs_summary["national_msrs"],
        precomputed_dcvs=nat_mean_dcvs,
    )

    assert adv_nat.msrs_score == nat_msrs_summary["national_msrs"]
    assert adv_nat.dcvs_score == nat_mean_dcvs


# =====================================================================
# 6. Green Theme & CSS Verification Tests
# =====================================================================

def test_theme_and_config_tokens():
    """Verify that agricultural green color tokens and tab styling are defined."""
    assert COLOR_FOREST_GREEN == "#1E5631"
    assert COLOR_SAGE_GREEN == "#4A7C59"
    assert 'data-baseweb="tab"' in CUSTOM_CSS
    assert "#1E5631" in CUSTOM_CSS


def test_dashboard_title_contrast_and_header_styling():
    """Verify that the dashboard title displays the complete research topic with high contrast and balanced lines."""
    assert "h1.dashboard-header" in CUSTOM_CSS
    assert ".dashboard-header-sub" in CUSTOM_CSS
    assert "[data-testid=\"stMarkdownContainer\"] .dashboard-header" in CUSTOM_CSS
    assert "color: #FFFFFF !important" in CUSTOM_CSS
    # Header clearance to avoid clipping beneath fixed stHeader
    assert "padding-top: 2.4rem !important" in CUSTOM_CSS

    # Verify app.py renders balanced two-line complete research topic
    app_path = Path(dashboard_root) / "app.py"
    app_code = app_path.read_text(encoding="utf-8")
    assert '<h1 class="dashboard-header">' in app_code
    assert 'Data-Driven Decision Support System to Mitigate the “Cobweb Effect”' in app_code
    assert 'in Sri Lankan Vegetable Markets' in app_code

    # Verify source badges
    assert "HARTI Prices — Real Data" in app_code
    assert "NASA POWER Weather — Real Data" in app_code
    assert "DOA Commitment — Synthetic Proxy" in app_code


def test_sidebar_branding_and_broken_image_elimination():
    """Verify that sidebar uses professional research identity and old repo/member branding is completely removed."""
    from config import SIDEBAR_BRAND_TITLE, SIDEBAR_BRAND_SUBTITLE, SIDEBAR_BRAND_DESC

    app_path = Path(dashboard_root) / "app.py"
    app_code = app_path.read_text(encoding="utf-8")

    # st.image must NOT be present in app.py (no broken external/missing logos)
    assert "st.image" not in app_code

    # Old repo-oriented branding, student ID, and member ownership MUST be completely removed from UI code
    assert "SLIIT RESEARCH PROJECT" not in app_code
    assert "SLIIT-Research-Project-2026-27-DS-322" not in app_code
    assert "sidebar-brand-badge" not in app_code
    assert "sidebar-project-id" not in app_code
    assert "sidebar-member-role" not in app_code
    assert "Member 1" not in app_code
    assert "Member 2" not in app_code
    assert "Member 3" not in app_code

    # Verify new professional research identity elements
    assert "sidebar-brand-box" in app_code
    assert "sidebar-brand-title" in app_code
    assert "sidebar-brand-subtitle" in app_code
    assert "sidebar-brand-desc" in app_code

    # Verify semantic values
    assert SIDEBAR_BRAND_TITLE == "Geo-Spatial Decision Support System"
    assert SIDEBAR_BRAND_SUBTITLE == "Sri Lankan Vegetable Market Intelligence"
    assert SIDEBAR_BRAND_DESC == "Data-driven insights to understand and mitigate the Cobweb Effect in Sri Lankan vegetable markets."

    # Verify CSS rules exist for branding box
    assert ".sidebar-brand-box" in CUSTOM_CSS
    assert ".sidebar-brand-title" in CUSTOM_CSS
    assert ".sidebar-brand-subtitle" in CUSTOM_CSS
    assert ".sidebar-brand-desc" in CUSTOM_CSS



def test_kpi_metric_cards_laptop_alignment_and_flex_layout():
    """Verify KPI metric card HTML structure and height consistency for laptop viewports."""
    from components.metrics_cards import render_kpi_card_html

    card_html = render_kpi_card_html(
        title="Market Price (National Aggregate)",
        value="Rs. 185.0 / kg",
        sub_text="Latest weekly quoted price",
        border_color="#1E5631",
        badge_text="REAL HARTI",
        badge_class="badge-real",
    )

    # Must contain metric-card class
    assert 'class="metric-card"' in card_html
    # Must use flex container without float: right
    assert "float: right" not in card_html
    assert "display: flex" in card_html
    assert "justify-content: space-between" in card_html
    assert "REAL HARTI" in card_html

    # In CUSTOM_CSS, metric-card must enforce height: 118px !important for alignment
    assert "height: 118px !important" in CUSTOM_CSS
    assert ".metric-card" in CUSTOM_CSS


def test_streamlit_dark_theme_configuration():
    """Verify that .streamlit/config.toml enforces dark base theme and correct tokens."""
    config_toml_path = Path(dashboard_root) / ".streamlit" / "config.toml"
    assert config_toml_path.exists(), ".streamlit/config.toml must exist inside geospatial-dss-dashboard/"

    toml_content = config_toml_path.read_text(encoding="utf-8")
    assert 'base = "dark"' in toml_content
    assert 'primaryColor = "#4ADE80"' in toml_content
    assert 'backgroundColor = "#0A140D"' in toml_content
    assert 'textColor = "#FFFFFF"' in toml_content

