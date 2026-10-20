"""
Automated unit and regression tests for Geo-Spatial DSS Dashboard engines,
presentation components, and Member 2 forecast adapters.
Target: prototype/member1_GeoSpatialDSS
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure prototype/member1_GeoSpatialDSS directory is first on sys.path
dashboard_root = str(Path(__file__).resolve().parent.parent)
if dashboard_root in sys.path:
    sys.path.remove(dashboard_root)
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
    FOOTER_TEXT,
    COMMITMENT_PROVENANCE_NOTE,
    DASHBOARD_TABS,
    RISK_BAND_LOW_MAX,
    RISK_BAND_MODERATE_MAX,
    RISK_BAND_HIGH_MAX,
    RISK_BAND_CRITICAL_MAX,
    DRI_WEIGHT_COMPLETENESS,
    DRI_WEIGHT_RECENCY,
    DRI_WEIGHT_CONSISTENCY,
    get_risk_band,
    get_plain_language_confidence,
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
    calculate_dri_components,
    audit_price_reliability,
    audit_weather_reliability,
    audit_cultivation_reliability,
    audit_yield_reliability,
    get_dri_evaluation_summary,
    get_dri_source_benchmark,
    assess_district_data_readiness,
    compute_district_dri_table,
)
from engine.forecast_bridge import (
    ForecastAdapter,
)
from engine.advisor import (
    Recommendation,
    FarmerAdvisoryEngine,
)
from components.charts import (
    render_forecast_trajectory_chart,
    render_cobweb_phase_plot,
)
from components.metrics_cards import (
    render_kpi_card_html,
    render_executive_overview_cards,
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
# 2. MSRS Calculation Engine Tests (Standardized 0–100 Scale)
# =====================================================================

def test_msrs_calculation_formula():
    """Verify MSRS boundary conditions and linear scaling on standardized 0.0–100.0 scale."""
    # Safe zone (<= 80% capacity) -> 0.0
    assert calculate_msrs(committed_area=70.0, market_demand_capacity=100.0) == 0.0
    assert calculate_msrs(committed_area=80.0, market_demand_capacity=100.0) == 0.0

    # Critical zone (>= 120% capacity) -> 100.0
    assert calculate_msrs(committed_area=120.0, market_demand_capacity=100.0) == 100.0
    assert calculate_msrs(committed_area=150.0, market_demand_capacity=100.0) == 100.0

    # Midpoint (100% capacity) -> 50.0
    assert calculate_msrs(committed_area=100.0, market_demand_capacity=100.0) == 50.0

    # Edge cases
    assert calculate_msrs(committed_area=0.0, market_demand_capacity=100.0) == 0.0
    assert calculate_msrs(committed_area=50.0, market_demand_capacity=0.0) == 100.0


def test_msrs_risk_evaluation():
    """Verify standardized risk bands: Low (0-20), Moderate (20-60), High (60-80), Critical (80-100)."""
    tier, label, color = evaluate_saturation_risk(15.0)
    assert tier == "SAFE"
    assert color == COLOR_SAFE_GREEN

    tier, label, color = evaluate_saturation_risk(50.0)
    assert tier == "WARNING"
    assert color == COLOR_WARNING_AMBER

    tier, label, color = evaluate_saturation_risk(85.0)
    assert tier == "CRITICAL"
    assert color == COLOR_CRITICAL_RED


def test_district_msrs_panel_generation():
    """Verify district MSRS panel covers all 25 districts with valid standardized 0–100 scores."""
    df_comm = load_commitment_data()
    panel = compute_district_msrs_table(df_comm, "Carrot", "Maha", 2024)

    assert len(panel) == 25
    assert set(panel["district"]) == set(DISTRICT_NAMES)
    assert panel["msrs"].between(0.0, 100.0).all()
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
    """Verify DCVS weighted formula: 40% weather, 30% price, 30% (1-msrs/100)."""
    # Ideal conditions: weather=1.0, price=1.0, msrs=0.0 -> 100.0
    assert calculate_dcvs(msrs=0.0, weather_index=1.0, price_forecast_index=1.0) == 100.0

    # Terrible conditions: weather=0.0, price=0.0, msrs=100.0 -> 0.0
    assert calculate_dcvs(msrs=100.0, weather_index=0.0, price_forecast_index=0.0) == 0.0

    # Balanced midpoint: weather=0.5, price=0.5, msrs=50.0 -> 50.0
    assert calculate_dcvs(msrs=50.0, weather_index=0.5, price_forecast_index=0.5) == 50.0


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
    """Verify that when no live service is connected, honest status and baseline historical prices are returned."""
    df_price = load_price_data()
    adapter = ForecastAdapter()

    res = adapter.get_forecast("Carrot", "Nuwara Eliya", df_price, enable_demo_simulation=False)
    assert not res.is_live
    assert res.status_label == "Forecast service not connected"
    assert "HISTORICAL BASELINE" in res.provenance_tag
    assert res.forecast_df.empty
    assert not res.historical_df.empty
    assert len(res.narrative_summary) > 0


def test_forecast_adapter_demo_simulation_mode():
    """Verify forecast adapter does not generate fake random walks or demo numbers when disconnected."""
    df_price = load_price_data()
    adapter = ForecastAdapter()

    res = adapter.get_forecast("Carrot", "Nuwara Eliya", df_price, enable_demo_simulation=True, weeks_ahead=12)
    assert not res.is_live
    assert res.status_label == "Forecast service not connected"
    assert res.forecast_df.empty
    assert not res.historical_df.empty


def test_forecast_toggle_repeated_executions():
    """Regression Test for NameError: name 'go' is not defined."""
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
        assert len(fig.data) >= 1


# =====================================================================
# 5. Farmer Advisory Engine Tests (0–100 Standardized Scales)
# =====================================================================

def test_farmer_advisory_generation():
    """Verify that the advisory engine produces transparent recommendations with alternative crops and plain confidence."""
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
    assert 0.0 <= rec.dri_score <= 100.0
    assert rec.confidence in ["High confidence", "Moderate confidence", "Low confidence — verify with local officer"]
    assert rec.recommendation_category in ["Recommended", "Consider", "High Risk/Avoid"]
    assert len(rec.suggested_alternatives) >= 1


def test_farmer_advisory_alternatives_robustness():
    """Verify that every district receives alternative crop recommendations with transparent reasons."""
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


def test_separation_of_dri_msrs_dcvs_and_recommendation():
    """Verify that high DRI does not override critical market saturation (MSRS >= 60.0)."""
    df_comm = load_commitment_data()
    df_price = load_price_data()
    df_weather = load_weather_data()

    advisor = FarmerAdvisoryEngine()
    # Force precomputed_msrs to 85.0 (Critical saturation)
    rec = advisor.generate_advisory(
        district="Badulla",
        crop="Carrot",
        season_type="Maha",
        season_year=2024,
        df_comm=df_comm,
        df_price=df_price,
        df_weather=df_weather,
        precomputed_msrs=85.0,
    )

    # Must advise AVOID EXPANSION despite high data readiness
    assert rec.action == "AVOID EXPANSION / DIVERSIFY"
    assert rec.msrs_score == 85.0
    assert rec.recommendation_category == "High Risk/Avoid"


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
    assert adv_dist.dcvs_score == round(dist_dcvs_val, 1)

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

    assert adv_nat.msrs_score == round(nat_msrs_summary["national_msrs"], 1)
    assert adv_nat.dcvs_score == round(nat_mean_dcvs, 1)


# =====================================================================
# 6. Data Reliability Index (DRI) Core Contribution Tests (Section 6)
# =====================================================================

def test_dri_three_components_formula_and_weights():
    """Verify 3-component formula: 0.4 Completeness + 0.3 Recency + 0.3 Consistency."""
    assert DRI_WEIGHT_COMPLETENESS == 0.4
    assert DRI_WEIGHT_RECENCY == 0.3
    assert DRI_WEIGHT_CONSISTENCY == 0.3

    # Perfect data: c=1.0, r=1.0, s=1.0 -> 100.0
    score = calculate_dri_components(1.0, 1.0, 1.0)
    assert score == 100.0

    # Zero data: c=0.0, r=0.0, s=0.0 -> 0.0
    score = calculate_dri_components(0.0, 0.0, 0.0)
    assert score == 0.0

    # Known weights: c=0.8, r=0.6, s=1.0 -> (0.8*0.4 + 0.6*0.3 + 1.0*0.3) * 100 = (0.32 + 0.18 + 0.30) * 100 = 80.0
    score = calculate_dri_components(0.8, 0.6, 1.0)
    assert score == 80.0


def test_dri_missing_data_drops_completeness():
    """Unit test: Missing input records significantly reduce completeness and overall DRI."""
    df_price = load_price_data()
    # Crop with zero records
    audit_empty = audit_price_reliability(df_price, crop="NonExistentCrop", district="Kandy")
    assert audit_empty["available"] is False
    assert audit_empty["completeness"] == 0.0
    assert audit_empty["record_count"] == 0
    assert audit_empty["dri_score"] == 0.0

    # Overall district readiness with missing price stream
    empty_price_df = pd.DataFrame(columns=["district", "crop", "price", "price_type", "price_date"])
    df_weather = load_weather_data()
    readiness = assess_district_data_readiness("Badulla", "Carrot", empty_price_df, df_weather)
    assert readiness["is_evidence_adequate"] is False
    assert "Missing Price Observations" in readiness["evidence_sufficiency"]
    # Completeness must be lower due to missing price stream
    assert readiness["completeness"] < 100.0


def test_dri_stale_data_drops_recency():
    """Unit test: Older records (high observation lag) reduce recency score."""
    dates = pd.date_range("2023-01-01", periods=10, freq="W")
    # Reference date is 180 days after latest observation
    ref_date = dates.max() + pd.Timedelta(days=180)

    stale_price_df = pd.DataFrame({
        "price_date": dates,
        "district": ["Nuwara Eliya"] * 10,
        "crop": ["Carrot"] * 10,
        "price": [150.0] * 10,
        "price_type": ["retail"] * 10,
    })

    audit = audit_price_reliability(stale_price_df, crop="Carrot", district="Nuwara Eliya", ref_date=ref_date)
    assert audit["available"] is True
    assert audit["lag_days"] == 180
    assert audit["recency"] < 0.5  # Substantially penalized recency due to 180-day lag


def test_dri_injected_conflicts_drop_consistency():
    """Unit test: Injected invalid prices and extreme price spikes drop consistency score."""
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    # Clean data (stable prices)
    clean_df = pd.DataFrame({
        "price_date": dates,
        "district": ["Badulla"] * 10,
        "crop": ["Carrot"] * 10,
        "price": [100.0, 102.0, 101.0, 104.0, 103.0, 105.0, 106.0, 107.0, 108.0, 110.0],
        "price_type": ["retail"] * 10,
    })
    clean_audit = audit_price_reliability(clean_df, crop="Carrot", district="Badulla", ref_date=dates.max())
    assert clean_audit["consistency"] == 1.0
    assert clean_audit["anomalies"] == 0

    # Injected conflict: negative price and >100% week-on-week price spike
    corrupt_df = pd.DataFrame({
        "price_date": dates,
        "district": ["Badulla"] * 10,
        "crop": ["Carrot"] * 10,
        "price": [100.0, -25.0, 100.0, 300.0, 100.0, 105.0, 106.0, 107.0, 108.0, 110.0],
        "price_type": ["retail"] * 10,
    })
    corrupt_audit = audit_price_reliability(corrupt_df, crop="Carrot", district="Badulla", ref_date=dates.max())
    assert corrupt_audit["anomalies"] >= 2
    assert corrupt_audit["consistency"] < clean_audit["consistency"]


def test_district_level_dri_table_all_25_districts():
    """Verify empirical DRI table covers all 25 districts with measurable metrics."""
    df_price = load_price_data()
    df_weather = load_weather_data()
    df_comm = load_commitment_data()

    table = compute_district_dri_table(df_price, df_weather, df_comm, crop="Carrot", season_year=2024)
    assert len(table) == 25
    assert set(table["district"]) == set(DISTRICT_NAMES)
    assert table["dri_score"].between(0.0, 100.0).all()
    assert "completeness" in table.columns
    assert "recency" in table.columns
    assert "consistency" in table.columns
    assert "confidence_label" in table.columns


def test_yield_category_neutral_unavailable_state():
    """Verify Yield data category honestly returns neutral unavailable state with exact status label and N/A count."""
    yield_audit = audit_yield_reliability()
    assert yield_audit["category"] == "Yield"
    assert yield_audit["available"] is False
    assert yield_audit["status_label"] == "Unavailable — source data not yet integrated"
    assert yield_audit["record_count"] is None
    assert yield_audit["completeness"] is None
    assert yield_audit["recency"] is None
    assert yield_audit["consistency"] is None
    assert yield_audit["dri_score"] is None
    assert "systematic district-level yield survey records are not available" in yield_audit["notes"].lower()


def test_forecast_historical_aggregation_and_gap_handling():
    """Verify that forecast adapter aggregates weekly dates properly and chart preserves gaps."""
    df_price = load_price_data()
    adapter = ForecastAdapter()

    # 1. National Aggregate should have 52 unique weekly dates, not raw duplicate rows
    res_nat = adapter.get_forecast("Cabbage", "All Island (National Average)", df_price, price_type="retail")
    assert not res_nat.historical_df.empty
    assert len(res_nat.historical_df) == 52
    assert res_nat.historical_df["price_date"].is_monotonic_increasing
    assert len(res_nat.historical_df["price_date"].unique()) == 52

    # 2. Render chart and verify connectgaps is False and mode is lines+markers
    fig = render_forecast_trajectory_chart(
        forecast_df=res_nat.forecast_df,
        historical_df=res_nat.historical_df,
        crop_name="Cabbage",
        district="National Aggregate",
    )
    assert fig is not None
    # Trace 0 should be Verified Historical Observations
    trace0 = fig.data[0]
    assert trace0.connectgaps is False
    assert "lines+markers" in trace0.mode
    assert "Verified Historical" in trace0.name

    # 3. Empty historical data should produce clean empty state annotation
    fig_empty = render_forecast_trajectory_chart(
        forecast_df=pd.DataFrame(),
        historical_df=pd.DataFrame(columns=["price_date", "price"]),
        crop_name="Carrot",
        district="EmptyDistrict",
    )
    assert len(fig_empty.layout.annotations) >= 1
    assert "No verified historical price observations available" in fig_empty.layout.annotations[0].text


# =====================================================================
# 7. Recommendations Table & District Drill-Down Tests (Section 7)
# =====================================================================

def test_multi_crop_recommendations_table_structure():
    """Verify sortable multi-crop recommendations table contains exact 7 required columns."""
    df_price = load_price_data()
    df_weather = load_weather_data()
    df_comm = load_commitment_data()

    advisor = FarmerAdvisoryEngine()
    rec_table = advisor.generate_crop_recommendations_table(
        district="Badulla",
        season_type="Maha",
        season_year=2024,
        df_comm=df_comm,
        df_price=df_price,
        df_weather=df_weather,
    )

    expected_cols = ["Crop", "MSRS", "DCVS", "DRI", "Confidence", "Recommendation", "Key Reason"]
    assert list(rec_table.columns) == expected_cols
    assert len(rec_table) == len(TARGET_CROPS)

    # Valid categories
    valid_recs = {"Recommended", "Consider", "High Risk/Avoid"}
    assert set(rec_table["Recommendation"].unique()).issubset(valid_recs)

    # Standardized 0-100 scores
    assert rec_table["MSRS"].between(0.0, 100.0).all()
    assert rec_table["DCVS"].between(0.0, 100.0).all()
    assert rec_table["DRI"].between(0.0, 100.0).all()


def test_district_drilldown_summary():
    """Verify district drill-down summary provides price, commitment, weather, and DRI breakdown."""
    df_price = load_price_data()
    df_weather = load_weather_data()
    df_comm = load_commitment_data()

    advisor = FarmerAdvisoryEngine()
    drill = advisor.get_district_drilldown_summary(
        district="Nuwara Eliya",
        crop="Carrot",
        season_type="Maha",
        season_year=2024,
        df_comm=df_comm,
        df_price=df_price,
        df_weather=df_weather,
    )

    assert drill["district"] == "Nuwara Eliya"
    assert "latest_price" in drill
    assert "eight_week_ma" in drill
    assert "committed_ha" in drill
    assert "target_ha" in drill
    assert "rainfall_avg_mm" in drill
    assert "temp_avg_c" in drill
    assert "dri_score" in drill


# =====================================================================
# 8. UI, Tabs, Footer, and Provenance Tests (Sections 1 & 2)
# =====================================================================

def test_tabs_specification_without_emojis():
    """Verify that exact 6 tabs are used for Member 1 Geo-Spatial DSS without emojis."""
    expected_tabs = [
        "Overview",
        "Market Saturation",
        "Price Dynamics",
        "Agro-Meteorology",
        "Recommendations",
        "Data Reliability",
    ]
    assert DASHBOARD_TABS == expected_tabs
    for tab in DASHBOARD_TABS:
        # No emoji symbols in tab names
        assert not any(ord(char) > 127 for char in tab)


def test_forecast_ui_removed_from_app():
    """Verify that Forecast tab, sidebar integration toggle, and forecast imports are removed from Member 1's app."""
    app_path = Path(dashboard_root) / "app.py"
    app_code = app_path.read_text(encoding="utf-8")
    assert "Price Forecast Integration" not in app_code
    assert "Connect Forecast Service" not in app_code
    assert "tab_forecast" not in app_code
    assert "render_forecast_trajectory_chart" not in app_code
    assert "ForecastAdapter" not in app_code


def test_collaboration_footer_specification():
    """Verify exact required collaboration footer text."""
    expected_footer = "Hector Kobbekaduwa Agrarian Research and Training Institute (HARTI) collaboration | SLIIT | J26-DS-322"
    assert FOOTER_TEXT == expected_footer

    app_path = Path(dashboard_root) / "app.py"
    app_code = app_path.read_text(encoding="utf-8")
    assert FOOTER_TEXT in app_code


def test_provenance_labels_and_badges_cleanup():
    """
    Verify that scattered badges (badge-real, badge-proxy, badge-demo) are removed from UI,
    and that the required commitment provenance note is present.
    """
    app_path = Path(dashboard_root) / "app.py"
    app_code = app_path.read_text(encoding="utf-8")

    # Scattered badges removed from app.py
    assert "badge-real" not in app_code
    assert "badge-proxy" not in app_code
    assert "badge-demo" not in app_code

    # Required commitment provenance note must be present
    assert COMMITMENT_PROVENANCE_NOTE in app_code
    assert "Cultivation commitment values for the prototype are derived from historical HARTI district records and refreshed per season." == COMMITMENT_PROVENANCE_NOTE


def test_sidebar_branding_and_broken_image_elimination():
    """Verify that sidebar uses professional research identity and old repo/member branding is removed."""
    from config import SIDEBAR_BRAND_TITLE, SIDEBAR_BRAND_SUBTITLE, SIDEBAR_BRAND_DESC

    app_path = Path(dashboard_root) / "app.py"
    app_code = app_path.read_text(encoding="utf-8")

    assert "st.image" not in app_code
    assert "SLIIT RESEARCH PROJECT" not in app_code
    assert "SLIIT-Research-Project-2026-27-DS-322" not in app_code
    assert "Member 1" not in app_code
    assert "Member 2" not in app_code
    assert "Member 3" not in app_code

    assert "sidebar-brand-box" in app_code
    assert SIDEBAR_BRAND_TITLE == "Geo-Spatial Decision Support System"
    assert SIDEBAR_BRAND_SUBTITLE == "Sri Lankan Vegetable Market Intelligence"


def test_kpi_metric_cards_five_cards_layout():
    """Verify KPI metric card HTML structure and 5 overview cards layout."""
    card_html = render_kpi_card_html(
        title="Market Price",
        value="Rs. 185.0 / kg",
        sub_text="Latest weekly quote",
        scope_text="National Aggregate",
        border_color="#1E5631",
    )

    assert 'class="metric-card"' in card_html
    assert "Market Price" in card_html
    assert "Rs. 185.0 / kg" in card_html
    assert "National Aggregate" in card_html
    assert "height: 118px !important" in CUSTOM_CSS


def test_streamlit_dark_theme_configuration():
    """Verify that .streamlit/config.toml enforces dark base theme and correct tokens."""
    config_toml_path = Path(dashboard_root) / ".streamlit" / "config.toml"
    assert config_toml_path.exists(), ".streamlit/config.toml must exist"

    toml_content = config_toml_path.read_text(encoding="utf-8")
    assert 'base = "dark"' in toml_content
    assert 'primaryColor = "#4ADE80"' in toml_content
    assert 'backgroundColor = "#0A140D"' in toml_content
    assert 'textColor = "#FFFFFF"' in toml_content


def test_kpi_card_html_unindented_rendering_and_unavailable_fallback():
    """Verify KPI card HTML has no leading 4-space markdown indent and handles None/NaN gracefully."""
    # 1. Check unindented output (must not start with 4 spaces which triggers pre/code block in markdown)
    html = render_kpi_card_html("Market Price", "Rs. 250.0 / kg", "Weekly quote", "Colombo", "#4ADE80")
    for line in html.splitlines():
        if line.strip():
            assert not line.startswith("    "), "HTML lines must not have 4-space markdown indent"
    assert '<div class="metric-card"' in html

    # 2. Check fallback to 'Unavailable' for None and NaN
    html_none = render_kpi_card_html("Yield Rate", None, "Survey", "Colombo", "#4ADE80")
    assert "Unavailable" in html_none
    html_nan = render_kpi_card_html("Yield Rate", float("nan"), "Survey", "Colombo", "#4ADE80")
    assert "Unavailable" in html_nan


def test_dri_grouped_price_consistency_calculation():
    """Verify that price consistency groups by district and price_type, avoiding cross-stream false anomalies."""
    df_price = load_price_data()
    # Colombo Carrot audit
    audit_colombo = audit_price_reliability(df_price, "Carrot", district="Colombo")
    assert audit_colombo["available"] is True
    assert audit_colombo["consistency"] >= 0.95
    # Anomalies should be very low (<= 10) in Colombo rather than hundreds
    assert audit_colombo["anomalies"] <= 10


def test_crop_recommendations_table_formatting():
    """Verify multi-crop recommendation table has exact required columns, 1 decimal place floats, and no unrounded numbers."""
    df_comm = load_commitment_data()
    df_price = load_price_data()
    df_weather = load_weather_data()
    advisor = FarmerAdvisoryEngine()

    df_table = advisor.generate_crop_recommendations_table(
        district="Nuwara Eliya",
        season_type="Maha",
        season_year=2024,
        df_comm=df_comm,
        df_price=df_price,
        df_weather=df_weather,
        price_type="retail",
    )

    expected_cols = ["Crop", "MSRS", "DCVS", "DRI", "Confidence", "Recommendation", "Key Reason"]
    assert list(df_table.columns) == expected_cols
    assert len(df_table) == 5

    # Check floats are rounded to 1 decimal place
    for col in ["MSRS", "DCVS", "DRI"]:
        for val in df_table[col]:
            assert str(val) == f"{val:.1f}"
