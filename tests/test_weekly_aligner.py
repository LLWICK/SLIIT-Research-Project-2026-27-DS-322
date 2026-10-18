"""
Automated test suite for the Weekly Temporal Alignment Pipeline.

Tests schema contracts, canonical 5-crop vocabulary enforcement, alias normalization,
Big Onion origin isolation, ambiguous commodity quarantine, Monday-to-Sunday ISO calendar
windowing, daily weather sum vs. mean aggregations, market granularity preservation,
incomplete trading week handling, zero-fabrication contextual linking, and prototype compatibility.
"""

import os
import json
import datetime
import pytest
import pandas as pd
import numpy as np

from pipeline.alignment.schemas import (
    validate_dataframe,
    CLEAN_PRICE_REQUIRED_COLUMNS,
    CLEAN_WEATHER_REQUIRED_COLUMNS,
    CLEAN_DOA_REQUIRED_COLUMNS,
    CLEAN_DCS_REQUIRED_COLUMNS,
)
from pipeline.alignment.crop_vocab import (
    standardize_crop,
    CropVocabularyEnforcer,
    CANONICAL_TARGET_CROPS,
)
from pipeline.alignment.calendar_utils import (
    get_iso_week_metadata,
    get_agricultural_season,
    add_iso_week_columns,
    parse_date,
)
from pipeline.alignment.weather_aggregator import WeatherAggregator
from pipeline.alignment.price_aggregator import PriceAggregator
from pipeline.alignment.weekly_aligner import WeeklyAligner
from pipeline.alignment.adapters import (
    export_to_prototype_v1_format,
    compute_weekly_dri_status,
    check_dri_calculator_available,
)


# =====================================================================
# 1. Schema Validation Tests
# =====================================================================

def test_schema_validation_passes_valid_data():
    """Verify that schema validation succeeds for properly formed DataFrames."""
    valid_price_df = pd.DataFrame({
        "obs_date": ["2024-01-01"],
        "district": ["Colombo"],
        "market_name": ["Pettah"],
        "crop_name": ["Carrot"],
        "price_type": ["retail"],
        "price_lkr_per_kg": [250.0],
    })
    validate_dataframe(valid_price_df, CLEAN_PRICE_REQUIRED_COLUMNS, "CleanPrice")

    valid_weather_df = pd.DataFrame({
        "obs_date": ["2024-01-01"],
        "district": ["Badulla"],
        "precipitation_mm": [12.5],
        "temp_mean_c": [22.4],
    })
    validate_dataframe(valid_weather_df, CLEAN_WEATHER_REQUIRED_COLUMNS, "CleanWeather")


def test_schema_validation_fails_missing_columns():
    """Verify that missing required columns raise descriptive ValueErrors."""
    invalid_price_df = pd.DataFrame({
        "obs_date": ["2024-01-01"],
        "district": ["Colombo"],
        # Missing market_name, crop_name, price_type, price_lkr_per_kg
    })
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataframe(invalid_price_df, CLEAN_PRICE_REQUIRED_COLUMNS, "CleanPrice")


def test_schema_validation_fails_empty_df():
    """Verify that empty DataFrames raise ValueErrors when allow_empty is False."""
    empty_df = pd.DataFrame(columns=CLEAN_PRICE_REQUIRED_COLUMNS)
    with pytest.raises(ValueError, match="is empty"):
        validate_dataframe(empty_df, CLEAN_PRICE_REQUIRED_COLUMNS, "CleanPrice", allow_empty=False)


# =====================================================================
# 2. Crop Vocabulary & Alias Normalization Tests
# =====================================================================

def test_crop_alias_normalization():
    """Verify that crop aliases normalize strictly to canonical target names."""
    # Leek alias: plural 'Leeks' must map to singular 'Leek'
    c, origin, quarantined, _ = standardize_crop("Leeks")
    assert c == "Leek"
    assert not quarantined

    c, _, _, _ = standardize_crop("leek")
    assert c == "Leek"

    # Cabbage regional varieties
    c, _, _, _ = standardize_crop("Cabbage (N'Eliya)")
    assert c == "Cabbage"

    c, _, _, _ = standardize_crop("Cabbage (Kandy)")
    assert c == "Cabbage"

    # Carrot and Tomato plurals
    c, _, _, _ = standardize_crop("Carrots")
    assert c == "Carrot"

    c, _, _, _ = standardize_crop("Tomatoes")
    assert c == "Tomato"


def test_target_crop_scope_filtering():
    """Verify that non-target crops (Beans, Raddish, Brinjal, etc.) are excluded."""
    non_target_samples = [
        "Beans", "Green Beans", "Raddish", "White Raddish",
        "Beetroot", "Brinjal", "Pumpkin", "Ladies Fingers",
        "Knol Khol", "Capsicum", "Potato", "Cassava"
    ]
    for sample in non_target_samples:
        canonical, _, is_quarantined, _ = standardize_crop(sample)
        assert canonical is None, f"Crop '{sample}' should not be in the 5 canonical research crops."
        assert not is_quarantined


def test_ambiguous_onion_quarantined(tmp_path):
    """Verify that ambiguous 'Onion' terms are NEVER mapped to Big Onion and are quarantined."""
    quarantine_log = str(tmp_path / "quarantine.jsonl")
    enforcer = CropVocabularyEnforcer(quarantine_sink_path=quarantine_log)

    test_df = pd.DataFrame({
        "crop_name": ["Carrot", "Onion", "Onions", "Local Onion", "Tomato"],
        "source_commodity_name": ["Carrot", "Onion", "Imported Onions", None, "Tomato"],
        "price_lkr_per_kg": [200.0, 150.0, 160.0, 140.0, 300.0],
    })

    filtered = enforcer.filter_and_standardize(test_df)

    # Output DataFrame must contain ONLY Carrot and Tomato
    assert set(filtered["crop_name"].unique()) == {"Carrot", "Tomato"}
    assert "Onion" not in filtered["crop_name"].values
    assert "Big Onion" not in filtered["crop_name"].values

    # Check quarantine tracking
    assert len(enforcer.quarantined_records) == 3
    assert os.path.exists(quarantine_log)

    with open(quarantine_log, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 3
    assert "Ambiguous commodity" in lines[0]["reason"] or "Unresolved onion" in lines[0]["reason"]


def test_red_onion_excluded_not_mapped_to_big_onion():
    """Verify that Red Onion / Shallots are excluded and never conflated with Big Onion."""
    red_onion_variants = [
        "Red Onion", "Red-onion", "Red Onion (Vedalan)", "Red Onion (Sindo)", "Shallot"
    ]
    for variant in red_onion_variants:
        canonical, _, is_quarantined, _ = standardize_crop(variant)
        assert canonical is None, f"'{variant}' must be excluded from target crops."
        assert not is_quarantined


def test_big_onion_local_vs_imported_separation():
    """Verify that Big Onion Local and Imported quotes remain separate and isolated."""
    c_local, origin_local, _, _ = standardize_crop("Big Onion", "Big-onion Local")
    assert c_local == "Big Onion"
    assert origin_local == "Local"

    c_imp, origin_imp, _, _ = standardize_crop("Big Onion", "B'Onion Imported")
    assert c_imp == "Big Onion"
    assert origin_imp == "Imported"

    c_local2, origin_local2, _, _ = standardize_crop("Big Onion (Local)")
    assert c_local2 == "Big Onion"
    assert origin_local2 == "Local"

    c_imp2, origin_imp2, _, _ = standardize_crop("Big Onion (Imp)")
    assert c_imp2 == "Big Onion"
    assert origin_imp2 == "Imported"


# =====================================================================
# 3. Calendar & Seasonal Windowing Tests
# =====================================================================

def test_iso_calendar_monday_to_sunday():
    """Verify that all dates within an ISO week map to the same Monday start and Sunday end."""
    # Week 1 of 2024: Jan 1 (Mon) to Jan 7 (Sun)
    for day in range(1, 8):
        dt_str = f"2024-01-0{day}"
        meta = get_iso_week_metadata(dt_str)
        assert meta["week_start_date"] == "2024-01-01", f"Failed on {dt_str}"
        assert meta["week_end_date"] == "2024-01-07", f"Failed on {dt_str}"
        assert meta["iso_year"] == 2024
        assert meta["iso_week"] == 1


def test_iso_year_end_boundary_rollover():
    """Verify proper boundary handling when ISO year differs from calendar year."""
    # 2024-12-30 (Monday): Thursday is Jan 2 2025, so this week belongs to ISO year 2025 Week 1
    meta_w01 = get_iso_week_metadata("2024-12-30")
    assert meta_w01["week_start_date"] == "2024-12-30"
    assert meta_w01["week_end_date"] == "2025-01-05"
    assert meta_w01["iso_year"] == 2025
    assert meta_w01["iso_week"] == 1

    # 2020-12-31 (Thursday): Belongs to ISO year 2020 Week 53 (leap week year)
    meta_w53 = get_iso_week_metadata("2020-12-31")
    assert meta_w53["week_start_date"] == "2020-12-28"
    assert meta_w53["week_end_date"] == "2021-01-03"
    assert meta_w53["iso_year"] == 2020
    assert meta_w53["iso_week"] == 53

    # 2023-01-01 (Sunday): Belongs to ISO year 2022 Week 52
    meta_prev = get_iso_week_metadata("2023-01-01")
    assert meta_prev["week_start_date"] == "2022-12-26"
    assert meta_prev["week_end_date"] == "2023-01-01"
    assert meta_prev["iso_year"] == 2022
    assert meta_prev["iso_week"] == 52


def test_agricultural_season_mapping():
    """Verify Sri Lankan agricultural season windowing (Yala: Apr-Sep; Maha: Oct-Mar)."""
    # Yala
    yala_meta = get_agricultural_season("2024-05-15")
    assert yala_meta["season_type"] == "Yala"
    assert yala_meta["season_year"] == 2024

    # Maha early phase (Oct - Dec)
    maha_early = get_agricultural_season("2024-11-20")
    assert maha_early["season_type"] == "Maha"
    assert maha_early["season_year"] == 2024

    # Maha late phase (Jan - Mar)
    maha_late = get_agricultural_season("2025-02-10")
    assert maha_late["season_type"] == "Maha"
    assert maha_late["season_year"] == 2024  # Base season year is 2024 for Maha 2024/25


# =====================================================================
# 4. Weather Aggregation Tests
# =====================================================================

def test_weather_sum_rainfall_and_average_temperature():
    """Verify that rainfall is strictly summed and temperatures/humidity are averaged."""
    aggregator = WeatherAggregator()

    daily_weather = pd.DataFrame({
        "obs_date": [
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
            "2024-01-05", "2024-01-06", "2024-01-07"
        ],
        "district": ["Nuwara Eliya"] * 7,
        "precipitation_mm": [10.0, 5.5, 0.0, 20.0, 0.0, 4.5, 15.0],
        "temp_mean_c": [18.0, 19.0, 17.5, 18.5, 19.5, 17.0, 18.0],
        "relative_humidity_pct": [80.0, 85.0, 75.0, 90.0, 78.0, 82.0, 88.0],
    })

    weekly_w = aggregator.aggregate(daily_weather)

    assert len(weekly_w) == 1
    row = weekly_w.iloc[0]

    # Expected rainfall: sum = 55.0 mm
    assert row["weekly_precip_sum_mm"] == 55.0
    # Expected temperature: mean = (18+19+17.5+18.5+19.5+17+18)/7 = 18.21 C
    assert row["weekly_temp_mean_c"] == 18.21
    # Expected humidity: mean = 82.57%
    assert row["weekly_humidity_mean_pct"] == 82.57
    # Observation count must be 7
    assert row["weather_obs_count"] == 7
    assert row["week_start_date"] == "2024-01-01"


# =====================================================================
# 5. Price Aggregation & Market Separation Tests
# =====================================================================

def test_price_market_granularity_and_incomplete_weeks(tmp_path):
    """Verify that distinct markets remain separate and incomplete trading weeks are counted accurately."""
    quarantine_sink = str(tmp_path / "price_quarantine.jsonl")
    aggregator = PriceAggregator(quarantine_sink_path=quarantine_sink)

    daily_prices = pd.DataFrame({
        "obs_date": [
            # Pettah: 3 trading days
            "2024-01-01", "2024-01-03", "2024-01-05",
            # Dambulla: 5 trading days
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        ],
        "district": ["Colombo", "Colombo", "Colombo", "Matale", "Matale", "Matale", "Matale", "Matale"],
        "market_name": ["Pettah", "Pettah", "Pettah", "Dambulla", "Dambulla", "Dambulla", "Dambulla", "Dambulla"],
        "crop_name": ["Carrot"] * 8,
        "price_type": ["wholesale"] * 8,
        "price_lkr_per_kg": [200.0, 220.0, 210.0, 180.0, 185.0, 190.0, 175.0, 190.0],
    })

    weekly_p = aggregator.aggregate(daily_prices)

    # Must preserve 2 separate market rows
    assert len(weekly_p) == 2
    pettah_row = weekly_p[weekly_p["market_name"] == "Pettah"].iloc[0]
    dambulla_row = weekly_p[weekly_p["market_name"] == "Dambulla"].iloc[0]

    # Check observation counts (incomplete weeks)
    assert pettah_row["price_obs_count"] == 3
    assert dambulla_row["price_obs_count"] == 5

    # Check computed statistics
    assert pettah_row["weekly_price_mean"] == 210.0
    assert pettah_row["weekly_price_min"] == 200.0
    assert pettah_row["weekly_price_max"] == 220.0

    assert dambulla_row["weekly_price_mean"] == 184.0


# =====================================================================
# 6. Contextual Macro-Linking Tests (Zero Weekly Fabrication)
# =====================================================================

def test_zero_fabrication_contextual_macro_linking():
    """Verify that annual and seasonal figures are attached as whole contextual numbers without weekly division."""
    aligner = WeeklyAligner()

    daily_price = pd.DataFrame({
        "obs_date": ["2024-05-06", "2024-05-08"],
        "district": ["Matale", "Matale"],
        "market_name": ["Dambulla", "Dambulla"],
        "crop_name": ["Big Onion (Local)", "Big Onion (Local)"],
        "price_type": ["wholesale", "wholesale"],
        "price_lkr_per_kg": [250.0, 260.0],
    })

    daily_weather = pd.DataFrame({
        "obs_date": ["2024-05-06", "2024-05-08"],
        "district": ["Matale", "Matale"],
        "precipitation_mm": [5.0, 10.0],
        "temp_mean_c": [28.0, 27.5],
    })

    # Seasonal DOA targets: 2500 hectares target for Yala 2024
    doa_targets = pd.DataFrame({
        "season": ["Yala"],
        "year": [2024],
        "district": ["Matale"],
        "crop": ["Big Onion"],
        "target_extent_ha": [2500.0],
        "cultivated_extent_ha": [2100.0],
        "expected_production_mt": [32000.0],
    })

    # Annual DCS yield baseline: 14500 kg/ha
    dcs_baselines = pd.DataFrame({
        "district": ["Matale"],
        "crop": ["Big Onion"],
        "metric": ["Average Yield (kg/ha)"],
        "value": [14500.0],
        "unit": ["kg_per_hectare"],
    })

    # Step 1: Align weather and prices
    weekly_base = aligner.align_price_and_weather(daily_price, daily_weather)
    assert not weekly_base.empty

    # Step 2: Link contextual statistics
    enriched = aligner.link_contextual_statistics(weekly_base, clean_doa_df=doa_targets, clean_dcs_df=dcs_baselines)

    assert len(enriched) == 1
    row = enriched.iloc[0]

    # Verify that the contextual figures are EXACTLY as published by DOA/DCS
    # (STRICT ASSERTION: Not divided by 52 or fabricated into fake weekly harvests)
    assert row["doa_target_extent_ha"] == 2500.0
    assert row["doa_cultivated_extent_ha"] == 2100.0
    assert row["doa_expected_production_mt"] == 32000.0
    assert row["dcs_baseline_yield_kg_ha"] == 14500.0

    # Ensure no division happened (e.g., 2500 / 52 approx 48.07)
    assert row["doa_target_extent_ha"] != pytest.approx(48.07, rel=0.1)


# =====================================================================
# 7. District Rollup & Prototype Compatibility Tests
# =====================================================================

def test_district_rollup_and_prototype_v1_export():
    """Verify that district rollup and prototype v1 export bridge match analysis_ready_price.csv schema."""
    aligner = WeeklyAligner()

    granular_df = pd.DataFrame({
        "district": ["Matale", "Matale"],
        "market_name": ["Dambulla", "Galewela"],
        "crop_name": ["Big Onion", "Big Onion"],
        "origin_variety": ["Local", "Local"],
        "price_type": ["retail", "retail"],
        "iso_year": [2024, 2024],
        "iso_week": [19, 19],
        "week_start_date": ["2024-05-06", "2024-05-06"],
        "week_end_date": ["2024-05-12", "2024-05-12"],
        "active_season": ["Yala", "Yala"],
        "season_year": [2024, 2024],
        "weekly_price_mean": [280.0, 300.0],
        "weekly_price_min": [270.0, 290.0],
        "weekly_price_max": [290.0, 310.0],
        "price_obs_count": [4, 3],
        "source_agency": ["HARTI", "CBSL"],
    })

    district_rollup = aligner.create_district_rollup(granular_df)

    assert len(district_rollup) == 1
    d_row = district_rollup.iloc[0]
    assert d_row["weekly_price_mean"] == 290.0  # mean of 280 and 300
    assert d_row["market_count"] == 2
    assert d_row["price_obs_count"] == 7  # 4 + 3

    # Test prototype v1 export adapter
    proto_df = aligner.export_prototype_dataset(district_rollup, price_type="retail")

    expected_cols = ["price_date", "district", "crop", "price_type", "granularity", "price", "source"]
    assert list(proto_df.columns) == expected_cols

    p_row = proto_df.iloc[0]
    assert p_row["price_date"] == "2024-05-06"  # Monday ISO anchor
    assert p_row["district"] == "Matale"
    # STRICT COMPATIBILITY: Must map Big Onion Local to 'Big Onion (Local)' matching legacy prototype
    assert p_row["crop"] == "Big Onion (Local)"
    assert p_row["granularity"] == "Retail - W"
    assert p_row["price"] == 290.0


# =====================================================================
# 8. Dynamic Adapters & Zero-Fabrication DRI Handling Tests
# =====================================================================

def test_adapters_dri_availability_handling():
    """Verify that the adapter gracefully reports integration pending without fabricating fake DRI scores."""
    sample_df = pd.DataFrame({"sample_col": [1, 2, 3]})
    dri_status = compute_weekly_dri_status(sample_df)

    # Since the price/DRI PR is not yet merged into this branch, it must report pending
    if not check_dri_calculator_available():
        assert dri_status["dri_available"] is False
        assert "pending" in dri_status["status"].lower()
        # MUST NOT fabricate a numerical score
        assert dri_status["dri_score"] is None
        assert "Unverified" in dri_status["quality_tier"]
    else:
        assert dri_status["dri_available"] is True
