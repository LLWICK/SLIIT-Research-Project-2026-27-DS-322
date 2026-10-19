"""
Dynamic Crop Viability Score (DCVS) Calculation Engine.

Member 3 Analytical Bridge:
Calculates a holistic recommendation score (0% to 100%) that proactively balances
agro-climatic suitability, price trend incentives, and market saturation risk.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Ensure dashboard root is on sys.path
_dashboard_root = str(Path(__file__).resolve().parent.parent)
if _dashboard_root not in sys.path:
    sys.path.insert(0, _dashboard_root)

from config import (
    DISTRICT_COORDS,
    DISTRICT_NAMES,
    BASE_REFERENCE_PRICES,
)
from engine.msrs_engine import (
    compute_district_msrs_table,
)


def calculate_dcvs(
    msrs: float,
    weather_index: float,
    price_forecast_index: float,
    weather_weight: float = 0.4,
    price_weight: float = 0.3,
    saturation_weight: float = 0.3,
) -> float:
    """
    Calculate the Dynamic Crop Viability Score (DCVS) (0 to 100%).

    Formula (Member 3 Standard):
    - Saturation factor: (1.0 - msrs) [lower saturation increases viability]
    - Raw score: (weather_index * 0.4) + (price_index * 0.3) + ((1 - msrs) * 0.3)
    - Scale: 0 to 100%

    Args:
        msrs: Market Saturation Risk Score (0.0 to 1.0).
        weather_index: Agro-meteorological suitability index (0.0 to 1.0).
        price_forecast_index: Price incentive / trend index (0.0 to 1.0).
        weather_weight: Weight for weather suitability (default 0.4).
        price_weight: Weight for price incentive (default 0.3).
        saturation_weight: Weight for market saturation (default 0.3).

    Returns:
        DCVS percentage between 0.0 and 100.0.
    """
    msrs_clamped = min(max(msrs, 0.0), 1.0)
    weather_clamped = min(max(weather_index, 0.0), 1.0)
    price_clamped = min(max(price_forecast_index, 0.0), 1.0)

    saturation_score = 1.0 - msrs_clamped

    dcvs_raw = (
        (weather_clamped * weather_weight) +
        (price_clamped * price_weight) +
        (saturation_score * saturation_weight)
    )

    return round(dcvs_raw * 100.0, 1)


def evaluate_viability_tier(dcvs: float) -> Tuple[str, str, str]:
    """
    Evaluate qualitative viability tier, descriptive label, and display color.

    Returns:
        Tuple of (tier_name, descriptive_label, hex_color).
    """
    if dcvs >= 70.0:
        return "HIGH", "High Viability (Favorable Conditions)", "#2E7D32"
    elif dcvs >= 45.0:
        return "MODERATE", "Moderate Viability (Monitor Quota)", "#D97706"
    else:
        return "LOW", "Low Viability (High Crash Risk)", "#C53030"


def compute_district_dcvs_table(
    df_comm: pd.DataFrame,
    df_price: pd.DataFrame,
    df_weather: pd.DataFrame,
    crop_name: str,
    season_type: str,
    season_year: int,
    price_type: str = "retail",
) -> pd.DataFrame:
    """
    Compute DCVS scores for all 25 districts combining real price momentum,
    real weather suitability, and seasonal commitment saturation.

    Args:
        df_comm: Commitment dataset.
        df_price: Weekly price dataset.
        df_weather: Daily weather dataset.
        crop_name: Selected vegetable crop.
        season_type: 'Maha' or 'Yala'.
        season_year: Planning year.
        price_type: 'retail' or 'wholesale'. Defaults to 'retail'.

    Returns:
        DataFrame containing DCVS, MSRS, component indices, and coordinates.
    """
    # Step 1: Compute MSRS table for districts
    df_msrs = compute_district_msrs_table(df_comm, crop_name, season_type, season_year)

    # Step 2: Compute recent price index from real price data matching quotation level
    ref_price = BASE_REFERENCE_PRICES.get(crop_name, 120.0)
    
    price_sub = df_price[
        (df_price["crop"] == crop_name) & 
        (df_price["price_type"] == price_type)
    ]
    if price_sub.empty:
        price_sub = df_price[df_price["crop"] == crop_name]

    recent_prices = (
        price_sub.groupby("district")["price"]
        .last()
        .to_dict()
    )

    # Step 3: Compute weather suitability index per district
    recent_weather = (
        df_weather.groupby("district")
        .agg({"rainfall_mm": "mean", "temp_c": "mean"})
        .to_dict(orient="index")
    )

    records = []
    for _, row in df_msrs.iterrows():
        district = row["district"]
        msrs_val = row["msrs"]

        # Derive Price Index (0.0 to 1.0)
        p_val = recent_prices.get(district, ref_price)
        p_ratio = p_val / ref_price if ref_price > 0 else 1.0
        price_index = min(max((p_ratio - 0.5) / 1.0, 0.2), 1.0)

        # Derive Weather Index (0.0 to 1.0) based on NASA POWER observations
        w_info = recent_weather.get(district, {"rainfall_mm": 5.0, "temp_c": 24.0})
        rainfall = w_info["rainfall_mm"]
        temp = w_info["temp_c"]

        # Favorable moisture & temperature scoring
        w_score = 0.5
        if 2.0 <= rainfall <= 35.0:
            w_score += 0.3
        if 16.0 <= temp <= 30.0:
            w_score += 0.2
        weather_index = min(max(w_score, 0.1), 1.0)

        # Compute DCVS
        dcvs_score = calculate_dcvs(msrs_val, weather_index, price_index)
        tier, label, color = evaluate_viability_tier(dcvs_score)

        records.append({
            "district": district,
            "crop": crop_name,
            "season_type": season_type,
            "season_year": season_year,
            "dcvs": dcvs_score,
            "viability_tier": tier,
            "viability_label": label,
            "viability_color": color,
            "msrs": msrs_val,
            "weather_index": round(weather_index, 2),
            "price_index": round(price_index, 2),
            "current_price": round(p_val, 1),
            "rainfall_avg_mm": round(rainfall, 1),
            "temp_avg_c": round(temp, 1),
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "province": row["province"],
            "data_provenance": "EVIDENCE-BASED (Price: HARTI / Weather: NASA / Commitment: Proxy)",
        })

    return pd.DataFrame(records).sort_values(by="dcvs", ascending=False).reset_index(drop=True)
