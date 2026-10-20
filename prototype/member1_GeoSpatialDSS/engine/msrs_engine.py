"""
Market Saturation Risk Score (MSRS) Calculation Engine.

Member 3 Analytical Bridge:
Calculates real-time farmer commitment saturation risk to predict and mitigate
Cobweb price crashes before planting decisions crystallize into oversupply.
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
    RISK_BAND_LOW_MAX,
    RISK_BAND_MODERATE_MAX,
    RISK_BAND_HIGH_MAX,
    RISK_BAND_CRITICAL_MAX,
    COLOR_SAFE_GREEN,
    COLOR_WARNING_AMBER,
    COLOR_HIGH_ORANGE,
    COLOR_CRITICAL_RED,
    NATIONAL_BASE_CAPACITY,
    get_risk_band,
)
from data_loader import cache_decorator


def calculate_msrs(committed_area: float, market_demand_capacity: float) -> float:
    """
    Calculate Market Saturation Risk Score (MSRS) standardized between 0.0 and 100.0.

    Formula (SLIIT DS-322 Member 3 Standard, 0–100 scale):
    - overshoot_ratio = committed_area / market_demand_capacity
    - Safe zone: overshoot_ratio <= 0.8 -> MSRS = 0.0
    - Critical zone: overshoot_ratio >= 1.2 -> MSRS = 100.0
    - Linear scaling between 0.8 and 1.2: MSRS = ((overshoot_ratio - 0.8) / 0.4) * 100.0

    Args:
        committed_area: Registered committed planting area in hectares.
        market_demand_capacity: Sustainable target demand capacity in hectares.

    Returns:
        MSRS score between 0.0 (safe) and 100.0 (imminent crash) with 1 decimal place.
    """
    if market_demand_capacity <= 0:
        return 100.0 if committed_area > 0 else 0.0

    overshoot_ratio = committed_area / market_demand_capacity

    if overshoot_ratio <= 0.8:
        return 0.0
    if overshoot_ratio >= 1.2:
        return 100.0

    msrs = ((overshoot_ratio - 0.8) / (1.2 - 0.8)) * 100.0
    return round(min(max(msrs, 0.0), 100.0), 1)


def evaluate_saturation_risk(msrs: float) -> Tuple[str, str, str]:
    """
    Evaluate qualitative risk tier, descriptive label, and display color on 0–100 scale.
    Bands:
      Low: 0.0 <= msrs <= 20.0
      Moderate: 20.0 < msrs <= 60.0
      High: 60.0 < msrs <= 80.0
      Critical: 80.0 < msrs <= 100.0

    Returns:
        Tuple of (tier_name, descriptive_label, hex_color).
    """
    s = min(max(float(msrs), 0.0), 100.0)
    if s <= RISK_BAND_LOW_MAX:
        return "SAFE", "Low Saturation (Capacity Available)", COLOR_SAFE_GREEN
    elif s <= RISK_BAND_MODERATE_MAX:
        return "WARNING", "Moderate Crowding (Monitor Quota)", COLOR_WARNING_AMBER
    elif s <= RISK_BAND_HIGH_MAX:
        return "HIGH", "High Crowding (Acreage Alert)", COLOR_HIGH_ORANGE
    else:
        return "CRITICAL", "Critical Crowding (Glut Crash Risk)", COLOR_CRITICAL_RED



def get_district_msrs_record(
    df_dist_msrs: pd.DataFrame,
    district: str,
) -> Dict[str, Any]:
    """
    Retrieve single district's MSRS metrics, ensuring 100% numerical consistency across views.
    """
    row = df_dist_msrs[df_dist_msrs["district"] == district]
    if not row.empty:
        r = row.iloc[0]
        return {
            "district": district,
            "msrs": float(r["msrs"]),
            "committed_ha": float(r["committed_hectares"]),
            "target_ha": float(r["target_hectares"]),
            "overshoot_ratio": float(r["overshoot_ratio"]),
            "overshoot_pct": float(r["overshoot_pct"]),
            "risk_tier": str(r["risk_tier"]),
            "risk_label": str(r["risk_label"]),
            "risk_color": str(r["color"]),
            "province": str(r["province"]),
            "data_source": str(r["data_source"]),
        }
    return {
        "district": district,
        "msrs": 0.0,
        "committed_ha": 0.0,
        "target_ha": 100.0,
        "overshoot_ratio": 0.0,
        "overshoot_pct": 0.0,
        "risk_tier": "SAFE",
        "risk_label": "Low Saturation (Capacity Available)",
        "risk_color": COLOR_SAFE_GREEN,
        "province": "Unknown",
        "data_source": "Planning Baseline",
    }


@cache_decorator
def _cached_compute_district_msrs_table(
    crop_name: str,
    season_type: str,
    season_year: int,
    _df_comm: pd.DataFrame,
    data_sig: tuple = (),
) -> pd.DataFrame:
    """Internal cached implementation of district MSRS calculation."""
    # Filter dataset for the selected slice
    sub = _df_comm[
        (_df_comm["crop_name"] == crop_name) &
        (_df_comm["season_type"] == season_type) &
        (_df_comm["season_year"] == season_year)
    ]

    records = []
    # Benchmark national capacity share
    national_cap = NATIONAL_BASE_CAPACITY.get(crop_name, 2500.0)
    default_district_cap = national_cap / len(DISTRICT_NAMES)

    for district in DISTRICT_NAMES:
        row_match = sub[sub["district"] == district]
        
        if not row_match.empty:
            comm_ha = float(row_match["committed_hectares"].iloc[0])
            tgt_ha = float(row_match["target_hectares"].iloc[0])
            source_tag = "Cultivation Commitments (HARTI/DOA district records)"
        else:
            comm_ha = 0.0
            tgt_ha = default_district_cap
            source_tag = "Planning Baseline"

        if tgt_ha <= 0:
            tgt_ha = default_district_cap

        msrs_val = calculate_msrs(comm_ha, tgt_ha)
        tier, label, color = evaluate_saturation_risk(msrs_val)
        overshoot_pct = ((comm_ha - tgt_ha) / tgt_ha) * 100.0 if tgt_ha > 0 else 0.0

        coords = DISTRICT_COORDS.get(district, {"lat": 7.8731, "lon": 80.7718, "province": "Unknown"})

        records.append({
            "district": district,
            "crop": crop_name,
            "season_type": season_type,
            "season_year": season_year,
            "committed_hectares": round(comm_ha, 1),
            "target_hectares": round(tgt_ha, 1),
            "overshoot_ratio": round(comm_ha / tgt_ha, 2) if tgt_ha > 0 else 0.0,
            "overshoot_pct": round(overshoot_pct, 1),
            "msrs": round(msrs_val, 1),
            "risk_tier": tier,
            "risk_label": label,
            "color": color,
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "province": coords["province"],
            "data_source": source_tag,
        })

    return pd.DataFrame(records).sort_values(by="msrs", ascending=False).reset_index(drop=True)


def compute_district_msrs_table(
    df_comm: pd.DataFrame,
    crop_name: str,
    season_type: str,
    season_year: int,
) -> pd.DataFrame:
    """
    Generate district-level MSRS panel for all 25 districts of Sri Lanka.

    Args:
        df_comm: Loaded commitment DataFrame.
        crop_name: Selected vegetable crop.
        season_type: 'Maha' or 'Yala'.
        season_year: Planning year (e.g. 2024).

    Returns:
        DataFrame containing MSRS (0-100), coordinates, and risk labels for all districts.
    """
    data_sig = (len(df_comm) if df_comm is not None else 0,)
    return _cached_compute_district_msrs_table(
        crop_name=crop_name,
        season_type=season_type,
        season_year=season_year,
        _df_comm=df_comm,
        data_sig=data_sig,
    )


@cache_decorator
def _cached_compute_national_msrs_summary(
    crop_name: str,
    season_type: str,
    season_year: int,
    _df_comm: pd.DataFrame,
    data_sig: tuple = (),
) -> Dict[str, Any]:
    """Internal cached implementation of national MSRS summary."""
    df_dist = compute_district_msrs_table(_df_comm, crop_name, season_type, season_year)

    total_committed = float(df_dist["committed_hectares"].sum())
    total_target = float(df_dist["target_hectares"].sum())
    national_msrs = calculate_msrs(total_committed, total_target)
    tier, label, color = evaluate_saturation_risk(national_msrs)

    critical_count = int((df_dist["risk_tier"] == "CRITICAL").sum())
    high_count = int((df_dist["risk_tier"] == "HIGH").sum())
    warning_count = int((df_dist["risk_tier"] == "WARNING").sum())
    safe_count = int((df_dist["risk_tier"] == "SAFE").sum())

    return {
        "crop": crop_name,
        "season_type": season_type,
        "season_year": season_year,
        "total_committed_ha": round(total_committed, 1),
        "total_target_ha": round(total_target, 1),
        "national_msrs": round(national_msrs, 1),
        "risk_tier": tier,
        "risk_label": label,
        "risk_color": color,
        "critical_districts": critical_count,
        "high_districts": high_count,
        "warning_districts": warning_count,
        "safe_districts": safe_count,
        "data_provenance": "Cultivation Commitments (HARTI/DOA district records)",
    }


def compute_national_msrs_summary(
    df_comm: pd.DataFrame,
    crop_name: str,
    season_type: str,
    season_year: int,
) -> Dict[str, Any]:
    """Compute aggregate national saturation statistics."""
    data_sig = (len(df_comm) if df_comm is not None else 0,)
    return _cached_compute_national_msrs_summary(
        crop_name=crop_name,
        season_type=season_type,
        season_year=season_year,
        _df_comm=df_comm,
        data_sig=data_sig,
    )
