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
    MSRS_SAFE_MAX,
    MSRS_WARNING_MAX,
    NATIONAL_BASE_CAPACITY,
)


def calculate_msrs(committed_area: float, market_demand_capacity: float) -> float:
    """
    Calculate Market Saturation Risk Score (MSRS) between 0.0 and 1.0.

    Formula (SLIIT DS-322 Member 3 Standard):
    - overshoot_ratio = committed_area / market_demand_capacity
    - Safe zone: overshoot_ratio <= 0.8 -> MSRS = 0.0
    - Critical zone: overshoot_ratio >= 1.2 -> MSRS = 1.0
    - Linear scaling between 0.8 and 1.2: MSRS = (overshoot_ratio - 0.8) / 0.4

    Args:
        committed_area: Registered committed planting area in hectares.
        market_demand_capacity: Sustainable target demand capacity in hectares.

    Returns:
        MSRS score between 0.0 (safe) and 1.0 (imminent crash).
    """
    if market_demand_capacity <= 0:
        return 1.0 if committed_area > 0 else 0.0

    overshoot_ratio = committed_area / market_demand_capacity

    if overshoot_ratio <= 0.8:
        return 0.0
    if overshoot_ratio >= 1.2:
        return 1.0

    msrs = (overshoot_ratio - 0.8) / (1.2 - 0.8)
    return round(min(max(msrs, 0.0), 1.0), 3)


def evaluate_saturation_risk(msrs: float) -> Tuple[str, str, str]:
    """
    Evaluate qualitative risk tier, descriptive label, and display color.

    Returns:
        Tuple of (tier_name, descriptive_label, hex_color).
    """
    if msrs >= MSRS_WARNING_MAX:
        return "CRITICAL", "Critical Crowding (Glut Crash Risk)", "#C53030"
    elif msrs >= MSRS_SAFE_MAX:
        return "WARNING", "Moderate Crowding (Monitor Quota)", "#D97706"
    else:
        return "SAFE", "Safe / Low Saturation (Capacity Available)", "#2E7D32"


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
        "risk_label": "Safe / Low Saturation",
        "risk_color": "#2E7D32",
        "province": "Unknown",
        "data_source": "synthetic_proxy",
    }


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
        DataFrame containing MSRS, coordinates, and risk labels for all districts.
    """
    # Filter dataset for the selected slice
    sub = df_comm[
        (df_comm["crop_name"] == crop_name) &
        (df_comm["season_type"] == season_type) &
        (df_comm["season_year"] == season_year)
    ].copy()

    records = []
    # Benchmark national capacity share
    national_cap = NATIONAL_BASE_CAPACITY.get(crop_name, 2500.0)
    default_district_cap = national_cap / len(DISTRICT_NAMES)

    for district in DISTRICT_NAMES:
        row_match = sub[sub["district"] == district]
        
        if not row_match.empty:
            comm_ha = float(row_match["committed_hectares"].iloc[0])
            tgt_ha = float(row_match["target_hectares"].iloc[0])
            source_tag = str(row_match["source"].iloc[0])
        else:
            comm_ha = 0.0
            tgt_ha = default_district_cap
            source_tag = "synthetic_proxy"

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
            "msrs": msrs_val,
            "risk_tier": tier,
            "risk_label": label,
            "color": color,
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "province": coords["province"],
            "data_source": source_tag,
        })

    return pd.DataFrame(records).sort_values(by="msrs", ascending=False).reset_index(drop=True)


def compute_national_msrs_summary(
    df_comm: pd.DataFrame,
    crop_name: str,
    season_type: str,
    season_year: int,
) -> Dict[str, Any]:
    """Compute aggregate national saturation statistics."""
    df_dist = compute_district_msrs_table(df_comm, crop_name, season_type, season_year)

    total_committed = float(df_dist["committed_hectares"].sum())
    total_target = float(df_dist["target_hectares"].sum())
    national_msrs = calculate_msrs(total_committed, total_target)
    tier, label, color = evaluate_saturation_risk(national_msrs)

    critical_count = int((df_dist["risk_tier"] == "CRITICAL").sum())
    warning_count = int((df_dist["risk_tier"] == "WARNING").sum())
    safe_count = int((df_dist["risk_tier"] == "SAFE").sum())

    return {
        "crop": crop_name,
        "season_type": season_type,
        "season_year": season_year,
        "total_committed_ha": round(total_committed, 1),
        "total_target_ha": round(total_target, 1),
        "national_msrs": national_msrs,
        "risk_tier": tier,
        "risk_label": label,
        "risk_color": color,
        "critical_districts": critical_count,
        "warning_districts": warning_count,
        "safe_districts": safe_count,
        "data_provenance": "SYNTHETIC PROXY (analysis_ready_commitment.csv)",
    }
