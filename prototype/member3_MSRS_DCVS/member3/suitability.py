"""Mock agro-climatic suitability engine mapping planting areas to crop requirements.

Provides realistic environmental profiles (altitude, seasonal rainfall, mean temperature,
climate zone, and soil type) for Sri Lankan agricultural districts, and calculates
physiographic suitability scores for candidate crops.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PlantingAreaProfile:
    district: str
    name: str
    altitude_m: float
    mean_temp_c: float
    seasonal_rainfall_mm: float
    climate_zone: str
    soil_type: str
    description: str


@dataclass(frozen=True)
class CropAgronomicRequirement:
    crop: str
    optimal_altitude_min: float
    optimal_altitude_max: float
    optimal_temp_min: float
    optimal_temp_max: float
    optimal_rainfall_min: float
    optimal_rainfall_max: float
    growth_duration_weeks: int
    water_demand_level: float  # 0.0 to 1.0
    labor_intensity_level: float  # 0.0 to 1.0
    capital_cost_per_ha_lkr: float


# Sri Lankan agricultural planting area profiles
PLANTING_AREAS: dict[str, PlantingAreaProfile] = {
    "nuwara_eliya": PlantingAreaProfile(
        district="nuwara_eliya",
        name="Nuwara Eliya (Up-Country Wet Zone)",
        altitude_m=1868.0,
        mean_temp_c=16.0,
        seasonal_rainfall_mm=1900.0,
        climate_zone="Up Country Wet Zone (UW1)",
        soil_type="Red Yellow Podzolic (High Organic)",
        description="High elevation, cool montane climate with consistent mist and rainfall. Premier zone for temperate hill vegetables.",
    ),
    "badulla": PlantingAreaProfile(
        district="badulla",
        name="Badulla / Welimada (Up-Country Intermediate)",
        altitude_m=680.0,
        mean_temp_c=23.0,
        seasonal_rainfall_mm=1600.0,
        climate_zone="Up Country Intermediate Zone (UCI)",
        soil_type="Red Yellow Podzolic & Immature Brown Loams",
        description="Mid to high elevation valleys with moderate temperatures and good vegetable cultivation tradition.",
    ),
    "dambulla": PlantingAreaProfile(
        district="dambulla",
        name="Dambulla / Matale North (Dry/Intermediate Zone)",
        altitude_m=165.0,
        mean_temp_c=29.0,
        seasonal_rainfall_mm=1150.0,
        climate_zone="Low Country Dry Zone (DL1b)",
        soil_type="Reddish Brown Earths & Low Humic Gley",
        description="Lowland warm agricultural hub with irrigated farming; suitable for heat-tolerant and solanaceous crops.",
    ),
    "colombo": PlantingAreaProfile(
        district="colombo",
        name="Colombo / Western Basin (Low-Country Wet Zone)",
        altitude_m=12.0,
        mean_temp_c=28.5,
        seasonal_rainfall_mm=2450.0,
        climate_zone="Low Country Wet Zone (WL1)",
        soil_type="Alluvial & Red Yellow Podzolic",
        description="Lowland coastal plain with high humidity and heavy monsoon rainfall; limited highland root vegetable suitability.",
    ),
    "meegoda": PlantingAreaProfile(
        district="meegoda",
        name="Meegoda / Homagama (Low-Country Semi-Wet)",
        altitude_m=35.0,
        mean_temp_c=28.0,
        seasonal_rainfall_mm=2300.0,
        climate_zone="Low Country Wet Zone (WL2)",
        soil_type="Red Yellow Podzolic Loam",
        description="Peri-urban agricultural belt in Western province with tropical climate.",
    ),
}

# Crop requirements based on Sri Lanka Department of Agriculture guidelines
CROP_REQUIREMENTS: dict[str, CropAgronomicRequirement] = {
    "carrot": CropAgronomicRequirement(
        crop="carrot",
        optimal_altitude_min=1000.0,
        optimal_altitude_max=2200.0,
        optimal_temp_min=14.0,
        optimal_temp_max=21.0,
        optimal_rainfall_min=1000.0,
        optimal_rainfall_max=1800.0,
        growth_duration_weeks=13,
        water_demand_level=0.65,
        labor_intensity_level=0.75,
        capital_cost_per_ha_lkr=420_000.0,
    ),
    "leek": CropAgronomicRequirement(
        crop="leek",
        optimal_altitude_min=900.0,
        optimal_altitude_max=2200.0,
        optimal_temp_min=13.0,
        optimal_temp_max=20.0,
        optimal_rainfall_min=1200.0,
        optimal_rainfall_max=2000.0,
        growth_duration_weeks=15,
        water_demand_level=0.70,
        labor_intensity_level=0.80,
        capital_cost_per_ha_lkr=480_000.0,
    ),
    "tomato": CropAgronomicRequirement(
        crop="tomato",
        optimal_altitude_min=100.0,
        optimal_altitude_max=1000.0,
        optimal_temp_min=20.0,
        optimal_temp_max=29.0,
        optimal_rainfall_min=700.0,
        optimal_rainfall_max=1400.0,
        growth_duration_weeks=11,
        water_demand_level=0.60,
        labor_intensity_level=0.70,
        capital_cost_per_ha_lkr=350_000.0,
    ),
}


def _range_match_score(value: float, min_val: float, max_val: float, tolerance: float) -> float:
    """Computes a smooth continuous membership score in [0.0, 1.0]."""
    if min_val <= value <= max_val:
        return 1.0
    if value < min_val:
        distance = min_val - value
    else:
        distance = value - max_val
    score = math.exp(-0.5 * (distance / max(tolerance, 1e-4)) ** 2)
    return max(0.05, min(1.0, score))


def calculate_suitability(crop: str, area_key: str) -> dict[str, Any]:
    """Calculate agro-climatic suitability of a crop for a given planting area."""
    key = area_key.strip().lower().replace(" ", "_")
    profile = PLANTING_AREAS.get(key, PLANTING_AREAS.get("nuwara_eliya"))
    req = CROP_REQUIREMENTS.get(crop.lower())
    if not req:
        raise ValueError(f"Unknown crop for suitability evaluation: {crop}")

    altitude_score = _range_match_score(
        profile.altitude_m,
        req.optimal_altitude_min,
        req.optimal_altitude_max,
        tolerance=500.0,
    )
    temp_score = _range_match_score(
        profile.mean_temp_c,
        req.optimal_temp_min,
        req.optimal_temp_max,
        tolerance=4.0,
    )
    rainfall_score = _range_match_score(
        profile.seasonal_rainfall_mm,
        req.optimal_rainfall_min,
        req.optimal_rainfall_max,
        tolerance=500.0,
    )

    # Weighted combination: Temperature (40%), Altitude (35%), Rainfall (25%)
    composite_score = round(
        (0.35 * altitude_score) + (0.40 * temp_score) + (0.25 * rainfall_score),
        3,
    )

    return {
        "crop": crop,
        "planting_area": profile.district,
        "area_name": profile.name,
        "composite_suitability": composite_score,
        "sub_scores": {
            "altitude": round(altitude_score, 3),
            "temperature": round(temp_score, 3),
            "rainfall": round(rainfall_score, 3),
        },
        "environmental_factors": {
            "altitude_m": profile.altitude_m,
            "mean_temp_c": profile.mean_temp_c,
            "seasonal_rainfall_mm": profile.seasonal_rainfall_mm,
            "climate_zone": profile.climate_zone,
            "soil_type": profile.soil_type,
        },
    }


def get_area_suitability_map(area_key: str, crops: list[str] | None = None) -> dict[str, float]:
    """Return dictionary of {crop: composite_suitability} for an area."""
    target_crops = crops or list(CROP_REQUIREMENTS.keys())
    return {crop: calculate_suitability(crop, area_key)["composite_suitability"] for crop in target_crops}
