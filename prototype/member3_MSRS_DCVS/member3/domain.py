"""Pure, configurable MSRS/DCVS candidate calculations."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ScoringConfig


@dataclass(frozen=True)
class PressureResult:
    score: float
    pressure_ratio: float
    label: str


def calculate_pressure_risk(
    committed_area_ha: float,
    reference_capacity_ha: float,
    config: ScoringConfig,
) -> PressureResult:
    """Experimental cultivation-pressure score, bounded to [0, 1]."""
    if committed_area_ha < 0:
        raise ValueError("committed_area_ha cannot be negative")
    if reference_capacity_ha <= 0:
        raise ValueError("reference_capacity_ha must be positive")

    ratio = committed_area_ha / reference_capacity_ha
    low, high = config.warning_pressure_ratio, config.critical_pressure_ratio
    if not 0 <= low < high:
        raise ValueError("pressure thresholds must satisfy 0 <= warning < critical")

    score = min(1.0, max(0.0, (ratio - low) / (high - low)))
    label = "low" if score < 0.30 else "moderate" if score < 0.70 else "high"
    return PressureResult(round(score, 4), round(ratio, 4), label)


def calculate_dcvs(
    price_attractiveness: float,
    suitability: float,
    pressure_risk: float,
    config: ScoringConfig,
) -> float:
    """Return an experimental 0-100 viability score.

    Inputs must already be comparable [0, 1] quantities. Price attractiveness
    is normalized within the recommendation set, not treated as an absolute
    farm-profit estimate.
    """
    values = (price_attractiveness, suitability, pressure_risk)
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("score inputs must be between 0 and 1")
    total = config.price_weight + config.suitability_weight + config.pressure_weight
    if total <= 0:
        raise ValueError("at least one DCVS weight must be positive")
    viability = (
        config.price_weight * price_attractiveness
        + config.suitability_weight * suitability
        + config.pressure_weight * (1 - pressure_risk)
    ) / total
    return round(100 * viability, 2)


def min_max_normalize(values: dict[str, float]) -> dict[str, float]:
    """Normalize comparable forecast prices without inventing a price ceiling."""
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high == low:
        return {key: 0.5 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}
