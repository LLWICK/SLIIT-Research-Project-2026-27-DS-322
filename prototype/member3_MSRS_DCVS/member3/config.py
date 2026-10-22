"""Explicitly experimental configuration for the prototype.

The capacities and score weights below are demonstration controls, not validated
agricultural thresholds. Replace them after HARTI demand validation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringConfig:
    warning_pressure_ratio: float = 0.80
    critical_pressure_ratio: float = 1.20
    price_weight: float = 0.34
    suitability_weight: float = 0.33
    pressure_weight: float = 0.33


# Temporary UI/API configuration only. It is deliberately kept separate from
# observed data so it cannot be mistaken for an estimated market demand value.
DEMO_CAPACITY_HA = {
    "carrot": 5_000.0,
    "leek": 800.0,
    "tomato": 900.0,
}
