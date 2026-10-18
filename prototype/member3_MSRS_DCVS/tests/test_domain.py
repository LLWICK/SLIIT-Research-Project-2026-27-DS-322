import pytest

from member3.config import ScoringConfig
from member3.domain import calculate_dcvs, calculate_pressure_risk


def test_pressure_risk_is_monotonic():
    config = ScoringConfig()
    lower = calculate_pressure_risk(850, 1_000, config)
    higher = calculate_pressure_risk(1_000, 1_000, config)
    assert higher.score >= lower.score
    assert higher.pressure_ratio > lower.pressure_ratio


def test_pressure_requires_positive_reference_capacity():
    with pytest.raises(ValueError, match="positive"):
        calculate_pressure_risk(1, 0, ScoringConfig())


def test_dcvs_rewards_lower_pressure_at_same_inputs():
    config = ScoringConfig(price_weight=1, suitability_weight=1, pressure_weight=1)
    assert calculate_dcvs(0.5, 0.5, 0.1, config) > calculate_dcvs(0.5, 0.5, 0.9, config)
