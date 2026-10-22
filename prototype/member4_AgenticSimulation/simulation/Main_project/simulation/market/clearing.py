"""Inverse demand: a larger planted area clears at a lower price."""

from __future__ import annotations


def clear_price(area_ha: float, demand_intercept: float, demand_slope: float) -> float:
    """Rs/kg. The line is intercept - slope * hectares, and price cannot be negative."""

    if area_ha < 0:
        raise ValueError("area_ha must be zero or positive")
    if demand_slope < 0:
        raise ValueError("demand_slope must be zero or positive")
    return max(0.0, demand_intercept - demand_slope * area_ha)
