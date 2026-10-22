"""Naive planting: area depends on last season's price and on nothing else."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Farmer:
    farmer_id: int
    land_ha: float

    def __post_init__(self) -> None:
        if self.land_ha <= 0:
            raise ValueError("land_ha must be positive")


def plant_area(last_price: float, intercept_ha: float, supply_slope: float, land_ha: float) -> float:
    """Hectares one farmer plants.

    The uncapped line is intercept + slope * last price. Area cannot be
    negative and cannot exceed the farmer's land.
    """

    if last_price < 0:
        raise ValueError("last_price must be zero or positive")
    if supply_slope < 0:
        raise ValueError("supply_slope must be zero or positive")
    if land_ha <= 0:
        raise ValueError("land_ha must be positive")
    desired = intercept_ha + supply_slope * last_price
    return min(land_ha, max(0.0, desired))


def equal_farmers(n_farmers: int, land_ha: float) -> tuple[Farmer, ...]:
    if n_farmers < 1:
        raise ValueError("n_farmers must be at least 1")
    return tuple(Farmer(farmer_id=i, land_ha=land_ha) for i in range(n_farmers))


def aggregate_area(
    farmers: tuple[Farmer, ...],
    last_price: float,
    intercept_ha: float,
    supply_slope: float,
) -> float:
    """Sum of identical farmers. Intercept and slope are the national totals."""

    if not farmers:
        raise ValueError("at least one farmer is required")
    n_farmers = len(farmers)
    each_intercept = intercept_ha / n_farmers
    each_slope = supply_slope / n_farmers
    return sum(
        plant_area(last_price, each_intercept, each_slope, farmer.land_ha)
        for farmer in farmers
    )
