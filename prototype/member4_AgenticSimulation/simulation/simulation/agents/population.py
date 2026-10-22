"""District farmers with a fixed adopter flag.

The adopter set is chosen once. A farmer who follows the score in season 1
follows it in every later season. Adoption is not re-drawn each season.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from simulation.config.districts import district_land_ha, district_targets


@dataclass(frozen=True)
class DistrictFarmer:
    farmer_id: int
    district: str
    land_ha: float
    adopter: bool
    crop_targets: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.land_ha <= 0:
            raise ValueError("land_ha must be positive")
        if not self.crop_targets:
            raise ValueError("a farmer needs at least one crop target")


def farmer_counts(n_farmers: int) -> dict[str, int]:
    """Give every district one farmer, then split the rest by land share."""

    land = district_land_ha()
    districts = sorted(land)
    if n_farmers < len(districts):
        raise ValueError(f"n_farmers must be at least {len(districts)} so every district is present")
    total_land = sum(land.values())
    extra = n_farmers - len(districts)
    raw = {district: extra * land[district] / total_land for district in districts}
    counts = {district: 1 + int(raw[district]) for district in districts}
    leftover = extra - sum(int(raw[district]) for district in districts)
    by_fraction = sorted(districts, key=lambda district: raw[district] - int(raw[district]), reverse=True)
    for district in by_fraction[:leftover]:
        counts[district] += 1
    return counts


def choose_adopters(n_farmers: int, adoption_rate: float, seed: int) -> frozenset[int]:
    """A fixed subset. The same seed returns the same farmers."""

    if not 0 <= adoption_rate <= 1:
        raise ValueError("adoption_rate must be between 0 and 1")
    if n_farmers < 1:
        raise ValueError("n_farmers must be at least 1")
    chosen = int(round(n_farmers * adoption_rate))
    chosen = min(n_farmers, max(0, chosen))
    order = list(range(n_farmers))
    random.Random(seed).shuffle(order)
    return frozenset(order[:chosen])


def build_population(n_farmers: int, adoption_rate: float, seed: int) -> tuple[DistrictFarmer, ...]:
    counts = farmer_counts(n_farmers)
    adopters = choose_adopters(n_farmers, adoption_rate, seed)
    land = district_land_ha()
    targets_by_district: dict[str, list[tuple[str, float]]] = {}
    for target in district_targets():
        targets_by_district.setdefault(target.district, []).append((target.crop_code, target.target_ha))

    farmers: list[DistrictFarmer] = []
    farmer_id = 0
    for district in sorted(counts):
        each_land = land[district] / counts[district]
        scale = 1.0 / counts[district]
        crop_targets = tuple(
            (crop_code, hectares * scale) for crop_code, hectares in targets_by_district[district]
        )
        for _ in range(counts[district]):
            farmers.append(
                DistrictFarmer(
                    farmer_id=farmer_id,
                    district=district,
                    land_ha=each_land,
                    adopter=farmer_id in adopters,
                    crop_targets=crop_targets,
                )
            )
            farmer_id += 1
    return tuple(farmers)


def plant_hectares(
    farmer: DistrictFarmer,
    prices: dict[str, float],
    reference_prices: dict[str, float],
    msrs: dict[tuple[str, str], float],
) -> dict[str, float]:
    """Split this farmer's land. Historical targets are kept when prices are at their references.

    An adopter multiplies a crop's weight by (1 - MSRS). A non-adopter ignores the score.
    """

    weights: dict[str, float] = {}
    for crop_code, target_ha in farmer.crop_targets:
        price = prices[crop_code]
        reference = reference_prices[crop_code]
        if price < 0 or reference <= 0:
            raise ValueError(f"prices for {crop_code} must be positive")
        weight = (target_ha / farmer.land_ha) * (price / reference)
        if farmer.adopter:
            score = msrs.get((farmer.district, crop_code), 0.0)
            weight *= max(0.0, 1.0 - score)
        weights[crop_code] = weight

    total = sum(weights.values())
    if total <= 0:
        ranked = sorted(
            farmer.crop_targets,
            key=lambda item: (msrs.get((farmer.district, item[0]), 0.0), -item[1]),
        )
        planted = {crop_code: 0.0 for crop_code, _ in farmer.crop_targets}
        planted[ranked[0][0]] = farmer.land_ha
        return planted
    return {crop_code: farmer.land_ha * weight / total for crop_code, weight in weights.items()}
