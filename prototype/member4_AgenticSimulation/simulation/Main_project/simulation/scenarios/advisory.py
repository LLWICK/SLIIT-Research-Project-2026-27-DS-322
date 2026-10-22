"""Carrot and leeks seasons with a fixed set of MSRS adopters.

Farmers share the checked district targets. They can move land between
carrot and leeks. Tomato is not in this run: it has no checked district
table. The score they see is last season's score, because everyone in a
season plants at the same time.
"""

from __future__ import annotations

import random

import pandas as pd

from simulation.agents.population import build_population, plant_hectares
from simulation.config.districts import ADVISORY_CROPS
from simulation.config.settings import (
    DEFAULT_SLOPE_RATIO,
    BaselineSettings,
    default_settings,
    demand_intercept_rs,
    demand_slope_rs_per_ha,
    season_sequence,
    supply_slope_ha_per_rs,
)
from simulation.market.clearing import clear_price
from simulation.market.msrs import district_scores
from simulation.market.price_process import combine_price
from simulation.metrics.crash import season_is_crash


def run_advisory(
    adoption_rate: float,
    seed: int = 0,
    n_seasons: int = 16,
    slope_ratio: float = DEFAULT_SLOPE_RATIO,
    n_farmers: int = 100,
    settings: BaselineSettings | None = None,
    initial_prices: dict[str, float] | None = None,
    persistence: dict[str, float] | None = None,
    shock_std: dict[str, float] | None = None,
    shock_seed: int = 0,
) -> pd.DataFrame:
    """One row per crop per season. Adopters are fixed for the whole run.

    persistence and shock_std are Rs/kg AR(1) terms. Both default to zero,
    which leaves the price equal to the clearing price.
    """

    settings = default_settings() if settings is None else settings
    persistence = {} if persistence is None else persistence
    shock_std = {} if shock_std is None else shock_std
    shock_rng = random.Random(shock_seed)
    crops = tuple(settings.crop(crop_code) for crop_code in ADVISORY_CROPS)
    if initial_prices is None:
        initial_prices = {crop.crop_code: crop.reference_price for crop in crops}
    reference_prices = {crop.crop_code: crop.reference_price for crop in crops}
    farmers = build_population(n_farmers, adoption_rate, seed)
    seasons = season_sequence(settings.start_year, n_seasons)

    price_rules = {}
    for crop in crops:
        supply_slope = supply_slope_ha_per_rs(crop, settings.price_response_share)
        demand_slope = demand_slope_rs_per_ha(supply_slope, slope_ratio)
        price_rules[crop.crop_code] = (demand_intercept_rs(crop, demand_slope), demand_slope)

    prices = dict(initial_prices)
    previous_realized: dict[str, float | None] = {crop.crop_code: None for crop in crops}
    committed: dict[tuple[str, str], float] = {}
    rows: list[dict[str, object]] = []

    for index, (season_code, season_label) in enumerate(seasons):
        msrs = district_scores(committed)
        committed = {}
        national = {crop.crop_code: 0.0 for crop in crops}
        for farmer in farmers:
            planted = plant_hectares(farmer, prices, reference_prices, msrs)
            for crop_code, hectares in planted.items():
                national[crop_code] += hectares
                key = (farmer.district, crop_code)
                committed[key] = committed.get(key, 0.0) + hectares

        for crop in crops:
            intercept, demand_slope = price_rules[crop.crop_code]
            cleared = clear_price(national[crop.crop_code], intercept, demand_slope)
            scale = shock_std.get(crop.crop_code, 0.0)
            shock = shock_rng.gauss(0.0, scale) if scale > 0 else 0.0
            last = previous_realized[crop.crop_code]
            if last is None:
                price = max(0.0, cleared + shock)
            else:
                price = combine_price(
                    last, cleared, persistence.get(crop.crop_code, 0.0), shock
                )
            rows.append(
                {
                    "season_index": index,
                    "season_code": season_code,
                    "season_label": season_label,
                    "crop_code": crop.crop_code,
                    "area_ha": national[crop.crop_code],
                    "price": price,
                    "is_crash": season_is_crash(
                        price, previous_realized[crop.crop_code], settings.crash_rule.drop_fraction
                    ),
                    "adoption_rate": adoption_rate,
                    "seed": seed,
                }
            )
            previous_realized[crop.crop_code] = price
            prices[crop.crop_code] = price
    return pd.DataFrame(rows)
