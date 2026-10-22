"""Baseline seasons: farmers plant from last season's price, then the market clears.

Crops are separate markets in this step. A farmer's land cap applies to each
crop on its own. Competition between crops for the same hectares is later work.
While price and area stay positive, a gap from the reference price flips sign
each season and is multiplied by the slope ratio.
"""

from __future__ import annotations

import pandas as pd

from simulation.agents.farmer import aggregate_area, equal_farmers
from simulation.config.settings import (
    DEFAULT_SLOPE_RATIO,
    BaselineSettings,
    default_settings,
    demand_intercept_rs,
    demand_slope_rs_per_ha,
    season_sequence,
    supply_intercept_ha,
    supply_slope_ha_per_rs,
)
from simulation.market.clearing import clear_price
from simulation.metrics.crash import season_is_crash

# Large enough that the land cap does not bind around the reference areas.
LAND_HA_PER_FARMER = 10_000.0


def run_baseline(
    settings: BaselineSettings | None = None,
    n_seasons: int = 20,
    initial_prices: dict[str, float] | None = None,
    slope_ratio: float = DEFAULT_SLOPE_RATIO,
) -> pd.DataFrame:
    """One row per crop per season. initial_prices are what farmers see before season 0."""

    settings = default_settings() if settings is None else settings
    if initial_prices is None:
        initial_prices = {crop.crop_code: crop.reference_price for crop in settings.crops}
    missing = [crop.crop_code for crop in settings.crops if crop.crop_code not in initial_prices]
    if missing:
        raise KeyError(f"initial price missing for {', '.join(missing)}")
    for crop_code, price in initial_prices.items():
        if price < 0:
            raise ValueError(f"initial price for {crop_code} must be zero or positive")

    farmers = equal_farmers(settings.n_farmers, LAND_HA_PER_FARMER)
    seasons = season_sequence(settings.start_year, n_seasons)
    rows: list[dict[str, object]] = []
    for crop in settings.crops:
        supply_slope = supply_slope_ha_per_rs(crop, settings.price_response_share)
        intercept = supply_intercept_ha(crop, supply_slope)
        demand_slope = demand_slope_rs_per_ha(supply_slope, slope_ratio)
        demand_intercept = demand_intercept_rs(crop, demand_slope)
        previous_price: float | None = None
        price_seen = initial_prices[crop.crop_code]
        for index, (season_code, season_label) in enumerate(seasons):
            area = aggregate_area(farmers, price_seen, intercept, supply_slope)
            price = clear_price(area, demand_intercept, demand_slope)
            rows.append(
                {
                    "season_index": index,
                    "season_code": season_code,
                    "season_label": season_label,
                    "crop_code": crop.crop_code,
                    "area_ha": area,
                    "price": price,
                    "previous_price": previous_price,
                    "is_crash": season_is_crash(price, previous_price, settings.crash_rule.drop_fraction),
                    "slope_ratio": slope_ratio,
                }
            )
            previous_price = price
            price_seen = price
    return pd.DataFrame(rows)


def main() -> None:
    settings = default_settings()
    initial = {crop.crop_code: crop.reference_price * 1.2 for crop in settings.crops}
    for ratio in (0.5, 1.5):
        frame = run_baseline(settings, n_seasons=8, initial_prices=initial, slope_ratio=ratio)
        print(f"slope ratio {ratio}")
        carrot = frame[frame["crop_code"] == "carrot"]
        for row in carrot.itertuples(index=False):
            print(
                f"  {row.season_label:<16} area {row.area_ha:8.1f} ha"
                f"  price {row.price:8.2f}  crash {row.is_crash}"
            )


if __name__ == "__main__":
    main()
