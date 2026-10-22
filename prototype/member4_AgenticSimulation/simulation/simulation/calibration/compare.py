"""Compare the baseline cobweb with Colombo wholesale HARTI season prices.

The model price for a season is the reference price minus slope_ratio times
the gap of the previous season from that reference. The reference used here is
the mean of the observed season prices, so the comparison is about the swings.
Point-by-point history is not something this model can track: planted area is
not observed week by week.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from simulation.agents.farmer import aggregate_area, equal_farmers
from simulation.config.settings import (
    CropSetting,
    BaselineSettings,
    default_settings,
    demand_intercept_rs,
    demand_slope_rs_per_ha,
    supply_intercept_ha,
    supply_slope_ha_per_rs,
)
from simulation.inputs.harti import MIN_WEEKS, colombo_wholesale_seasons
from simulation.market.clearing import clear_price
from simulation.metrics.crash import season_is_crash
from simulation.scenarios.baseline import LAND_HA_PER_FARMER

# 0.05 through 2.00. A ratio above 1 makes a price gap grow.
RATIO_GRID = tuple(round(step * 0.05, 2) for step in range(1, 41))
MIN_PAIRS = 4


@dataclass(frozen=True)
class SeriesMoments:
    crop_code: str
    n_seasons: int
    n_pairs: int
    crashes: int
    change_std: float | None
    lag_corr: float | None
    lag_corr_deseasoned: float | None


@dataclass(frozen=True)
class FitResult:
    crop_code: str
    price_center: float
    best_ratio: float
    cobweb_rmse: float
    persist_rmse: float
    reference_rmse: float
    moments: SeriesMoments

    @property
    def persist_is_closer(self) -> bool:
        return self.persist_rmse < self.cobweb_rmse

    @property
    def closest(self) -> str:
        """Which one-step rule has the smallest price error."""

        scores = {
            "flat": self.reference_rmse,
            "persist": self.persist_rmse,
            "cobweb": self.cobweb_rmse,
        }
        return min(scores, key=scores.get)


def predict_price(
    crop: CropSetting,
    settings: BaselineSettings,
    last_price: float,
    slope_ratio: float,
) -> float:
    """One season of the baseline, using last_price as the price farmers see."""

    supply_slope = supply_slope_ha_per_rs(crop, settings.price_response_share)
    farmers = equal_farmers(settings.n_farmers, LAND_HA_PER_FARMER)
    area = aggregate_area(
        farmers,
        last_price,
        supply_intercept_ha(crop, supply_slope),
        supply_slope,
    )
    demand_slope = demand_slope_rs_per_ha(supply_slope, slope_ratio)
    return clear_price(area, demand_intercept_rs(crop, demand_slope), demand_slope)


def consecutive_pairs(crop_rows: pd.DataFrame) -> pd.DataFrame:
    """Seasons that follow each other on the calendar. A gap is not a pair."""

    ordered = crop_rows.sort_values("ordinal").reset_index(drop=True)
    ordered["prev_price"] = ordered["price"].shift(1)
    ordered["prev_ordinal"] = ordered["ordinal"].shift(1)
    ordered["prev_code"] = ordered["season_code"].shift(1)
    pairs = ordered[ordered["ordinal"] - ordered["prev_ordinal"] == 1].copy()
    return pairs.reset_index(drop=True)


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < MIN_PAIRS:
        return None
    if float(left.std(ddof=1)) == 0 or float(right.std(ddof=1)) == 0:
        return None
    return float(left.corr(right))


def series_moments(crop_rows: pd.DataFrame, drop_fraction: float) -> SeriesMoments:
    crop_code = str(crop_rows["crop_code"].iloc[0])
    pairs = consecutive_pairs(crop_rows)
    crashes = 0
    for row in pairs.itertuples(index=False):
        if season_is_crash(float(row.price), float(row.prev_price), drop_fraction):
            crashes += 1
    change_std = None
    if len(pairs) >= 2:
        change = (pairs["price"] - pairs["prev_price"]) / pairs["prev_price"]
        change_std = float(change.std(ddof=1))
    log_price = crop_rows[["ordinal", "season_code", "price"]].copy()
    if (log_price["price"] <= 0).any():
        raise ValueError("deseasoned lag needs positive prices")
    log_price["log_price"] = log_price["price"].map(math.log)
    type_mean = log_price.groupby("season_code")["log_price"].transform("mean")
    dev_rows = log_price[["ordinal", "season_code"]].copy()
    dev_rows["price"] = log_price["log_price"] - type_mean
    dev_pairs = consecutive_pairs(dev_rows)
    return SeriesMoments(
        crop_code=crop_code,
        n_seasons=int(len(crop_rows)),
        n_pairs=int(len(pairs)),
        crashes=crashes,
        change_std=change_std,
        lag_corr=_correlation(pairs["price"], pairs["prev_price"]),
        lag_corr_deseasoned=_correlation(dev_pairs["price"], dev_pairs["prev_price"])
        if not dev_pairs.empty
        else None,
    )


def _rmse(errors: list[float]) -> float:
    if not errors:
        raise ValueError("no season pairs to score")
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def fit_crop(crop: CropSetting, crop_rows: pd.DataFrame, settings: BaselineSettings) -> FitResult:
    """Pick the slope ratio on RATIO_GRID with the smallest one-step price error."""

    pairs = consecutive_pairs(crop_rows)
    if pairs.empty:
        raise ValueError(f"{crop.crop_code} has no consecutive seasons")
    center = float(crop_rows["price"].mean())
    centered = CropSetting(
        crop.crop_code,
        crop.crop_name,
        crop.reference_area_ha,
        center,
        crop.area_is_assumed,
    )
    previous = [float(value) for value in pairs["prev_price"]]
    actual = [float(value) for value in pairs["price"]]

    def cobweb_rmse(ratio: float) -> float:
        errors = [
            predict_price(centered, settings, last, ratio) - observed
            for last, observed in zip(previous, actual, strict=True)
        ]
        return _rmse(errors)

    scored = [(ratio, cobweb_rmse(ratio)) for ratio in RATIO_GRID]
    best_ratio, best_rmse = min(scored, key=lambda item: item[1])
    persist = _rmse([last - observed for last, observed in zip(previous, actual, strict=True)])
    reference = _rmse([center - observed for observed in actual])
    return FitResult(
        crop_code=crop.crop_code,
        price_center=center,
        best_ratio=best_ratio,
        cobweb_rmse=best_rmse,
        persist_rmse=persist,
        reference_rmse=reference,
        moments=series_moments(crop_rows, settings.crash_rule.drop_fraction),
    )


def compare_with_harti(settings: BaselineSettings | None = None) -> list[FitResult]:
    settings = default_settings() if settings is None else settings
    observed = colombo_wholesale_seasons()
    results = []
    for crop in settings.crops:
        rows = observed[observed["crop_code"] == crop.crop_code]
        if rows.empty:
            raise ValueError(f"no Colombo wholesale seasons for {crop.crop_code}")
        results.append(fit_crop(crop, rows, settings))
    return results


def _blank(value: float | None) -> str:
    if value is None:
        return "  n/a"
    return f"{value:6.2f}"


def format_report(results: list[FitResult]) -> str:
    lines = [
        "Colombo wholesale HARTI, mean of weeks in each Maha or Yala season.",
        f"Seasons with fewer than {MIN_WEEKS} priced weeks are left out.",
        "A crash is a season at least 25% below the previous season.",
        "The cobweb error uses the slope ratio with the smallest one-step miss.",
        "Persist predicts that the next season repeats the last one.",
        "",
    ]
    for result in results:
        moments = result.moments
        ratio_note = ""
        if result.best_ratio == RATIO_GRID[0]:
            ratio_note = " (smallest ratio on the grid)"
        lines.append(
            f"{result.crop_code:<8} seasons {moments.n_seasons:2d}"
            f"  pairs {moments.n_pairs:2d}"
            f"  crashes {moments.crashes:2d}"
            f"  change std {_blank(moments.change_std)}"
            f"  lag {_blank(moments.lag_corr)}"
            f"  deseasoned lag {_blank(moments.lag_corr_deseasoned)}"
        )
        lines.append(
            f"         price center {result.price_center:7.2f}"
            f"  best ratio {result.best_ratio:4.2f}"
            f"  cobweb rmse {result.cobweb_rmse:6.1f}"
            f"  persist rmse {result.persist_rmse:6.1f}"
            f"  flat rmse {result.reference_rmse:6.1f}"
            f"  closest: {result.closest}{ratio_note}"
        )
    return "\n".join(lines)


def main() -> None:
    results = compare_with_harti()
    print(format_report(results))


if __name__ == "__main__":
    main()
