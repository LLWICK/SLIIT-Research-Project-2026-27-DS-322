"""Crops, seasons, and the price-crash rule for the baseline simulation.

Reference areas for carrot and leeks are national targets from checked DOA
snapshots. Tomato has no checked snapshot, so its area is an explicit
assumption. Reference prices are Colombo wholesale medians from the HARTI
workbook. The slopes are parameters: the warehouse has no quantity sold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from data_layer.season import season_for

# Share of the reference area that responds to price at the reference price.
# The rest is the supply intercept, so the intercept stays positive.
PRICE_RESPONSE_SHARE = 0.20

# Below 1 the cobweb damps. Above 1 it grows. Equal to 1 it cycles.
# The product is supply slope (ha per Rs/kg) times demand slope (Rs/kg per ha).
DEFAULT_SLOPE_RATIO = 0.8


@dataclass(frozen=True)
class CropSetting:
    crop_code: str
    crop_name: str
    reference_area_ha: float
    reference_price: float
    area_is_assumed: bool


@dataclass(frozen=True)
class CrashRule:
    """A season is a crash when its price is at least `drop_fraction` below the previous season.

    The first season is never a crash. A week is not a crash. A following season
    that stays low is not a second crash unless it falls by this fraction again.
    0.25 is the definition used by the experiments. It is not a fitted threshold.
    """

    drop_fraction: float = 0.25

    def __post_init__(self) -> None:
        if not 0 < self.drop_fraction <= 1:
            raise ValueError("drop_fraction must be between 0 and 1")


@dataclass(frozen=True)
class BaselineSettings:
    crops: tuple[CropSetting, ...]
    crash_rule: CrashRule = CrashRule()
    n_farmers: int = 100
    start_year: int = 2015
    price_response_share: float = PRICE_RESPONSE_SHARE

    def __post_init__(self) -> None:
        if self.n_farmers < 1:
            raise ValueError("n_farmers must be at least 1")
        if not 0 < self.price_response_share < 1:
            raise ValueError("price_response_share must be between 0 and 1")
        codes = [crop.crop_code for crop in self.crops]
        if len(codes) != len(set(codes)):
            raise ValueError("crop codes must be unique")

    def crop(self, crop_code: str) -> CropSetting:
        for crop in self.crops:
            if crop.crop_code == crop_code:
                return crop
        raise KeyError(crop_code)


def default_settings() -> BaselineSettings:
    return BaselineSettings(
        crops=(
            CropSetting(
                "carrot",
                "Carrot",
                reference_area_ha=2001.0,  # Maha 2024/25 final, report 2025-04-15
                reference_price=149.09,  # Colombo wholesale median, Rs/kg
                area_is_assumed=False,
            ),
            CropSetting(
                "leeks",
                "Leeks",
                reference_area_ha=1333.0,  # 2025 Yala final, report 2025-10-15
                reference_price=125.83,
                area_is_assumed=False,
            ),
            CropSetting(
                "tomato",
                "Tomato",
                # No tomato page passed the sum check. This matches the leeks
                # final target only so the crop can run. It is not a DOA figure.
                reference_area_ha=1333.0,
                reference_price=120.0,
                area_is_assumed=True,
            ),
        )
    )


def season_sequence(start_year: int, n_seasons: int) -> list[tuple[str, str]]:
    """Maha then Yala, starting in October of `start_year`.

    Labels are the same strings as data_layer.season.season_for.
    Maha is October-March. Yala is April-September.
    """

    if n_seasons < 1:
        raise ValueError("n_seasons must be at least 1")
    seasons: list[tuple[str, str]] = []
    year = start_year
    for index in range(n_seasons):
        if index % 2 == 0:
            seasons.append(season_for(date(year, 10, 1)))
        else:
            seasons.append(season_for(date(year + 1, 4, 1)))
            year += 1
    return seasons


def season_ordinal(season_code: str, season_label: str, origin_year: int = 2015) -> int:
    """Position on the Maha/Yala calendar. October 2015 is 0 when origin_year is 2015."""

    if season_code == "maha":
        year = int(season_label.split("/")[0])
        return (year - origin_year) * 2
    if season_code == "yala":
        year = int(season_label.split()[0])
        return (year - origin_year - 1) * 2 + 1
    raise ValueError(f"unknown season code {season_code}")


def supply_slope_ha_per_rs(crop: CropSetting, price_response_share: float) -> float:
    """Hectares planted per extra Rs/kg of the price farmers last saw."""

    return price_response_share * crop.reference_area_ha / crop.reference_price


def supply_intercept_ha(crop: CropSetting, supply_slope: float) -> float:
    """Hectares planted when the last price is zero, before the zero floor."""

    return crop.reference_area_ha - supply_slope * crop.reference_price


def demand_slope_rs_per_ha(supply_slope: float, slope_ratio: float) -> float:
    """Rs/kg lost per extra hectare. slope_ratio is the cobweb multiplier."""

    if supply_slope <= 0:
        raise ValueError("supply_slope must be positive")
    if slope_ratio <= 0:
        raise ValueError("slope_ratio must be positive")
    return slope_ratio / supply_slope


def demand_intercept_rs(crop: CropSetting, demand_slope: float) -> float:
    """Price at zero hectares, before the zero floor."""

    return crop.reference_price + demand_slope * crop.reference_area_ha
