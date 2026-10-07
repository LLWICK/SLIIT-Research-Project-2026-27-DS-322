"""Feature configurations A, B, and C.

A is price history and season only.
B adds origin weather, real macros, and lagged finished-season extent.
C adds observed within-season cultivation progress.
B is omitted when diesel, USD/LKR, and inflation are not all non-null.
C is omitted when that progress column has no observed values.
The synthetic calendar is not a feature set here.
"""
from __future__ import annotations

import pandas as pd

PRICE_HISTORY = (
    "price_lag_1w",
    "price_lag_2w",
    "price_lag_4w",
    "price_lag_8w",
    "price_lag_52w",
    "price_mean_4w",
    "price_mean_8w",
    "price_change_4w_pct",
)
SEASON_FEATURES = (
    "season",
    "season_transition",
    "forecast_week_of_year",
    "week_within_season",
)
WEATHER_FEATURES = (
    "origin_rainfall_1w_mm",
    "origin_rainfall_sum_4w_mm",
    "origin_rainfall_sum_8w_mm",
    "origin_mean_temp_4w_c",
    "origin_rainfall_anomaly_4w_mm",
)
MACRO_FEATURES = (
    "diesel_price_lkr_litre",
    "usd_lkr_rate",
    "inflation_rate_pct",
)
EXTENT_FEATURES = (
    "previous_season_extent",
    "same_season_previous_year_extent",
    "historical_mean_season_extent",
    "extent_change_vs_previous_season",
)
PROGRESS_FEATURE = "cultivation_progress_ratio"


def _present(frame: pd.DataFrame, names: tuple[str, ...] | list[str]) -> list[str]:
    return [name for name in names if name in frame.columns]


def historical_columns(frame: pd.DataFrame) -> list[str]:
    """Price lags and seasonal features. No weather, macros, progress, or lagged extent."""
    return _present(frame, list(PRICE_HISTORY) + list(SEASON_FEATURES))


def weather_columns(frame: pd.DataFrame) -> list[str]:
    return _present(frame, WEATHER_FEATURES)


def macro_columns(frame: pd.DataFrame) -> list[str]:
    """Names that exist. Does not drop a series from B; macros_ready decides that."""
    return _present(frame, MACRO_FEATURES)


def macros_ready(frame: pd.DataFrame) -> tuple[bool, str]:
    missing = []
    for name in MACRO_FEATURES:
        if name not in frame.columns or not bool(frame[name].notna().any()):
            missing.append(name)
    if missing:
        return False, (
            "Experiment B is unavailable: diesel, USD/LKR, and inflation are not all non-null "
            f"from a real as-of series. Missing or all-null: {', '.join(missing)}."
        )
    return True, "Diesel, USD/LKR, and inflation each have non-null as-of values."


def progress_available(frame: pd.DataFrame) -> bool:
    if PROGRESS_FEATURE not in frame.columns:
        return False
    return bool(frame[PROGRESS_FEATURE].notna().any())


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str] | None]:
    """A historical prices and season. B adds weather, macros, and lagged extent. C adds observed progress."""
    historical = historical_columns(frame)
    ready, _reason = macros_ready(frame)
    extent = _present(frame, EXTENT_FEATURES)
    weather = weather_columns(frame)
    if ready and len(extent) == len(EXTENT_FEATURES) and len(weather) == len(WEATHER_FEATURES):
        multi: list[str] | None = historical + weather + list(MACRO_FEATURES) + extent
    else:
        multi = None
    if multi and progress_available(frame):
        proposed: list[str] | None = multi + [PROGRESS_FEATURE]
    else:
        proposed = None
    return {"A": historical, "B": multi, "C": proposed}


def fullest(sets: dict[str, list[str] | None]) -> tuple[str, list[str]]:
    if sets.get("C"):
        return "C", list(sets["C"])
    if sets.get("B"):
        return "B", list(sets["B"])
    return "A", list(sets.get("A") or [])
