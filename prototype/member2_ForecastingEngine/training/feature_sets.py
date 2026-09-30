"""Feature configurations A, B, and C from Methodology.docx section 3.13."""
from __future__ import annotations

import pandas as pd

from training.config import PRICE_LAGS, WEATHER_LAGS, WEATHER_LEVELS

MACRO_COLUMNS = ("diesel_lkr_per_litre", "usd_lkr", "inflation_yoy")


def _present(frame: pd.DataFrame, names: list[str]) -> list[str]:
    return [name for name in names if name in frame.columns]


def historical_columns(frame: pd.DataFrame) -> list[str]:
    names = [f"price_lag_{lag}" for lag in PRICE_LAGS]
    names += ["roll_mean_4", "roll_std_4", "roll_mean_12", "roll_std_12", "log_price", "log_return_1"]
    names += ["week", "month", "is_yala", "is_maha", "week_within_season", "is_transition"]
    return _present(frame, names)


def weather_columns(frame: pd.DataFrame) -> list[str]:
    names = [column for column in WEATHER_LEVELS if column in frame.columns]
    names += [f"{column}_lag_{lag}" for column in WEATHER_LEVELS if column in frame.columns for lag in WEATHER_LAGS]
    return names


def macro_columns(frame: pd.DataFrame) -> list[str]:
    kept = []
    for name in MACRO_COLUMNS:
        if name in frame.columns and frame[name].notna().any():
            kept.append(name)
    return kept


def progress_available(frame: pd.DataFrame) -> bool:
    return "cultivation_progress" in frame.columns and bool(frame["cultivation_progress"].notna().any())


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str] | None]:
    """A historical, B multi-source, C proposed. C is None when progress has no real values."""
    historical = historical_columns(frame)
    multi = historical + weather_columns(frame) + macro_columns(frame)
    proposed = multi + ["cultivation_progress"] if progress_available(frame) else None
    return {"A": historical, "B": multi, "C": proposed}


def fullest(sets: dict[str, list[str] | None]) -> tuple[str, list[str]]:
    if sets["C"]:
        return "C", list(sets["C"])
    return "B", list(sets["B"] or [])
