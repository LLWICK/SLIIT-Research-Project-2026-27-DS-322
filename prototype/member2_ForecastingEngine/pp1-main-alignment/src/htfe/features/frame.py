"""Direct multi-horizon feature frame. Lags never cross series.

Usage (from member2_ForecastingEngine):
  python -m training.build_feature_frame
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from htfe.training.config import (
    FEATURE_PATH,
    HORIZONS,
    PANEL_PATH,
    PRICE_LAGS,
    WEATHER_LAGS,
    WEATHER_LEVELS,
    assign_split,
)


def week_within_season(week: int, season: str) -> int:
    """Weeks since the cultivation season started. Yala starts at week 14; Maha at week 40."""
    week = int(week)
    if season == "Yala":
        return week - 13
    if week >= 40:
        return week - 39
    return week + 14


def feature_columns(frame: pd.DataFrame) -> list[str]:
    names = [f"price_lag_{lag}" for lag in PRICE_LAGS]
    names += ["roll_mean_4", "roll_std_4", "roll_mean_12", "roll_std_12", "log_price", "log_return_1"]
    names += ["week", "month", "is_yala", "is_maha", "missing_run_length", "dri", "is_imputed"]
    names += [column for column in WEATHER_LEVELS if column in frame.columns]
    names += [f"{column}_lag_{lag}" for column in WEATHER_LEVELS if column in frame.columns for lag in WEATHER_LAGS]
    names += [
        "supply_extent_ha_lag1_season",
        "supply_production_mt_lag1_season",
        "cultivation_intensity",
    ]
    names += [column for column in frame.columns if column.startswith("holiday_")]
    return [name for name in names if name in frame.columns]


def build(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = panel.copy() if panel is not None else pd.read_parquet(PANEL_PATH)
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    frame = frame.sort_values(["crop", "market", "week_start"]).reset_index(drop=True)
    frame["unique_id"] = frame["crop"] + "__" + frame["market"]
    grouped = frame.groupby("unique_id", sort=False)
    observed_price = frame["price"].where(frame["is_observed"])
    feature_price = pd.to_numeric(frame["price_feature"], errors="coerce")
    frame["price_feature"] = feature_price

    for lag in PRICE_LAGS:
        frame[f"price_lag_{lag}"] = grouped["price_feature"].shift(lag)
    frame["roll_mean_4"] = grouped["price_feature"].transform(lambda s: s.rolling(4, min_periods=4).mean())
    frame["roll_std_4"] = grouped["price_feature"].transform(lambda s: s.rolling(4, min_periods=4).std())
    frame["roll_mean_12"] = grouped["price_feature"].transform(lambda s: s.rolling(12, min_periods=8).mean())
    frame["roll_std_12"] = grouped["price_feature"].transform(lambda s: s.rolling(12, min_periods=8).std())
    frame["log_price"] = np.log(feature_price.where(feature_price > 0))
    frame["log_return_1"] = frame["log_price"] - grouped["log_price"].shift(1)
    frame["month"] = frame["week_start"].dt.month.astype(int)
    frame["is_yala"] = (frame["season"] == "Yala").astype(int)
    frame["is_maha"] = (frame["season"] == "Maha").astype(int)
    frame["week_within_season"] = [
        week_within_season(week, season) for week, season in zip(frame["week"], frame["season"])
    ]
    frame["is_transition"] = frame["week"].isin([12, 13, 14, 15, 38, 39, 40, 41]).astype(int)
    frame["recent_median_4"] = grouped["price"].transform(lambda s: s.rolling(4, min_periods=1).median())

    for column in WEATHER_LEVELS:
        if column not in frame.columns:
            continue
        for lag in WEATHER_LAGS:
            frame[f"{column}_lag_{lag}"] = grouped[column].shift(lag)

    for horizon in HORIZONS:
        future = grouped["price"].shift(-horizon)
        future_obs = grouped["is_observed"].shift(-horizon).fillna(False).astype(bool)
        frame[f"y_price_h{horizon}"] = future.where(future_obs)
        frame[f"target_week_h{horizon}"] = grouped["week_start"].shift(-horizon)
        base = observed_price.where(observed_price > 0)
        frame[f"y_logret_h{horizon}"] = np.log(frame[f"y_price_h{horizon}"] / base)
        frame[f"y_ratio_h{horizon}"] = frame[f"y_price_h{horizon}"] / frame["recent_median_4"].replace(0, np.nan)

    splits = assign_split(frame["week_start"])
    frame["split"] = splits["split"]
    frame["is_shock"] = splits["is_shock"]
    return frame


def main() -> None:
    frame = build()
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FEATURE_PATH, index=False)
    print(f"wrote {FEATURE_PATH} rows={len(frame)} features={len(feature_columns(frame))}")


if __name__ == "__main__":
    main()
