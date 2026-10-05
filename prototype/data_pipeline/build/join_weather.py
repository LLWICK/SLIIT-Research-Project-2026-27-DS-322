"""Collapse origin-district Open-Meteo weeks onto each crop-market row."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.build.scope import WEATHER_KEEP

# Several weather points sit in one DCS district. Split that district's extent
# across them so Badulla is not triple-counted.
# Weather locations come from origin_map[crop][market]["weather"], not from the
# market town. Weights use the lagged finished season's extent when it exists,
# and equal weights only when that extent is missing. They are not trade shares.
ORIGIN_WEIGHT_METHOD = "extent_weighted_weather"
WEATHER_DISTRICT = {
    "nuwara_eliya": "Nuwara Eliya",
    "badulla": "Badulla",
    "welimada": "Badulla",
    "bandarawela": "Badulla",
    "matale": "Matale",
    "kandy": "Kandy",
    "dambulla": "Matale",
    "keppetipola": "Badulla",
}


def collapse_weather(
    panel: pd.DataFrame,
    weather: pd.DataFrame,
    seasonal: pd.DataFrame,
    origin_map: dict,
) -> pd.DataFrame:
    keep = [column for column in WEATHER_KEEP if column in weather.columns and column != "complete"]
    wx = weather.copy()
    wx["period_start"] = pd.to_datetime(wx["period_start"])
    wx = wx.set_index(["location", "period_start"])

    extent = seasonal.set_index(["crop", "district", "year", "season"])["extent_ha"]
    value_columns = {f"weather_{column}": [] for column in keep}
    n_origins: list[int] = []
    complete_min: list[float] = []

    for crop, market, week_start, season_name, season_year in zip(
        panel["crop"],
        panel["market"],
        pd.to_datetime(panel["week_start"]),
        panel["supply_season_name"],
        panel["supply_season_year"],
    ):
        # Origin-map weather points only. The market name is not a location key.
        points = origin_map.get(crop, {}).get(market, {}).get("weather") or []
        weights = _weights(points, crop, season_name, int(season_year), extent)
        totals = {column: 0.0 for column in keep}
        weight_used = {column: 0.0 for column in keep}
        completes = []
        used = 0
        for point, weight in weights:
            key = (point, week_start)
            if key not in wx.index:
                continue
            row = wx.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            used += 1
            if "complete" in row.index and pd.notna(row["complete"]):
                completes.append(float(bool(row["complete"])))
            for column in keep:
                value = row[column]
                if pd.notna(value):
                    totals[column] += float(value) * weight
                    weight_used[column] += weight
        n_origins.append(used)
        complete_min.append(min(completes) if completes else np.nan)
        for column in keep:
            if weight_used[column] > 0:
                value_columns[f"weather_{column}"].append(totals[column] / weight_used[column])
            else:
                value_columns[f"weather_{column}"].append(np.nan)

    out = panel.copy()
    for name, values in value_columns.items():
        out[name] = values
    out["weather_n_origins"] = n_origins
    out["weather_complete_min"] = complete_min
    return out


def _weights(points: list[str], crop: str, season: str, year: int, extent: pd.Series) -> list[tuple[str, float]]:
    if not points:
        return []
    raw: list[tuple[str, float]] = []
    counts: dict[str, int] = {}
    for point in points:
        counts[WEATHER_DISTRICT.get(point, point)] = counts.get(WEATHER_DISTRICT.get(point, point), 0) + 1
    for point in points:
        district = WEATHER_DISTRICT.get(point, point)
        key = (crop, district, year, season)
        value = np.nan
        if key in extent.index:
            got = extent.loc[key]
            value = float(got.iloc[-1] if isinstance(got, pd.Series) else got)
        share = (value / counts[district]) if pd.notna(value) and counts[district] else np.nan
        raw.append((point, share))
    if not any(np.isfinite(weight) and weight > 0 for _, weight in raw):
        return [(point, 1.0) for point, _ in raw]
    return [(point, weight if np.isfinite(weight) and weight > 0 else 0.0) for point, weight in raw]


def confirm_weather_uses_origin_map(origin_map: dict) -> None:
    """Consumer markets must not be joined to a weather point named after the town."""
    for crop, markets in origin_map.items():
        for market, spec in markets.items():
            points = spec.get("weather") or []
            if not points:
                raise AssertionError(f"{crop}/{market} has no origin weather points")
            slug = str(market).strip().lower().replace(" ", "_")
            names = {str(point).strip().lower() for point in points}
            if slug in {"colombo", "meegoda"} and slug in names:
                raise AssertionError(f"{crop}/{market} weather includes the market town")
