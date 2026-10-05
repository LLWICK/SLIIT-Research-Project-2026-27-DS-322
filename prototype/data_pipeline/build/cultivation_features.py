"""Observed monthly cultivation and a labelled synthetic calendar.

Both providers emit the same columns. The synthetic provider is a prototype
sensitivity series. It uses the final DCS extent of the same season, so those
rows must not be joined onto the primary forecasting table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_pipeline.build.cultivation_progress import assert_formula
from data_pipeline.build.join_weather import ORIGIN_WEIGHT_METHOD, confirm_weather_uses_origin_map
from data_pipeline.build.supply_lags import (
    assert_lagged_extent_examples,
    attach_finished_extent_features,
    current_season,
    season_is_finished,
)
from data_pipeline.ingest.harti_excel_prices import harti_week_start

CALENDAR_PATH = Path(__file__).resolve().parents[1] / "maps" / "crop_calendar.yaml"
SOURCE_OBSERVED_MONTHLY = "observed_monthly"
SOURCE_LAGGED_SEASONAL_EXTENT = "lagged_seasonal_extent"
SOURCE_SYNTHETIC_CALENDAR = "synthetic_calendar"
SOURCE_NONE = "none"
SOURCES = (
    SOURCE_OBSERVED_MONTHLY,
    SOURCE_LAGGED_SEASONAL_EXTENT,
    SOURCE_SYNTHETIC_CALENDAR,
    SOURCE_NONE,
)
CULTIVATION_COLUMNS = [
    "crop",
    "market",
    "origin_district",
    "week_start",
    "cultivation_area_asof",
    "cultivation_progress",
    "cultivation_reference_area",
    "cultivation_source",
    "cultivation_is_synthetic",
    "cultivation_scenario",
    "source_season",
    "data_available_date",
    "cultivation_target_exceeded",
]
_EXPECTED_PROFILES = {
    "early": (0.40, 8.0),
    "typical": (0.50, 8.0),
    "late": (0.60, 8.0),
}


def empty_cultivation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "crop": pd.Series(dtype="object"),
            "market": pd.Series(dtype="object"),
            "origin_district": pd.Series(dtype="object"),
            "week_start": pd.Series(dtype="datetime64[ns]"),
            "cultivation_area_asof": pd.Series(dtype="float64"),
            "cultivation_progress": pd.Series(dtype="float64"),
            "cultivation_reference_area": pd.Series(dtype="float64"),
            "cultivation_source": pd.Series(dtype="object"),
            "cultivation_is_synthetic": pd.Series(dtype="bool"),
            "cultivation_scenario": pd.Series(dtype="object"),
            "source_season": pd.Series(dtype="object"),
            "data_available_date": pd.Series(dtype="datetime64[ns]"),
            "cultivation_target_exceeded": pd.Series(dtype="boolean"),
        }
    )


def load_crop_calendar(path: Path | None = None) -> dict:
    loaded = yaml.safe_load((path or CALENDAR_PATH).read_text(encoding="utf-8"))
    profiles = loaded.get("profiles") or {}
    for name in ("early", "typical", "late"):
        profile = profiles.get(name) or {}
        if "midpoint" not in profile or "steepness" not in profile:
            raise ValueError(f"crop calendar profile {name} needs midpoint and steepness")
        midpoint = float(profile["midpoint"])
        steepness = float(profile["steepness"])
        if not 0 < midpoint < 1 or steepness <= 0:
            raise ValueError(f"crop calendar profile {name} is outside the logistic range")
    if loaded.get("default_scenario") not in profiles:
        raise ValueError("default_scenario is not one of the profiles")
    for crop, seasons in (loaded.get("crops") or {}).items():
        for season, window in seasons.items():
            _check_window(str(season), int(window["planting_start_week"]), int(window["planting_end_week"]))
    return loaded


def _check_window(season: str, start: int, end: int) -> None:
    if season == "Yala":
        if not (14 <= start <= end <= 39):
            raise ValueError(f"Yala planting window {start}-{end} is outside HARTI weeks 14-39")
        return
    if season != "Maha":
        raise ValueError(season)
    if end >= start:
        if start < 40 or end > 53:
            raise ValueError(f"Maha planting window {start}-{end} leaves the opening Maha weeks")
        return
    if not (40 <= start <= 53 and 1 <= end <= 13):
        raise ValueError(f"Maha planting window {start}-{end} is outside the Maha season")


def logistic_progress(u, midpoint: float, steepness: float) -> np.ndarray:
    """Logistic F normalized so F(0)=0 and F(1)=1. Midpoint and steepness come from YAML."""

    def raw(x):
        return 1.0 / (1.0 + np.exp(-steepness * (np.asarray(x, dtype=float) - midpoint)))

    baseline = float(raw(0.0))
    span = float(raw(1.0)) - baseline
    if span == 0:
        raise ValueError("logistic endpoints are identical")
    return (raw(u) - baseline) / span


def assert_logistic_and_calendar(calendar: dict | None = None) -> None:
    calendar = calendar or load_crop_calendar()
    if calendar.get("default_scenario") != "typical":
        raise AssertionError(calendar.get("default_scenario"))
    for name, (midpoint, steepness) in _EXPECTED_PROFILES.items():
        profile = calendar["profiles"][name]
        if abs(float(profile["midpoint"]) - midpoint) > 1e-9 or float(profile["steepness"]) != steepness:
            raise AssertionError(profile)
        curve = logistic_progress([0.0, 1.0], float(profile["midpoint"]), float(profile["steepness"]))
        if abs(float(curve[0])) > 1e-9 or abs(float(curve[1]) - 1) > 1e-9:
            raise AssertionError(curve)


class CultivationFeatureProvider:
    """Shared schema for observed monthly progress and the synthetic calendar."""

    columns = CULTIVATION_COLUMNS

    def build(self) -> pd.DataFrame:
        raise NotImplementedError


class ObservedMonthlyProvider(CultivationFeatureProvider):
    """As-of achieved/target. An empty frame when no monthly file is on disk."""

    def __init__(self, monthly: pd.DataFrame | None, panel: pd.DataFrame):
        self.monthly = monthly
        self.panel = panel

    def build(self) -> pd.DataFrame:
        if self.monthly is None or self.monthly.empty:
            return empty_cultivation_frame()
        matched = self.panel.loc[self.panel["cultivation_progress"].notna()].copy()
        if matched.empty:
            return empty_cultivation_frame()
        available = pd.to_datetime(matched["cultivation_asof_date"])
        frame = pd.DataFrame(
            {
                "crop": matched["crop"].to_numpy(),
                "market": matched["market"].to_numpy(),
                "origin_district": pd.NA,
                "week_start": pd.to_datetime(matched["week_start"]).to_numpy(),
                "cultivation_area_asof": pd.to_numeric(matched["cultivation_achieved_ha"], errors="coerce").to_numpy(),
                "cultivation_progress": pd.to_numeric(matched["cultivation_progress"], errors="coerce").to_numpy(),
                "cultivation_reference_area": pd.to_numeric(matched["cultivation_target_ha"], errors="coerce").to_numpy(),
                "cultivation_source": SOURCE_OBSERVED_MONTHLY,
                "cultivation_is_synthetic": False,
                "cultivation_scenario": pd.NA,
                "source_season": available.dt.strftime("%Y-%m").to_numpy(),
                "data_available_date": available.to_numpy(),
                "cultivation_target_exceeded": matched["cultivation_target_exceeded"].to_numpy(),
            }
        )
        return frame.loc[:, CULTIVATION_COLUMNS].reset_index(drop=True)


class SyntheticCalendarProvider(CultivationFeatureProvider):
    """Final DCS extent times a configured logistic. Prototype and sensitivity only."""

    def __init__(self, seasonal: pd.DataFrame, origin_map: dict, calendar_path: Path | None = None):
        self.seasonal = seasonal
        self.origin_map = origin_map
        self.calendar = load_crop_calendar(calendar_path)

    def build(self) -> pd.DataFrame:
        lookup = _district_extent(self.seasonal)
        districts = _origin_districts(self.origin_map)
        year_min = int(self.seasonal["year"].min())
        year_max = int(self.seasonal["year"].max())
        week_starts = _week_starts(range(year_min, year_max + 2))
        columns: dict[str, list] = {name: [] for name in CULTIVATION_COLUMNS}
        for crop, seasons in self.calendar["crops"].items():
            for district in districts.get(str(crop), []):
                district_key = str(district).strip().casefold()
                for season, window in seasons.items():
                    start_week = int(window["planting_start_week"])
                    end_week = int(window["planting_end_week"])
                    for season_year in range(year_min, year_max + 1):
                        extent = lookup.get((str(crop).strip().lower(), district_key, season_year, str(season)))
                        if extent is None:
                            continue
                        slots = _planting_slots(str(season), season_year, start_week, end_week, week_starts)
                        if not slots:
                            continue
                        available = _available_monday(str(season), season_year, week_starts)
                        u = np.array([1.0]) if len(slots) == 1 else np.linspace(0.0, 1.0, len(slots))
                        for scenario, profile in self.calendar["profiles"].items():
                            progress = logistic_progress(u, float(profile["midpoint"]), float(profile["steepness"]))
                            for (week_start, _year, _week), value in zip(slots, progress):
                                columns["crop"].append(crop)
                                columns["market"].append(pd.NA)
                                columns["origin_district"].append(district)
                                columns["week_start"].append(week_start)
                                columns["cultivation_area_asof"].append(float(extent) * float(value))
                                columns["cultivation_progress"].append(float(value))
                                columns["cultivation_reference_area"].append(float(extent))
                                columns["cultivation_source"].append(SOURCE_SYNTHETIC_CALENDAR)
                                columns["cultivation_is_synthetic"].append(True)
                                columns["cultivation_scenario"].append(scenario)
                                columns["source_season"].append(f"{season}_{season_year}")
                                columns["data_available_date"].append(available)
                                columns["cultivation_target_exceeded"].append(pd.NA)
        if not columns["crop"]:
            return empty_cultivation_frame()
        frame = pd.DataFrame(columns)
        frame["week_start"] = pd.to_datetime(frame["week_start"])
        frame["data_available_date"] = pd.to_datetime(frame["data_available_date"])
        frame["cultivation_is_synthetic"] = frame["cultivation_is_synthetic"].astype(bool)
        frame = frame.sort_values(
            ["crop", "origin_district", "source_season", "week_start", "cultivation_scenario"]
        ).reset_index(drop=True)
        return frame.loc[:, CULTIVATION_COLUMNS]


def _district_extent(seasonal: pd.DataFrame) -> dict[tuple, float]:
    frame = seasonal.dropna(subset=["extent_ha"]).drop_duplicates(
        ["crop", "district", "year", "season"], keep="last"
    )
    lookup: dict[tuple, float] = {}
    for crop, district, year, season, extent in zip(
        frame["crop"], frame["district"], frame["year"], frame["season"], frame["extent_ha"]
    ):
        lookup[(str(crop).strip().lower(), str(district).strip().casefold(), int(year), str(season))] = float(extent)
    return lookup


def _origin_districts(origin_map: dict) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for crop, markets in origin_map.items():
        ordered: list[str] = []
        seen: set[str] = set()
        for spec in markets.values():
            for district in spec.get("supply_districts") or []:
                name = str(district).strip()
                if name.casefold() in seen:
                    continue
                seen.add(name.casefold())
                ordered.append(name)
        found[str(crop)] = ordered
    return found


def _week_starts(years) -> dict[tuple[int, int], pd.Timestamp | None]:
    pairs = [(int(year), week) for year in years for week in range(1, 54)]
    starts = harti_week_start(pd.Series([year for year, _week in pairs]), pd.Series([week for _year, week in pairs]))
    cache: dict[tuple[int, int], pd.Timestamp | None] = {}
    for (year, week), start in zip(pairs, starts):
        stamp = pd.Timestamp(start)
        cache[(year, week)] = stamp if int(stamp.year) == year else None
    return cache


def _planting_slots(
    season: str,
    season_year: int,
    start_week: int,
    end_week: int,
    week_starts: dict[tuple[int, int], pd.Timestamp | None],
) -> list[tuple[pd.Timestamp, int, int]]:
    if season == "Yala" or end_week >= start_week:
        pairs = [(season_year, week) for week in range(start_week, end_week + 1)]
    else:
        pairs = [(season_year, week) for week in range(start_week, 54)]
        pairs += [(season_year + 1, week) for week in range(1, end_week + 1)]
    slots = []
    for year, week in pairs:
        start = week_starts.get((year, week))
        if start is None:
            continue
        slots.append((start, year, week))
    return slots


def _available_monday(season: str, season_year: int, week_starts: dict[tuple[int, int], pd.Timestamp | None]) -> pd.Timestamp:
    if season == "Yala":
        key = (season_year, 40)
    elif season == "Maha":
        key = (season_year + 1, 14)
    else:
        raise ValueError(season)
    start = week_starts.get(key)
    if start is None:
        raise ValueError(f"no HARTI week for the end of {season} {season_year}")
    return start


def _market_id(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _shares(frame: pd.DataFrame, column: str) -> float:
    return round(float(frame[column].notna().mean()), 4) if len(frame) else 0.0


def validate_cultivation_outputs(
    table: pd.DataFrame,
    synthetic: pd.DataFrame,
    observed: pd.DataFrame,
    mapping: pd.DataFrame,
    seasonal: pd.DataFrame,
    origin_map: dict,
    monthly_found: bool,
) -> dict:
    """Raise if leakage, zero-filled targets, or a synthetic row entered the primary table."""
    assert_formula()
    assert_lagged_extent_examples()
    calendar = load_crop_calendar()
    assert_logistic_and_calendar(calendar)
    confirm_weather_uses_origin_map(origin_map)

    identity = ["crop_id", "market_id", "forecast_date", "horizon_weeks"]
    duplicate_rows = int(table.duplicated(identity).sum())
    if duplicate_rows:
        raise AssertionError(f"duplicate crop-market-forecast_date-horizon rows: {duplicate_rows}")
    if bool(table["cultivation_is_synthetic"].astype(bool).any()):
        raise AssertionError("primary table contains synthetic cultivation rows")
    if bool(table["cultivation_source"].eq(SOURCE_SYNTHETIC_CALENDAR).any()):
        raise AssertionError("primary table uses synthetic_calendar")
    if bool(table["cultivation_scenario"].notna().any()):
        raise AssertionError("primary table has a cultivation scenario")
    if "cultivation_area_asof" in table.columns:
        raise AssertionError("same-season synthetic area was copied onto the primary table")
    if not monthly_found:
        if bool(table["cultivation_progress_ratio"].notna().any()):
            raise AssertionError("cultivation progress was filled without a monthly file")
        if bool(table["origin_target_hectares"].notna().any()) or bool(table["origin_achieved_hectares"].notna().any()):
            raise AssertionError("target or achieved hectares were invented")
        if len(observed):
            raise AssertionError("observed frame should be empty when no monthly file exists")
    elif len(observed):
        if not bool(observed["cultivation_is_synthetic"].eq(False).all()):
            raise AssertionError("observed rows must not be synthetic")
        if not bool(observed["cultivation_source"].eq(SOURCE_OBSERVED_MONTHLY).all()):
            raise AssertionError(observed["cultivation_source"].unique())
        if bool((observed["cultivation_progress"] > 1).any()) and not bool(observed["cultivation_target_exceeded"].fillna(False).any()):
            raise AssertionError("raw progress above 1 was not flagged")

    unobserved = ~table["target_is_observed"].astype(bool)
    if bool(table.loc[unobserved, "target_price_lkr_kg"].notna().any()):
        raise AssertionError("unobserved target price was filled")
    prices = pd.to_numeric(table["target_price_lkr_kg"], errors="coerce")
    if bool((prices == 0).any()):
        raise AssertionError("target price was zero-filled")

    _assert_lagged_matches_table(table, seasonal, origin_map)
    _assert_synthetic(synthetic, seasonal)

    if bool(mapping["supply_weight"].notna().any()) or bool(mapping["weight"].notna().any()):
        raise AssertionError("supply weight was filled")
    if not bool(mapping["origin_weight_method"].eq(ORIGIN_WEIGHT_METHOD).all()):
        raise AssertionError(mapping["origin_weight_method"].unique())
    if ORIGIN_WEIGHT_METHOD != "extent_weighted_weather":
        raise AssertionError(ORIGIN_WEIGHT_METHOD)

    forecast = table.drop_duplicates(["crop_id", "market_id", "forecast_date"])
    series_share = forecast.groupby(["crop_id", "market_id"])["previous_season_extent"].apply(
        lambda series: float(series.notna().mean())
    )
    empty_series = [f"{crop}/{market}" for (crop, market), share in series_share.items() if share == 0]
    partial_series = [
        f"{crop}/{market} ({share:.3f})"
        for (crop, market), share in series_share.sort_values().items()
        if share < 1
    ]
    return {
        "monthly_found": monthly_found,
        "observed_rows": int(len(observed)),
        "synthetic_rows": int(len(synthetic)),
        "synthetic_scenarios": sorted(synthetic["cultivation_scenario"].dropna().unique().tolist()),
        "primary_rows": int(len(table)),
        "progress_nonnull": int(table["cultivation_progress_ratio"].notna().sum()),
        "primary_synthetic": bool(table["cultivation_is_synthetic"].astype(bool).any()),
        "previous_season_extent_nonnull_share": _shares(forecast, "previous_season_extent"),
        "same_season_previous_year_extent_nonnull_share": _shares(forecast, "same_season_previous_year_extent"),
        "historical_mean_season_extent_nonnull_share": _shares(forecast, "historical_mean_season_extent"),
        "extent_change_vs_previous_season_nonnull_share": _shares(forecast, "extent_change_vs_previous_season"),
        "origin_rainfall_1w_mm_nonnull_share": _shares(table, "origin_rainfall_1w_mm"),
        "series_with_no_lagged_extent": empty_series,
        "series_with_partial_lagged_extent": partial_series,
        "origin_weight_method": ORIGIN_WEIGHT_METHOD,
        "duplicate_rows": duplicate_rows,
        "default_scenario": calendar["default_scenario"],
        "calendar": calendar,
    }


def _assert_lagged_matches_table(table: pd.DataFrame, seasonal: pd.DataFrame, origin_map: dict) -> None:
    horizon_span = table.groupby(["crop_id", "market_id", "forecast_date"])["previous_season_extent"].nunique(dropna=False)
    if bool((horizon_span > 1).any()):
        raise AssertionError("lagged extent changes across horizons of the same forecast")
    names = { _market_id(market): market for markets in origin_map.values() for market in markets }
    check = table.drop_duplicates(["crop_id", "market_id", "forecast_date"]).copy()
    check["crop"] = check["crop_id"]
    check["market"] = check["market_id"].map(names)
    if bool(check["market"].isna().any()):
        raise AssertionError("forecast market is not in the origin map")
    check["year"] = pd.to_datetime(check["forecast_date"]).dt.year
    check["week"] = check["forecast_week_of_year"].astype(int)
    rebuilt = attach_finished_extent_features(check, seasonal, origin_map)
    for column in (
        "previous_season_extent",
        "same_season_previous_year_extent",
        "historical_mean_season_extent",
        "extent_change_vs_previous_season",
    ):
        actual = pd.to_numeric(check[column], errors="coerce").reset_index(drop=True)
        expected = pd.to_numeric(rebuilt[column], errors="coerce").reset_index(drop=True)
        both_missing = actual.isna() & expected.isna()
        close = (actual - expected).abs().lt(1e-4)
        if not bool((both_missing | close).all()):
            raise AssertionError(f"{column} does not match a finished-season recomputation")
    for year, week, name, season_year in zip(
        rebuilt["year"],
        rebuilt["week"],
        [current_season(int(year), int(week))[0] for year, week in zip(rebuilt["year"], rebuilt["week"])],
        [current_season(int(year), int(week))[1] for year, week in zip(rebuilt["year"], rebuilt["week"])],
    ):
        if season_is_finished(name, int(season_year), int(year), int(week)):
            raise AssertionError(f"current season {name} {season_year} is already finished at {year} W{week}")


def _assert_synthetic(synthetic: pd.DataFrame, seasonal: pd.DataFrame) -> None:
    if synthetic.empty:
        raise AssertionError("synthetic cultivation frame is empty")
    if not bool(synthetic["cultivation_is_synthetic"].astype(bool).all()):
        raise AssertionError("synthetic rows must all be marked synthetic")
    if not bool(synthetic["cultivation_source"].eq(SOURCE_SYNTHETIC_CALENDAR).all()):
        raise AssertionError(synthetic["cultivation_source"].unique())
    if not bool(synthetic["cultivation_scenario"].isin(["early", "typical", "late"]).all()):
        raise AssertionError("synthetic scenario is outside early|typical|late")
    identity = ["crop", "origin_district", "week_start", "cultivation_scenario", "source_season"]
    if bool(synthetic.duplicated(identity).any()):
        raise AssertionError("duplicate synthetic rows")
    if bool((pd.to_datetime(synthetic["week_start"]) >= pd.to_datetime(synthetic["data_available_date"])).any()):
        raise AssertionError("a synthetic week is on or after the season's extent became available")
    progress = pd.to_numeric(synthetic["cultivation_progress"], errors="coerce")
    reference = pd.to_numeric(synthetic["cultivation_reference_area"], errors="coerce")
    area = pd.to_numeric(synthetic["cultivation_area_asof"], errors="coerce")
    if bool(progress.isna().any()) or bool(progress.lt(-1e-8).any()) or bool(progress.gt(1 + 1e-8).any()):
        raise AssertionError("synthetic progress left the unit interval")
    if bool((area - reference * progress).abs().gt(1e-4).any()):
        raise AssertionError("synthetic area is not final extent times F(u)")
    ordered = synthetic.sort_values("week_start")
    grouped = ordered.groupby(["crop", "origin_district", "source_season", "cultivation_scenario"], sort=False)
    if bool(grouped["cultivation_progress"].first().abs().gt(1e-6).any()):
        raise AssertionError("synthetic progress does not start at 0")
    if bool((grouped["cultivation_progress"].last() - 1).abs().gt(1e-6).any()):
        raise AssertionError("synthetic progress does not end at 1")
    scenario_count = synthetic.groupby(["crop", "origin_district", "source_season"])["cultivation_scenario"].nunique()
    if bool(scenario_count.ne(3).any()):
        raise AssertionError("a season is missing one of the three scenarios")
    lookup = _district_extent(seasonal)
    parsed = synthetic["source_season"].astype(str).str.rsplit("_", n=1, expand=True)
    expected = []
    for crop, district, season, year in zip(synthetic["crop"], synthetic["origin_district"], parsed[0], parsed[1]):
        expected.append(lookup.get((str(crop).strip().lower(), str(district).strip().casefold(), int(year), str(season))))
    expected_series = pd.Series(expected, index=synthetic.index, dtype="float64")
    if bool(expected_series.isna().any()) or bool((expected_series - reference).abs().gt(1e-4).any()):
        raise AssertionError("synthetic reference extent is not the real DCS extent_ha")


def render_modes_audit(summary: dict) -> str:
    calendar = summary["calendar"]
    window_lines = []
    for crop, seasons in calendar["crops"].items():
        for season, window in seasons.items():
            window_lines.append(
                f"- {crop} {season}: planting weeks {window['planting_start_week']}-{window['planting_end_week']}"
            )
    profile_lines = [
        f"- {name}: midpoint {profile['midpoint']}, steepness {profile['steepness']}"
        for name, profile in calendar["profiles"].items()
    ]
    monthly = "found" if summary["monthly_found"] else "not found"
    empty_series = ", ".join(summary["series_with_no_lagged_extent"]) or "none"
    partial_series = ", ".join(summary.get("series_with_partial_lagged_extent") or []) or "none"
    return "\n".join(
        [
            "# Cultivation modes",
            "",
            "Member 2 keeps two cultivation providers on one schema. They are not interchangeable, and they are not both inputs to the primary forecasting table.",
            "",
            "## Observed monthly progress",
            "",
            f"A district-month file with year, month, crop, district, target_ha, and achieved_ha was {monthly}.",
            f"Observed provider rows written: {summary['observed_rows']}.",
            "When that file is absent the observed provider writes an empty frame. `cultivation_progress_ratio`, `origin_target_hectares`, and `origin_achieved_hectares` stay null. Nothing is invented from price, rainfall, temperature, or zeros.",
            "If a file is present, progress is summed achieved hectares divided by summed target hectares. The ratio is not clipped at 1. `cultivation_target_exceeded` is true only when that raw ratio is above 1. A forecast Monday uses a report only when its available date is on or before that Monday. Later months are not interpolated.",
            "",
            "## Lagged seasonal extent on the primary table",
            "",
            "These columns are real DCS extent for seasons that have already finished. They are not cultivation progress.",
            "",
            "- `previous_season_extent`: last finished Yala or Maha, summed over the market's supply districts. Yala of year S is finished only after HARTI week 39 of S. Maha of year S is finished only after week 13 of S+1.",
            "- `same_season_previous_year_extent`: the same season one year earlier, and only when that season has finished.",
            "- `historical_mean_season_extent`: arithmetic mean of finished Yala and Maha extents before the forecast week. The current unfinished season is excluded.",
            "- `extent_change_vs_previous_season`: `previous_season_extent` minus the finished season immediately before it, in hectares.",
            "",
            f"On the rebuilt primary table, `cultivation_progress_ratio` non-null rows: {summary['progress_nonnull']}.",
            f"`previous_season_extent` non-null share of forecasts: {summary['previous_season_extent_nonnull_share']}.",
            f"`same_season_previous_year_extent` non-null share: {summary['same_season_previous_year_extent_nonnull_share']}.",
            f"`historical_mean_season_extent` non-null share: {summary['historical_mean_season_extent_nonnull_share']}.",
            f"`extent_change_vs_previous_season` non-null share: {summary['extent_change_vs_previous_season_nonnull_share']}.",
            f"Crop-markets with no lagged extent in this table: {empty_series}.",
            f"Series whose last finished season is missing for some weeks: {partial_series}. A blank DCS season stays null. A recorded 0 ha stays 0 ha and is not used to fill a blank season.",
            "`cultivation_source` is `lagged_seasonal_extent` where any finished-extent feature is present, and `none` where none of them are. `cultivation_is_synthetic` is false. `cultivation_scenario` is null. `cultivation_source_name` stays empty when no monthly report exists.",
            "The same season's final extent is not an input feature on this table.",
            "",
            "## Synthetic calendar (prototype and sensitivity only)",
            "",
            "Synthetic rows use the real final DCS `extent_ha` for that crop, district, and season, multiplied by a logistic planting curve. The curve is a research assumption. It is not an observed planting date and it is not achieved hectares divided by target hectares.",
            "",
            "The planting windows below are configuration in `prototype/data_pipeline/maps/crop_calendar.yaml`. They sit inside the existing HARTI season rule (Yala weeks 14-39, Maha the other weeks). They are not a record of when farmers planted.",
            "",
            *window_lines,
            "",
            "Normalized u runs from 0 to 1 across the planting window. F(u) = (raw(u) - raw(0)) / (raw(1) - raw(0)), where raw(u) = 1 / (1 + exp(-k (u - m))).",
            "",
            *profile_lines,
            "",
            f"The default scenario is `{summary['default_scenario']}`. The parquet contains all three scenarios so a sensitivity check can compare them.",
            f"Synthetic rows: {summary['synthetic_rows']}. Scenarios: {', '.join(summary['synthetic_scenarios'])}.",
            "Every synthetic row has `cultivation_source = synthetic_calendar` and `cultivation_is_synthetic = true`. `cultivation_progress` in this file is F(u), not achieved/target. `cultivation_reference_area` is that same season's final extent. `data_available_date` is the first HARTI Monday after the season has finished, which is after `week_start` for every planting-window row.",
            "",
            "The file is `processed/latest/cultivation_synthetic.parquet`. It is not joined to `processed/latest/training_table.parquet`.",
            "",
            "## Experiment names",
            "",
            "A, B, C, and S now use the proposal names.",
            "",
            "- A is price lags and seasonal features only. It has no weather, no macros, no cultivation progress, and no lagged extent.",
            "- B is A plus origin-aligned historical weather, real diesel, USD/LKR, and inflation, plus lagged finished-season extent. Lagged extent is a historical production feature. It is not current cultivation progress.",
            "- C is B plus observed within-season cultivation progress, summed achieved hectares over summed target hectares across supply districts, joined as-of. Ratios above 1 stay above 1, and `cultivation_target_exceeded` is set when the raw ratio is above 1.",
            "- S is the synthetic calendar curve (early, typical, late). It is a separate prototype and sensitivity experiment. Its scores are not stored in the A/B/C accuracy table.",
            "",
            "## What this does not show",
            "",
            "Synthetic calendar progress is experiment S. It is prototype/sensitivity only and must not be cited as proof that observed cultivation progress improves accuracy.",
            "S scores are not written into the A/B/C accuracy table.",
            "",
            "## Weather weights",
            "",
            f"`origin_weight_method` is `{summary['origin_weight_method']}`.",
            "The weather join reads origin points from `origin_map` (`weather`), not the market town. Colombo and Meegoda are not given a weather point named after the town. Weights use the lagged finished season's district extent, split across weather points that share a district. If that extent is missing, the join falls back to equal weights. `supply_weight` stays null. Measured trade shares such as 0.60/0.30/0.10 are not used.",
            "`origin_rainfall_1w_mm` is the origin rainfall of the one complete week before the forecast Monday (`shift(1)`). It is not filled with zero. The four-week and eight-week sums still start from that same shift. `weather_missing_flag` remains the missingness of the four-week sum.",
            "",
        ]
    )
