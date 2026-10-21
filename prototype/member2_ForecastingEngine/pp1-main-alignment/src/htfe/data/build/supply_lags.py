"""As-of lagged Yala/Maha supply. A week never sees a season that has not finished."""
from __future__ import annotations

import bisect

import pandas as pd


def last_finished_season(year: int, week: int) -> tuple[str, int, str]:
    """Return (season, dcs_year, label) of the latest season finished before this week.

    Yala of year S is finished after HARTI week 39 of S.
    Maha of year S is finished after HARTI week 13 of S+1.
    """
    best: tuple[int, int, str, int, str] | None = None
    for season_year in range(year - 3, year + 1):
        # Yala S ends at week 39 of S.
        if year > season_year or (year == season_year and week > 39):
            key = (season_year, 39, "Yala", season_year, f"Yala_{season_year}")
            if best is None or key[:2] > best[:2]:
                best = key
        # Maha S ends at week 13 of S+1.
        end_year = season_year + 1
        if year > end_year or (year == end_year and week > 13):
            key = (end_year, 13, "Maha", season_year, f"Maha_{season_year}")
            if best is None or key[:2] > best[:2]:
                best = key
    if best is None:
        return "Maha", year - 2, f"Maha_{year - 2}"
    return best[2], best[3], best[4]


def season_of_week(week: int) -> str:
    if 14 <= int(week) <= 39:
        return "Yala"
    return "Maha"


def current_season(year: int, week: int) -> tuple[str, int]:
    """Season that contains this HARTI week. Maha of year S runs into week 13 of S+1."""
    year = int(year)
    week = int(week)
    if 14 <= week <= 39:
        return "Yala", year
    if week <= 13:
        return "Maha", year - 1
    return "Maha", year


def season_is_finished(season: str, season_year: int, year: int, week: int) -> bool:
    """True only after the season's closing week, using the same cut as last_finished_season."""
    if season == "Yala":
        end = (int(season_year), 39)
    elif season == "Maha":
        end = (int(season_year) + 1, 13)
    else:
        raise ValueError(season)
    return (int(year), int(week)) > end


def season_immediately_before(season: str, season_year: int) -> tuple[str, int]:
    if season == "Yala":
        return "Maha", int(season_year) - 1
    if season == "Maha":
        return "Yala", int(season_year)
    raise ValueError(season)


def attach_supply(
    panel: pd.DataFrame,
    seasonal: pd.DataFrame,
    origin_map: dict,
) -> pd.DataFrame:
    """Add lagged extent/production and cultivation intensity for each panel row."""
    lookup = seasonal.set_index(["crop", "district", "year", "season"])
    finished = [
        last_finished_season(int(year), int(week))
        for year, week in zip(panel["year"], panel["week"])
    ]
    panel = panel.copy()
    panel["supply_season_name"] = [item[0] for item in finished]
    panel["supply_season_year"] = [item[1] for item in finished]
    panel["supply_season_label"] = [item[2] for item in finished]
    panel["season"] = panel["week"].map(season_of_week)

    extents: list[float | None] = []
    productions: list[float | None] = []
    for crop, market, season_name, season_year in zip(
        panel["crop"], panel["market"], panel["supply_season_name"], panel["supply_season_year"]
    ):
        spec = origin_map.get(crop, {}).get(market, {})
        districts = spec.get("supply_districts") or []
        extent_sum = 0.0
        production_sum = 0.0
        seen = False
        for district in districts:
            key = (crop, district, int(season_year), season_name)
            if key not in lookup.index:
                continue
            row = lookup.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            if pd.notna(row["extent_ha"]):
                extent_sum += float(row["extent_ha"])
                seen = True
            if pd.notna(row["production_mt"]):
                production_sum += float(row["production_mt"])
        extents.append(extent_sum if seen else None)
        productions.append(production_sum if seen else None)

    panel["supply_extent_ha_lag1_season"] = extents
    panel["supply_production_mt_lag1_season"] = productions
    panel["commitment_source"] = "historical_dcs_proxy"
    panel["cultivation_intensity"] = _intensity(panel)
    return panel


def _intensity(panel: pd.DataFrame) -> pd.Series:
    """Extent relative to the median of the same season in 2015–2019, per crop and market."""
    base = panel.loc[
        panel["supply_season_year"].between(2015, 2019),
        ["crop", "market", "supply_season_name", "supply_season_year", "supply_extent_ha_lag1_season"],
    ].drop_duplicates(["crop", "market", "supply_season_name", "supply_season_year"])
    benchmark = (
        base.groupby(["crop", "market", "supply_season_name"])["supply_extent_ha_lag1_season"]
        .median()
        .rename("benchmark_ha")
    )
    joined = panel.merge(
        benchmark.reset_index(),
        on=["crop", "market", "supply_season_name"],
        how="left",
    )
    intensity = joined["supply_extent_ha_lag1_season"] / joined["benchmark_ha"].replace(0, pd.NA)
    return intensity.to_numpy()


def _extent_lookup(seasonal: pd.DataFrame) -> dict[tuple, float]:
    lookup: dict[tuple, float] = {}
    frame = seasonal.dropna(subset=["extent_ha"])
    for crop, district, year, season, extent in zip(
        frame["crop"], frame["district"], frame["year"], frame["season"], frame["extent_ha"]
    ):
        key = (str(crop).strip().lower(), str(district).strip().casefold(), int(year), str(season))
        lookup[key] = float(extent)
    return lookup


def _sum_districts(lookup: dict, crop: str, districts: list, year: int, season: str) -> float | None:
    total = 0.0
    seen = False
    crop_key = str(crop).strip().lower()
    for district in districts:
        value = lookup.get((crop_key, str(district).strip().casefold(), int(year), season))
        if value is None:
            continue
        total += value
        seen = True
    return total if seen else None


def _season_timeline(year_min: int, year_max: int) -> list[tuple[str, int, int, int]]:
    events = []
    for season_year in range(int(year_min), int(year_max) + 1):
        events.append(("Yala", season_year, season_year, 39))
        events.append(("Maha", season_year, season_year + 1, 13))
    events.sort(key=lambda item: (item[2], item[3]))
    return events


def _markets(origin_map: dict, crop: str) -> dict:
    markets = origin_map.get(crop)
    if markets is None:
        markets = origin_map.get(str(crop).strip().lower(), {})
    return markets or {}


def attach_finished_extent_features(
    panel: pd.DataFrame,
    seasonal: pd.DataFrame,
    origin_map: dict,
) -> pd.DataFrame:
    """Leak-safe DCS extent features. The current unfinished season is never an input.

    previous_season_extent is the last finished Yala or Maha, summed over supply
    districts (same rule as last_finished_season). historical_mean_season_extent
    is the arithmetic mean of every finished season with a measured extent before
    the forecast week. Missing extent stays missing. It is not filled with zero.
    """
    out = panel.copy()
    lookup = _extent_lookup(seasonal)
    year_min = int(seasonal["year"].min()) - 1
    year_max = int(seasonal["year"].max())
    timeline = _season_timeline(year_min, year_max)
    ends = [(item[2], item[3]) for item in timeline]
    position = {(item[0], item[1]): index for index, item in enumerate(timeline)}
    series_cache: dict[tuple, list[float | None]] = {}

    def series_for(crop: str, market: str) -> list[float | None]:
        key = (str(crop), str(market))
        cached = series_cache.get(key)
        if cached is not None:
            return cached
        districts = _markets(origin_map, crop).get(market, {}).get("supply_districts") or []
        values = [
            _sum_districts(lookup, crop, districts, season_year, season)
            for season, season_year, _end_year, _end_week in timeline
        ]
        series_cache[key] = values
        return values

    previous: list[float] = []
    same_year: list[float] = []
    historical: list[float] = []
    change: list[float] = []
    for crop, market, year, week in zip(out["crop"], out["market"], out["year"], out["week"]):
        year = int(year)
        week = int(week)
        values = series_for(crop, market)
        index = bisect.bisect_left(ends, (year, week)) - 1
        finished_name, finished_year, _label = last_finished_season(year, week)
        if index < 0 or timeline[index][0] != finished_name or timeline[index][1] != finished_year:
            raise AssertionError(
                f"lagged season mismatch at {year} W{week}: timeline={None if index < 0 else timeline[index][:2]} "
                f"last_finished={(finished_name, finished_year)}"
            )
        if not season_is_finished(finished_name, finished_year, year, week):
            raise AssertionError(f"{finished_name} {finished_year} is not finished at {year} W{week}")
        if current_season(year, week) == (finished_name, finished_year):
            raise AssertionError(f"current season used as lagged extent at {year} W{week}")
        latest = values[index]
        prior = values[index - 1] if index else None
        before = season_immediately_before(finished_name, finished_year)
        if index and timeline[index - 1][:2] != before:
            raise AssertionError(f"season before {finished_name} {finished_year} is not {before}")
        previous.append(float("nan") if latest is None else float(latest))
        if latest is None or prior is None:
            change.append(float("nan"))
        else:
            change.append(float(latest) - float(prior))
        finished_values = [value for value in values[: index + 1] if value is not None]
        historical.append(sum(finished_values) / len(finished_values) if finished_values else float("nan"))
        current_name, current_year = current_season(year, week)
        prior_same = (current_name, current_year - 1)
        same_value = None
        if season_is_finished(prior_same[0], prior_same[1], year, week):
            slot = position.get(prior_same)
            if slot is not None:
                same_value = values[slot]
        same_year.append(float("nan") if same_value is None else float(same_value))

    out["previous_season_extent"] = previous
    out["same_season_previous_year_extent"] = same_year
    out["historical_mean_season_extent"] = historical
    out["extent_change_vs_previous_season"] = change
    if "supply_extent_ha_lag1_season" in out.columns:
        left = pd.to_numeric(out["supply_extent_ha_lag1_season"], errors="coerce")
        right = pd.to_numeric(out["previous_season_extent"], errors="coerce")
        mismatch = ~(left.isna() & right.isna()) & (left - right).abs().gt(1e-4)
        if bool(mismatch.fillna(False).any()):
            raise AssertionError(f"previous_season_extent diverges from the panel lag on {int(mismatch.sum())} rows")
    return out


def assert_lagged_extent_examples() -> None:
    """A week inside Yala or Maha must not see that season's own final extent."""
    seasonal = pd.DataFrame(
        [
            {"crop": "carrot", "district": "Nuwara Eliya", "year": 2018, "season": "Yala", "extent_ha": 100},
            {"crop": "carrot", "district": "Badulla", "year": 2018, "season": "Yala", "extent_ha": 50},
            {"crop": "carrot", "district": "Nuwara Eliya", "year": 2018, "season": "Maha", "extent_ha": 80},
            {"crop": "carrot", "district": "Badulla", "year": 2018, "season": "Maha", "extent_ha": 40},
            {"crop": "carrot", "district": "Nuwara Eliya", "year": 2019, "season": "Yala", "extent_ha": 110},
            {"crop": "carrot", "district": "Badulla", "year": 2019, "season": "Yala", "extent_ha": 40},
            {"crop": "carrot", "district": "Nuwara Eliya", "year": 2019, "season": "Maha", "extent_ha": 90},
            {"crop": "carrot", "district": "Badulla", "year": 2019, "season": "Maha", "extent_ha": 30},
            {"crop": "carrot", "district": "Nuwara Eliya", "year": 2020, "season": "Yala", "extent_ha": 999},
            {"crop": "carrot", "district": "Badulla", "year": 2020, "season": "Yala", "extent_ha": 999},
        ]
    )
    origin = {"carrot": {"Dambulla": {"supply_districts": ["Nuwara Eliya", "Badulla"]}}}
    panel = pd.DataFrame(
        [
            {"crop": "carrot", "market": "Dambulla", "year": 2020, "week": 10, "week_start": "2020-03-02"},
            {"crop": "carrot", "market": "Dambulla", "year": 2020, "week": 20, "week_start": "2020-05-11"},
            {"crop": "carrot", "market": "Dambulla", "year": 2019, "week": 39, "week_start": "2019-09-23"},
            {"crop": "carrot", "market": "Dambulla", "year": 2019, "week": 40, "week_start": "2019-09-30"},
        ]
    )
    checked = attach_finished_extent_features(panel, seasonal, origin)
    by_week = checked.set_index("week")
    # Week 10 of 2020 is still Maha 2019. Last finished season is Yala 2019 = 150.
    if float(by_week.loc[10, "previous_season_extent"]) != 150:
        raise AssertionError(by_week.loc[10, "previous_season_extent"])
    if float(by_week.loc[10, "same_season_previous_year_extent"]) != 120:
        raise AssertionError(by_week.loc[10, "same_season_previous_year_extent"])
    if abs(float(by_week.loc[10, "historical_mean_season_extent"]) - 140) > 1e-9:
        raise AssertionError(by_week.loc[10, "historical_mean_season_extent"])
    # Week 20 is inside Yala 2020. The 999 ha final extent must not be used.
    if float(by_week.loc[20, "previous_season_extent"]) != 120:
        raise AssertionError(by_week.loc[20, "previous_season_extent"])
    if float(by_week.loc[20, "extent_change_vs_previous_season"]) != 120 - 150:
        raise AssertionError(by_week.loc[20, "extent_change_vs_previous_season"])
    if float(by_week.loc[20, "same_season_previous_year_extent"]) != 150:
        raise AssertionError(by_week.loc[20, "same_season_previous_year_extent"])
    if abs(float(by_week.loc[20, "historical_mean_season_extent"]) - 135) > 1e-9:
        raise AssertionError(by_week.loc[20, "historical_mean_season_extent"])
    # Week 39 has not finished Yala. Week 40 has.
    if float(by_week.loc[39, "previous_season_extent"]) != 120:
        raise AssertionError(by_week.loc[39, "previous_season_extent"])
    if float(by_week.loc[40, "previous_season_extent"]) != 150:
        raise AssertionError(by_week.loc[40, "previous_season_extent"])
