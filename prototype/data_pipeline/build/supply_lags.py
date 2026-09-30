"""As-of lagged Yala/Maha supply. A week never sees a season that has not finished."""
from __future__ import annotations

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
