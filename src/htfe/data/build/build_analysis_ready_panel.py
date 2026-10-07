"""Build the versioned wholesale weekly analysis panel.

Usage (from the prototype folder):
  python -m htfe.data.build.build_analysis_ready_panel
"""
from __future__ import annotations

import argparse
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from htfe.data.build.holiday_flags import flags_for_weeks
from htfe.data.build.join_weather import collapse_weather
from htfe.data.build.missingness import apply_missingness
from htfe.data.build.scope import ALL_MARKETS, CROPS
from htfe.data.build.supply_lags import attach_supply
from htfe.config import processed_dir
from htfe.data.ingest.harti_excel_prices import harti_week_start
from htfe.data.schemas.panel_schema import validate

MAP_DIR = Path(__file__).resolve().parents[1] / "maps"


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((MAP_DIR / name).read_text(encoding="utf-8"))


def filter_harti(weekly: pd.DataFrame) -> pd.DataFrame:
    frame = weekly.loc[
        (weekly["price_type"] == "wholesale")
        & (weekly["frequency"] == "weekly")
        & weekly["commodity"].isin(CROPS)
        & weekly["location"].isin(ALL_MARKETS)
    ].copy()
    frame["week_start"] = pd.to_datetime(frame["period_start"])
    frame = frame.rename(
        columns={
            "commodity": "crop",
            "location": "market",
            "period": "week",
            "price_lkr_per_kg": "price",
        }
    )
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame.loc[frame["price"] <= 0, "price"] = pd.NA
    keep = [
        "crop",
        "market",
        "price_type",
        "year",
        "week",
        "week_start",
        "price",
        "quality_flag",
        "is_outlier_flag",
    ]
    frame = frame[keep]
    frame = frame.sort_values(["crop", "market", "week_start"]).drop_duplicates(
        ["crop", "market", "week_start"], keep="last"
    )
    return frame.reset_index(drop=True)


def _harti_calendar(year_min: int, year_max: int) -> pd.DataFrame:
    """Every HARTI week, including weeks the workbook left blank for every market."""
    rows = []
    for year in range(year_min, year_max + 1):
        for week in range(1, 54):
            start = harti_week_start(pd.Series([year]), pd.Series([week])).iloc[0]
            if int(start.year) != year:
                continue
            rows.append({"year": year, "week": week, "week_start": pd.Timestamp(start)})
    return pd.DataFrame(rows)


def _calendar_grid(scoped: pd.DataFrame) -> pd.DataFrame:
    weeks = _harti_calendar(2016, 2025)
    series = scoped[["crop", "market"]].drop_duplicates()
    grid = series.merge(weeks, how="cross")
    observed = scoped.drop(columns=["year", "week"])
    merged = grid.merge(observed, on=["crop", "market", "week_start"], how="left")
    merged["price_type"] = "wholesale"
    merged["quality_flag"] = merged["quality_flag"].fillna("missing")
    merged["is_outlier_flag"] = merged["is_outlier_flag"].fillna(False).astype(bool)
    merged["year"] = merged["year"].astype(int)
    merged["week"] = merged["week"].astype(int)
    return merged


def _dri(panel: pd.DataFrame) -> pd.Series:
    observed = panel["is_observed"].astype(float)
    return (
        observed.groupby([panel["crop"], panel["market"]], sort=False)
        .transform(lambda series: series.rolling(12, min_periods=1).mean())
    )


def build(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    processed = processed_dir()
    weekly = frame if frame is not None else pd.read_csv(processed / "harti_prices_weekly_long.csv", low_memory=False)
    scoped = filter_harti(weekly)
    from htfe.config import OUTPUTS

    interim = OUTPUTS / "panel" / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    scoped.to_csv(interim / "harti_scope_weekly.csv", index=False)

    seasonal_path = processed / "dcs_highland_seasonal.csv"
    if not seasonal_path.exists():
        from htfe.data.build.parse_dcs_highland import build as parse_dcs

        parse_dcs().to_csv(seasonal_path, index=False)
    seasonal = pd.read_csv(seasonal_path)
    origin_map = _load_yaml("origin_map.yaml")

    panel = _calendar_grid(scoped)
    panel = attach_supply(panel, seasonal, origin_map)
    weather = pd.read_csv(processed / "weather_weekly_open_meteo.csv")
    panel = collapse_weather(panel, weather, seasonal, origin_map)
    holidays = flags_for_weeks(panel["week_start"])
    panel = panel.merge(holidays, on="week_start", how="left")
    panel = apply_missingness(panel)
    panel["dri"] = _dri(panel)
    panel["price"] = pd.to_numeric(panel["price"], errors="coerce")
    return panel


def promote(panel: pd.DataFrame, stamp: str | None = None) -> Path:
    from htfe.config import OUTPUTS

    root = OUTPUTS / "panel"
    root.mkdir(parents=True, exist_ok=True)
    stamp = stamp or date.today().strftime("%Y%m%d")
    version = root / f"v{stamp}"
    version.mkdir(parents=True, exist_ok=True)
    destination = version / "analysis_ready_panel.parquet"
    panel.to_parquet(destination, index=False)
    current = root / "analysis_ready_panel.parquet"
    shutil.copy2(destination, current)
    for name in ("name_maps.yaml", "origin_map.yaml"):
        shutil.copy2(MAP_DIR / name, root / name)
        shutil.copy2(MAP_DIR / name, version / name)
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the analysis-ready weekly panel")
    parser.add_argument("--stamp", default=None)
    args = parser.parse_args()
    panel = build()
    checked = validate(panel)
    path = promote(checked, args.stamp)
    print(
        f"wrote {path} rows={len(checked)} "
        f"observed={int(checked['is_observed'].sum())} "
        f"series={checked.groupby(['crop', 'market']).ngroups}"
    )


if __name__ == "__main__":
    main()
