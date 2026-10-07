"""Coverage of the primary training table, written before Experiment A is fit."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from htfe.data.build.join_weather import ORIGIN_WEIGHT_METHOD
from htfe.config import OUTPUTS, built_file, processed_dir
from htfe.training.build_training_table import MAP_PATH, assert_trailing_rainfall
from htfe.training.config import PANEL_PATH

DOCS = OUTPUTS / "notes" / "Feature-Coverage-Audit-IT23415836.md"

COVERAGE_COLUMNS = [
    ("price_lag", "price_lag_1w"),
    ("price_lag", "price_lag_2w"),
    ("price_lag", "price_lag_4w"),
    ("price_lag", "price_lag_8w"),
    ("price_lag", "price_lag_52w"),
    ("price_lag", "price_mean_4w"),
    ("price_lag", "price_mean_8w"),
    ("price_lag", "price_change_4w_pct"),
    ("season", "season"),
    ("season", "season_transition"),
    ("season", "forecast_week_of_year"),
    ("season", "week_within_season"),
    ("origin_rainfall_1w", "origin_rainfall_1w_mm"),
    ("origin_rainfall_4w", "origin_rainfall_sum_4w_mm"),
    ("origin_temp", "origin_mean_temp_4w_c"),
    ("diesel", "diesel_price_lkr_litre"),
    ("usd_lkr", "usd_lkr_rate"),
    ("inflation", "inflation_rate_pct"),
    ("lagged_extent", "previous_season_extent"),
    ("lagged_extent", "same_season_previous_year_extent"),
    ("lagged_extent", "historical_mean_season_extent"),
    ("lagged_extent", "extent_change_vs_previous_season"),
    ("cultivation_progress_ratio", "cultivation_progress_ratio"),
]


def _share(series: pd.Series) -> float:
    return round(float(series.notna().mean()), 4) if len(series) else 0.0


def write_feature_coverage(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(table)
    for group, column in COVERAGE_COLUMNS:
        if column not in table.columns:
            share = 0.0
            non_null = 0
        else:
            non_null = int(table[column].notna().sum())
            share = round(non_null / n, 4) if n else 0.0
        rows.append(
            {
                "group": group,
                "column": column,
                "non_null_share": share,
                "non_null_rows": non_null,
                "rows": n,
            }
        )
    return pd.DataFrame(rows)


def write_weather_audit(table: pd.DataFrame, origin_map: dict, mapping: pd.DataFrame) -> pd.DataFrame:
    if bool(mapping["supply_weight"].notna().any()):
        raise AssertionError("supply_weight was filled")
    if not bool(mapping["origin_weight_method"].eq(ORIGIN_WEIGHT_METHOD).all()):
        raise AssertionError(mapping["origin_weight_method"].unique())
    if ORIGIN_WEIGHT_METHOD != "extent_weighted_weather":
        raise AssertionError(ORIGIN_WEIGHT_METHOD)
    names = {market.lower().replace(" ", "_"): market for markets in origin_map.values() for market in markets}
    forecasts = table.drop_duplicates(["crop_id", "market_id", "forecast_issue_week"])
    rows = []
    for crop, markets in origin_map.items():
        for market, spec in markets.items():
            market_id = market.lower().replace(" ", "_")
            part = forecasts[forecasts["crop_id"].eq(str(crop).lower()) & forecasts["market_id"].eq(market_id)]
            districts = spec.get("supply_districts") or []
            points = spec.get("weather") or []
            rows.append(
                {
                    "crop": crop,
                    "market": names.get(market_id, market),
                    "origin_districts": "; ".join(str(district) for district in districts),
                    "weather_points": "; ".join(str(point) for point in points),
                    "origin_weight_method": ORIGIN_WEIGHT_METHOD,
                    "supply_weight": pd.NA,
                    "rainfall_1w_non_null_share": _share(part["origin_rainfall_1w_mm"]) if len(part) else 0.0,
                    "rainfall_4w_non_null_share": _share(part["origin_rainfall_sum_4w_mm"]) if len(part) else 0.0,
                    "origin_temp_4w_non_null_share": _share(part["origin_mean_temp_4w_c"]) if len(part) else 0.0,
                    "previous_season_extent_non_null_share": _share(part["previous_season_extent"]) if len(part) else 0.0,
                    "equal_weight_fallback": (
                        "Equal weights are used only when lagged finished-season extent is missing. "
                        "Weeks with null previous_season_extent have share "
                        f"{round(1 - _share(part['previous_season_extent']), 4) if len(part) else 1}. "
                        "Measured trade shares such as 0.60/0.30/0.10 are not used."
                    ),
                }
            )
    return pd.DataFrame(rows)


def render_coverage_markdown(coverage: pd.DataFrame, weather: pd.DataFrame, macro_note: str, cultivation_note: str) -> str:
    lines = [
        "# Feature coverage audit",
        "",
        "Primary table: `processed/latest/training_table.parquet`.",
        "Shares are non-null fractions of forecast rows (crop, market, issue week, horizon).",
        "Weather coverage in the origin audit is computed once per issue week, not once per horizon.",
        "",
        macro_note,
        "",
        cultivation_note,
        "",
        "`origin_rainfall_1w_mm` was checked against the panel: it is `weather_precip_mm` shifted by one week inside each crop-market. It is the previous complete week, not the target week. Missing rain stays missing.",
        f"`origin_weight_method` is `{ORIGIN_WEIGHT_METHOD}` on every supply-origin row. `supply_weight` is null.",
        "",
        "| Group | Column | Non-null share |",
        "| --- | --- | ---: |",
    ]
    for _, row in coverage.iterrows():
        lines.append(f"| {row['group']} | `{row['column']}` | {row['non_null_share']} |")
    lines.extend(
        [
            "",
            "## Origin weather",
            "",
            "| Crop | Market | Origin districts | Weight method | Rain 1w | Rain 4w | Temp 4w |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in weather.iterrows():
        lines.append(
            f"| {row['crop']} | {row['market']} | {row['origin_districts']} | {row['origin_weight_method']} | "
            f"{row['rainfall_1w_non_null_share']} | {row['rainfall_4w_non_null_share']} | {row['origin_temp_4w_non_null_share']} |"
        )
    lines.extend(
        [
            "",
            "Equal weight is only the fallback when lagged finished-season extent is missing. It is not a measured 0.60/0.30/0.10 trade share.",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit(table: pd.DataFrame, panel: pd.DataFrame) -> dict:
    assert_trailing_rainfall(table, panel)
    origin_map = yaml.safe_load(MAP_PATH.read_text(encoding="utf-8"))
    mapping = pd.read_csv(built_file("supply_origin_mapping.csv"))
    coverage = write_feature_coverage(table)
    weather = write_weather_audit(table, origin_map, mapping)
    latest = OUTPUTS / "panel"
    DOCS.parent.mkdir(parents=True, exist_ok=True)
    latest.mkdir(parents=True, exist_ok=True)
    coverage_path = latest / "feature_coverage.csv"
    weather_path = latest / "weather_origin_audit.csv"
    coverage.to_csv(coverage_path, index=False)
    weather.to_csv(weather_path, index=False)
    macro_share = coverage[coverage["group"].isin(["diesel", "usd_lkr", "inflation"])]
    if bool((macro_share["non_null_share"] > 0).all()):
        macro_note = (
            "Diesel, USD/LKR, and inflation are joined as-of from `processed/macros_published.csv`. "
            "Annual World Bank series were not used."
        )
    else:
        macro_note = (
            "Experiment B is unavailable: diesel, USD/LKR, and inflation are not all non-null "
            "from a real as-of series."
        )
    progress = coverage.loc[coverage["column"].eq("cultivation_progress_ratio"), "non_null_share"]
    if progress.empty or float(progress.iloc[0]) == 0:
        cultivation_note = "Observed cultivation data unavailable. `cultivation_progress_ratio` is null. Experiment C is not trained."
    else:
        cultivation_note = "Observed cultivation progress is non-null on part of the table and was joined as-of without clipping at 1."
    DOCS.write_text(render_coverage_markdown(coverage, weather, macro_note, cultivation_note), encoding="utf-8")
    return {
        "coverage_csv": str(coverage_path),
        "weather_csv": str(weather_path),
        "coverage_md": str(DOCS),
        "macro_note": macro_note,
        "cultivation_note": cultivation_note,
    }
