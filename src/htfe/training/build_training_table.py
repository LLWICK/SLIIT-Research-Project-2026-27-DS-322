"""One row per crop, market, forecast date, and horizon.

A forecast dated on a Monday uses only prices and weather from earlier weeks.
The future wholesale price is the label, not an input.

Usage (from member2_ForecastingEngine):
  python -m training.build_training_table
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from htfe.training.config import HORIZONS, PANEL_PATH, assign_split
from htfe.data.build.cultivation_features import (
    SOURCE_LAGGED_SEASONAL_EXTENT,
    SOURCE_NONE,
    SOURCE_OBSERVED_MONTHLY,
    ObservedMonthlyProvider,
    SyntheticCalendarProvider,
    render_modes_audit,
    validate_cultivation_outputs,
)
from htfe.data.build.cultivation_progress import assert_formula, attach_progress, find_monthly_table, load_monthly
from htfe.data.build.join_weather import ORIGIN_WEIGHT_METHOD
from htfe.data.build.macros import MACRO_COLUMNS, asof_macros, write_macro_note
from htfe.data.build.supply_lags import assert_lagged_extent_examples, attach_finished_extent_features
from htfe.config import OUTPUTS, origin_map_path, processed_dir
from htfe.features.frame import week_within_season

MAP_PATH = origin_map_path()
MAPPING_VERSION = "origin_map_yaml_v1"
# Monday forecast sees the previous complete week. The current week's average is not known yet.
PRICE_LAGS = (1, 2, 4, 8, 52)


def _market_id(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _price_block(group: pd.DataFrame) -> pd.DataFrame:
    observed = pd.to_numeric(group["price"], errors="coerce").where(group["is_observed"])
    out = group.copy()
    published_before = observed.shift(1)
    out["last_observed_price_lkr_kg"] = published_before.ffill()
    for lag in PRICE_LAGS:
        out[f"price_lag_{lag}w"] = observed.shift(lag)
    out["price_mean_4w"] = observed.shift(1).rolling(4, min_periods=2).mean()
    out["price_mean_8w"] = observed.shift(1).rolling(8, min_periods=4).mean()
    baseline = out["price_lag_4w"]
    out["price_change_4w_pct"] = (out["price_lag_1w"] - baseline) / baseline.replace(0, pd.NA) * 100
    return out


def _weather_block(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    rain = pd.to_numeric(group.get("weather_precip_mm"), errors="coerce")
    temp = pd.to_numeric(group.get("weather_tmean_c"), errors="coerce")
    # Preceding complete weeks only. Missing rain stays missing; it is not filled with 0.
    out["origin_rainfall_1w_mm"] = rain.shift(1)
    out["origin_rainfall_sum_4w_mm"] = rain.shift(1).rolling(4, min_periods=3).sum()
    out["origin_rainfall_sum_8w_mm"] = rain.shift(1).rolling(8, min_periods=6).sum()
    out["origin_mean_temp_4w_c"] = temp.shift(1).rolling(4, min_periods=3).mean()
    return out


def _rain_anomaly(frame: pd.DataFrame) -> pd.Series:
    """4-week origin rain minus the median of earlier years at the same week. Current year is excluded."""
    work = frame[["crop", "market", "year", "week", "origin_rainfall_sum_4w_mm"]].copy()
    anomaly = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    for year in sorted(work["year"].dropna().unique()):
        prior = work[work["year"] < year].dropna(subset=["origin_rainfall_sum_4w_mm"])
        if prior.empty:
            continue
        norm = prior.groupby(["crop", "market", "week"])["origin_rainfall_sum_4w_mm"].median()
        current = work["year"].eq(year)
        keys = list(zip(work.loc[current, "crop"], work.loc[current, "market"], work.loc[current, "week"]))
        baseline = pd.Series([norm.get(key, pd.NA) for key in keys], index=work.index[current], dtype="Float64")
        counts = prior.groupby(["crop", "market", "week"]).size()
        enough = pd.Series([counts.get(key, 0) >= 3 for key in keys], index=work.index[current])
        anomaly.loc[current] = (work.loc[current, "origin_rainfall_sum_4w_mm"] - baseline).where(enough)
    return anomaly


def _weekly(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["crop", "market", "week_start"]).copy()
    panel["week_start"] = pd.to_datetime(panel["week_start"])
    pieces = []
    for _, group in panel.groupby(["crop", "market"], sort=False):
        block = _weather_block(_price_block(group))
        observed = pd.to_numeric(block["price"], errors="coerce").where(block["is_observed"])
        for horizon in HORIZONS:
            block[f"target_week_h{horizon}"] = block["week_start"].shift(-horizon)
            block[f"target_price_h{horizon}"] = observed.shift(-horizon)
            block[f"target_observed_h{horizon}"] = block["is_observed"].shift(-horizon)
            block[f"target_weeknum_h{horizon}"] = block["week"].shift(-horizon)
        pieces.append(block)
    weekly = pd.concat(pieces, ignore_index=True)
    weekly["origin_rainfall_anomaly_4w_mm"] = _rain_anomaly(weekly)
    return weekly


def _explode(weekly: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for horizon in HORIZONS:
        part = weekly.copy()
        part["horizon_weeks"] = horizon
        part["target_week"] = part[f"target_week_h{horizon}"]
        part["target_price_lkr_kg"] = part[f"target_price_h{horizon}"]
        part["target_is_observed"] = part[f"target_observed_h{horizon}"].fillna(False).astype(bool)
        part["target_week_of_year"] = part[f"target_weeknum_h{horizon}"]
        frames.append(part)
    long = pd.concat(frames, ignore_index=True)
    long = long[long["target_week"].notna()].copy()
    long["target_price_lkr_kg"] = long["target_price_lkr_kg"].where(long["target_is_observed"])
    return long


def _origin_mapping(origin_map: dict) -> pd.DataFrame:
    rows = []
    for crop, markets in origin_map.items():
        for market, spec in markets.items():
            for district in spec.get("supply_districts") or []:
                rows.append(
                    {
                        "crop_id": str(crop).lower(),
                        "market_id": _market_id(market),
                        "origin_district": district,
                        "weight": pd.NA,
                        "supply_weight": pd.NA,
                        "origin_weight_method": ORIGIN_WEIGHT_METHOD,
                        "effective_from": "2016-01-04",
                        "mapping_source": spec.get("note") or "",
                        "assumption": bool(spec.get("assumption", True)),
                        "aggregation_rule": (
                            "No measured trade share. supply_weight is null. "
                            "Weather is the extent-weighted mean of origin_map weather points, using lagged finished-season extent "
                            "and splitting a district across its weather points. If that extent is missing, the join uses equal weights. "
                            "Cultivation hectares, when a monthly file exists, are summed across these districts before dividing."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _attach_economics(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    merged, meta = asof_macros(frame, "forecast_date")
    write_macro_note(meta)
    rename = {
        "diesel_lkr_per_litre": "diesel_price_lkr_litre",
        "usd_lkr": "usd_lkr_rate",
        "inflation_yoy": "inflation_rate_pct",
    }
    for source, target in rename.items():
        merged[target] = merged[source] if source in merged.columns else pd.NA
    merged = merged.drop(columns=[name for name in MACRO_COLUMNS if name in merged.columns])
    if meta["status"] != "joined":
        return merged, "not_found"
    if len(meta["columns"]) < len(MACRO_COLUMNS):
        return merged, "incomplete:" + ",".join(meta["columns"])
    return merged, f"joined:{Path(meta['path']).name}"


def _assert_forecast_identity(table: pd.DataFrame) -> None:
    """target_week is the label week, horizon weeks after the issue Monday."""
    issue = pd.to_datetime(table["forecast_issue_week"])
    target = pd.to_datetime(table["target_week"])
    horizon = pd.to_numeric(table["forecast_horizon_weeks"], errors="coerce")
    if not bool(issue.eq(pd.to_datetime(table["forecast_date"])).all()):
        raise AssertionError("forecast_issue_week is not the forecast Monday")
    if not bool(issue.dt.dayofweek.eq(0).all()) or not bool(target.dt.dayofweek.eq(0).all()):
        raise AssertionError("forecast weeks are not Mondays")
    if not bool(horizon.eq(pd.to_numeric(table["horizon_weeks"])).all()):
        raise AssertionError("forecast_horizon_weeks does not match horizon_weeks")
    if not bool((target - issue).dt.days.eq(horizon * 7).all()):
        raise AssertionError("target_week is not horizon weeks after the issue Monday")


def assert_trailing_rainfall(table: pd.DataFrame, panel: pd.DataFrame) -> None:
    """The one-week rainfall feature is the previous complete week, with gaps left missing."""
    ordered = panel.sort_values(["crop", "market", "week_start"]).copy()
    rain = pd.to_numeric(ordered["weather_precip_mm"], errors="coerce")
    ordered["expected_1w"] = rain.groupby([ordered["crop"], ordered["market"]], sort=False).shift(1)
    ordered["crop_id"] = ordered["crop"].astype(str).str.lower()
    ordered["market_id"] = ordered["market"].map(_market_id)
    ordered["forecast_date"] = pd.to_datetime(ordered["week_start"])
    left = table.drop_duplicates(["crop_id", "market_id", "forecast_date"])[
        ["crop_id", "market_id", "forecast_date", "origin_rainfall_1w_mm"]
    ]
    merged = left.merge(
        ordered[["crop_id", "market_id", "forecast_date", "expected_1w"]],
        on=["crop_id", "market_id", "forecast_date"],
        how="left",
    )
    if len(merged) != len(left):
        raise AssertionError("rainfall check duplicated forecast weeks")
    actual = pd.to_numeric(merged["origin_rainfall_1w_mm"], errors="coerce")
    expected = pd.to_numeric(merged["expected_1w"], errors="coerce")
    both_missing = actual.isna() & expected.isna()
    close = (actual - expected).abs().lt(1e-6)
    if not bool((both_missing | close).all()):
        raise AssertionError("origin_rainfall_1w_mm is not the previous complete week")
    if bool((expected.isna() & actual.notna()).any()):
        raise AssertionError("missing rainfall was filled")


def build(panel: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    panel = panel.copy() if panel is not None else pd.read_parquet(PANEL_PATH)
    origin_map = yaml.safe_load(MAP_PATH.read_text(encoding="utf-8"))
    monthly_path = find_monthly_table()
    monthly = load_monthly(monthly_path) if monthly_path is not None else None
    panel = attach_progress(panel, monthly, origin_map)
    seasonal = pd.read_csv(processed_dir() / "dcs_highland_seasonal.csv")
    panel = attach_finished_extent_features(panel, seasonal, origin_map)
    observed = ObservedMonthlyProvider(monthly, panel).build()
    synthetic = SyntheticCalendarProvider(seasonal, origin_map).build()
    weekly = _weekly(panel)
    table = _explode(weekly)
    table["crop_id"] = table["crop"].astype(str).str.lower()
    table["market_id"] = table["market"].map(_market_id)
    table["forecast_date"] = pd.to_datetime(table["week_start"])
    table["forecast_issue_week"] = table["forecast_date"]
    table["forecast_horizon_weeks"] = table["horizon_weeks"].astype(int)
    table["price_type"] = "wholesale"
    table["price_unit"] = "LKR/kg"
    table["source_name"] = "HARTI"
    table["source_record_id"] = table["crop_id"] + "|" + table["market_id"] + "|" + table["forecast_date"].dt.strftime("%Y-%m-%d")
    table["observation_date"] = table["forecast_date"]
    table["price_available_at"] = table["forecast_date"] - pd.Timedelta(days=1)
    table["forecast_week_of_year"] = table["week"].astype("Int64")
    table["season"] = table["season"]
    table["season_transition"] = table["week"].isin([12, 13, 14, 15, 38, 39, 40, 41])
    table["week_within_season"] = [
        week_within_season(int(week), str(season)) for week, season in zip(table["week"], table["season"])
    ]
    table["supply_origin_mapping_version"] = MAPPING_VERSION
    table["price_missing_flag"] = ~table["is_observed"].astype(bool)
    table["weather_missing_flag"] = table["origin_rainfall_sum_4w_mm"].isna()
    if monthly is None:
        table["origin_target_hectares"] = pd.NA
        table["origin_achieved_hectares"] = pd.NA
        table["cultivation_progress_ratio"] = pd.NA
        table["achieved_hectares_change"] = pd.NA
        table["cultivation_report_age_days"] = pd.NA
        table["cultivation_report_month"] = pd.NA
        table["cultivation_available_at"] = pd.NaT
        table["cultivation_source_name"] = pd.NA
    else:
        table["origin_target_hectares"] = table["cultivation_target_ha"]
        table["origin_achieved_hectares"] = table["cultivation_achieved_ha"]
        table["cultivation_progress_ratio"] = table["cultivation_progress"]
        table["cultivation_available_at"] = pd.to_datetime(table["cultivation_asof_date"])
        table["cultivation_report_age_days"] = (table["forecast_date"] - table["cultivation_available_at"]).dt.days
        table["cultivation_report_month"] = table["cultivation_available_at"].dt.to_period("M").astype(str)
        table["cultivation_source_name"] = table["cultivation_progress_source"]
        table = table.sort_values(["crop_id", "market_id", "forecast_date"])
        previous = table.groupby(["crop_id", "market_id"], sort=False)["origin_achieved_hectares"].shift(1)
        changed = table.groupby(["crop_id", "market_id"], sort=False)["cultivation_available_at"].shift(1)
        table["achieved_hectares_change"] = (table["origin_achieved_hectares"] - previous).where(
            table["cultivation_available_at"].ne(changed)
        )
    table["cultivation_missing_flag"] = table["cultivation_progress_ratio"].isna()
    lagged_extent = (
        table["previous_season_extent"].notna()
        | table["same_season_previous_year_extent"].notna()
        | table["historical_mean_season_extent"].notna()
        | table["extent_change_vs_previous_season"].notna()
    )
    if monthly is None:
        table["cultivation_source"] = np.where(lagged_extent, SOURCE_LAGGED_SEASONAL_EXTENT, SOURCE_NONE)
    else:
        table["cultivation_source"] = np.where(
            table["cultivation_progress_ratio"].notna(),
            SOURCE_OBSERVED_MONTHLY,
            np.where(lagged_extent, SOURCE_LAGGED_SEASONAL_EXTENT, SOURCE_NONE),
        )
    table["cultivation_is_synthetic"] = False
    table["cultivation_scenario"] = pd.Series(pd.NA, index=table.index, dtype="object")
    table, macro_status = _attach_economics(table)
    table["diesel_missing_flag"] = table["diesel_price_lkr_litre"].isna()
    table["usd_missing_flag"] = table["usd_lkr_rate"].isna()
    table["inflation_missing_flag"] = table["inflation_rate_pct"].isna()
    splits = assign_split(table["forecast_issue_week"])
    table["split"] = splits["split"]
    table["is_shock"] = splits["is_shock"]
    _assert_forecast_identity(table)
    columns = [
        "crop_id",
        "market_id",
        "forecast_date",
        "forecast_issue_week",
        "horizon_weeks",
        "forecast_horizon_weeks",
        "target_week",
        "target_price_lkr_kg",
        "target_is_observed",
        "target_week_of_year",
        "last_observed_price_lkr_kg",
        "price_lag_1w",
        "price_lag_2w",
        "price_lag_4w",
        "price_lag_8w",
        "price_lag_52w",
        "price_mean_4w",
        "price_mean_8w",
        "price_change_4w_pct",
        "season",
        "season_transition",
        "forecast_week_of_year",
        "week_within_season",
        "origin_rainfall_1w_mm",
        "origin_rainfall_sum_4w_mm",
        "origin_rainfall_sum_8w_mm",
        "origin_mean_temp_4w_c",
        "origin_rainfall_anomaly_4w_mm",
        "diesel_price_lkr_litre",
        "usd_lkr_rate",
        "inflation_rate_pct",
        "origin_target_hectares",
        "origin_achieved_hectares",
        "cultivation_progress_ratio",
        "cultivation_target_exceeded",
        "previous_season_extent",
        "same_season_previous_year_extent",
        "historical_mean_season_extent",
        "extent_change_vs_previous_season",
        "cultivation_source",
        "cultivation_is_synthetic",
        "cultivation_scenario",
        "achieved_hectares_change",
        "cultivation_report_age_days",
        "cultivation_report_month",
        "cultivation_available_at",
        "observation_date",
        "price_available_at",
        "source_name",
        "source_record_id",
        "cultivation_source_name",
        "price_type",
        "price_unit",
        "supply_origin_mapping_version",
        "price_missing_flag",
        "weather_missing_flag",
        "cultivation_missing_flag",
        "diesel_missing_flag",
        "usd_missing_flag",
        "inflation_missing_flag",
        "split",
        "is_shock",
    ]
    table = table[columns].sort_values(["crop_id", "market_id", "forecast_date", "horizon_weeks"]).reset_index(drop=True)
    assert_trailing_rainfall(table, panel)
    mapping = _origin_mapping(origin_map)
    status = {
        "rows": int(len(table)),
        "horizons": list(HORIZONS),
        "cultivation": "joined" if monthly is not None else "missing_no_district_monthly_file",
        "observed_monthly_file": None if monthly_path is None else str(monthly_path),
        "observed_rows": int(len(observed)),
        "synthetic_rows": int(len(synthetic)),
        "origin_weight_method": ORIGIN_WEIGHT_METHOD,
        "macros": macro_status,
        "target_observed_share": round(float(table["target_is_observed"].mean()), 3),
        "weather_missing_share": round(float(table["weather_missing_flag"].mean()), 3),
    }
    return table, mapping, status, observed, synthetic


def _write_dictionary(table: pd.DataFrame, status: dict, path: Path) -> None:
    labels = {
        "crop_id": "A identity. Not a future value.",
        "market_id": "A identity. Wholesale market, not the growing district.",
        "forecast_date": "Monday on which inputs are frozen. Same value as forecast_issue_week.",
        "forecast_issue_week": "Monday the forecast is issued. Every time-varying input is known on or before this Monday.",
        "horizon_weeks": "How many weeks ahead the label sits. Same value as forecast_horizon_weeks.",
        "forecast_horizon_weeks": "Weeks from forecast_issue_week to target_week. The label is the price at target_week.",
        "target_week": "Future Monday. Known as a calendar date. The label is the price of this week, not an input.",
        "target_price_lkr_kg": "Label only. Blank when that future week was not published.",
        "last_observed_price_lkr_kg": "Latest published price strictly before forecast_date.",
        "price_lag_1w": "Published price one week before forecast_date. Blank if that week was not published.",
        "price_change_4w_pct": "Change from price_lag_4w to price_lag_1w. Both must be published.",
        "origin_rainfall_1w_mm": "Origin-district rain in the one complete week before forecast_date. Missing rain is left missing.",
        "origin_rainfall_sum_4w_mm": "Origin-district rain over the four complete weeks before forecast_date. The sum starts after shift(1).",
        "origin_rainfall_anomaly_4w_mm": "That four-week sum minus the median of earlier years at the same week. Blank until three earlier years exist.",
        "diesel_price_lkr_litre": "Empty until a dated diesel series is on disk. Not invented.",
        "usd_lkr_rate": "Empty until a dated exchange-rate series is on disk.",
        "inflation_rate_pct": "Empty until a dated published inflation series is on disk.",
        "origin_target_hectares": "Empty until district-month target hectares exist. Not copied from seasonal extent.",
        "origin_achieved_hectares": "Empty until district-month achieved hectares exist. Not copied from seasonal extent.",
        "cultivation_progress_ratio": "Sum of achieved divided by sum of target, not clipped at 1. Empty when the monthly file is absent. Not the synthetic logistic.",
        "cultivation_target_exceeded": "True only when the raw achieved/target ratio is above 1. Empty when progress is empty.",
        "previous_season_extent": "Supply-district extent of the last Yala or Maha that has already finished. Not the current season.",
        "same_season_previous_year_extent": "Extent of the same season one year earlier, included only after that season has finished.",
        "historical_mean_season_extent": "Mean of finished Yala and Maha supply extents before forecast_date. The unfinished season is excluded.",
        "extent_change_vs_previous_season": "previous_season_extent minus the finished season immediately before it, in hectares.",
        "cultivation_source": "observed_monthly, lagged_seasonal_extent, or none. synthetic_calendar is not used on this table.",
        "cultivation_is_synthetic": "False on this table. Synthetic rows are only in cultivation_synthetic.parquet.",
        "cultivation_scenario": "Null on this table. early, typical, and late exist only on the synthetic file.",
        "cultivation_report_age_days": "forecast_date minus the report availability date.",
        "price_available_at": "Day before forecast_date. The current week's average is not treated as known on Monday.",
        "target_is_observed": "Score only rows where this is true.",
    }
    lines = [
        "# Training table columns",
        "",
        "Each row is one forecast: one crop, one wholesale market, one Monday, one horizon.",
        f"Rows: {status['rows']}. Horizons: {status['horizons']}.",
        f"Cultivation: {status['cultivation']}. Economics: {status['macros']}.",
        f"Share of rows with an observed future price: {status['target_observed_share']}.",
        f"Share of rows missing the 4-week origin rainfall sum: {status['weather_missing_share']}.",
        "",
        "The future price is `target_price_lkr_kg`. It is not an input.",
        "Seasonal Census extent is not copied into `cultivation_progress_ratio`.",
        "Lagged extent columns use only seasons that have already finished.",
        "Synthetic calendar progress is not on this table. It is in `cultivation_synthetic.parquet` and is not evidence about observed cultivation progress.",
        "",
        "| Column | Non-null share | Role |",
        "| --- | ---: | --- |",
    ]
    for column in table.columns:
        share = round(float(table[column].notna().mean()), 3)
        lines.append(f"| `{column}` | {share} | {labels.get(column, '')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    assert_formula()
    assert_lagged_extent_examples()
    table, mapping, status, observed, synthetic = build()
    seasonal = pd.read_csv(processed_dir() / "dcs_highland_seasonal.csv")
    origin_map = yaml.safe_load(MAP_PATH.read_text(encoding="utf-8"))
    summary = validate_cultivation_outputs(
        table,
        synthetic,
        observed,
        mapping,
        seasonal,
        origin_map,
        monthly_found=status["observed_monthly_file"] is not None,
    )
    out = OUTPUTS / "panel"
    out.mkdir(parents=True, exist_ok=True)
    table_path = out / "training_table.parquet"
    map_path = out / "supply_origin_mapping.csv"
    observed_path = out / "cultivation_observed.parquet"
    synthetic_path = out / "cultivation_synthetic.parquet"
    table.to_parquet(table_path, index=False)
    mapping.to_csv(map_path, index=False)
    observed.to_parquet(observed_path, index=False)
    synthetic.to_parquet(synthetic_path, index=False)
    notes = OUTPUTS / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    docs = notes / "Training-Table-Columns-IT23415836.md"
    audit = notes / "Cultivation-Modes-IT23415836.md"
    _write_dictionary(table, status, docs)
    audit.write_text(render_modes_audit(summary), encoding="utf-8")
    print(f"wrote {table_path} rows={status['rows']} cultivation={status['cultivation']} macros={status['macros']}")
    print(f"wrote {map_path} rows={len(mapping)} origin_weight_method={status['origin_weight_method']}")
    print(f"wrote {observed_path} rows={status['observed_rows']} monthly_file={status['observed_monthly_file']}")
    print(f"wrote {synthetic_path} rows={status['synthetic_rows']}")
    print(f"wrote {docs}")
    print(f"wrote {audit}")
    print(
        "coverage "
        f"progress_nonnull={summary['progress_nonnull']} "
        f"previous_season_extent={summary['previous_season_extent_nonnull_share']} "
        f"same_season_previous_year={summary['same_season_previous_year_extent_nonnull_share']} "
        f"historical_mean={summary['historical_mean_season_extent_nonnull_share']} "
        f"extent_change={summary['extent_change_vs_previous_season_nonnull_share']} "
        f"rainfall_1w={summary['origin_rainfall_1w_mm_nonnull_share']} "
        f"empty_lagged_series={summary['series_with_no_lagged_extent']}"
    )


if __name__ == "__main__":
    main()
