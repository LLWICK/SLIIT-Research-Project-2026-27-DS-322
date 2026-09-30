"""One row per crop, market, forecast date, and horizon.

A forecast dated on a Monday uses only prices and weather from earlier weeks.
The future wholesale price is the label, not an input.

Usage (from member2_ForecastingEngine):
  python -m training.build_training_table
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from training.config import HORIZONS, PANEL_PATH, assign_split
from data_pipeline.build.cultivation_progress import attach_progress, find_monthly_table, load_monthly
from data_pipeline.build.macros import MACRO_COLUMNS, find_macro_table
from data_pipeline.config import processed_dir
from training.build_feature_frame import week_within_season

MAP_PATH = Path(__file__).resolve().parents[2] / "data_pipeline" / "maps" / "origin_map.yaml"
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
                        "effective_from": "2016-01-04",
                        "mapping_source": spec.get("note") or "",
                        "assumption": bool(spec.get("assumption", True)),
                        "aggregation_rule": (
                            "No measured trade share. Weather on the panel is the extent-weighted mean of mapped origin points. "
                            "Cultivation hectares, when a monthly file exists, are summed across these districts before dividing."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _attach_economics(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    found = find_macro_table()
    frame["diesel_price_lkr_litre"] = pd.NA
    frame["usd_lkr_rate"] = pd.NA
    frame["inflation_rate_pct"] = pd.NA
    if found is None:
        return frame, "not_found"
    series = pd.read_csv(found) if found.suffix.lower() == ".csv" else pd.read_excel(found)
    series.columns = [str(column).strip().lower().replace(" ", "_") for column in series.columns]
    if "available_date" not in series.columns:
        return frame, "rejected_no_available_date"
    series["available_date"] = pd.to_datetime(series["available_date"])
    keep = ["available_date", *[name for name in MACRO_COLUMNS if name in series.columns]]
    series = series[keep].sort_values("available_date")
    left = frame.sort_values("forecast_date")
    merged = pd.merge_asof(left, series, left_on="forecast_date", right_on="available_date", direction="backward")
    rename = {
        "diesel_lkr_per_litre": "diesel_price_lkr_litre",
        "usd_lkr": "usd_lkr_rate",
        "inflation_yoy": "inflation_rate_pct",
    }
    for source, target in rename.items():
        if source in merged.columns:
            merged[target] = merged[source]
    return merged.drop(columns=["available_date"], errors="ignore"), f"joined:{found.name}"


def build(panel: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    panel = panel.copy() if panel is not None else pd.read_parquet(PANEL_PATH)
    origin_map = yaml.safe_load(MAP_PATH.read_text(encoding="utf-8"))
    monthly_path = find_monthly_table()
    monthly = load_monthly(monthly_path) if monthly_path is not None else None
    panel = attach_progress(panel, monthly, origin_map)
    weekly = _weekly(panel)
    table = _explode(weekly)
    table["crop_id"] = table["crop"].astype(str).str.lower()
    table["market_id"] = table["market"].map(_market_id)
    table["forecast_date"] = pd.to_datetime(table["week_start"])
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
    table, macro_status = _attach_economics(table)
    table["diesel_missing_flag"] = table["diesel_price_lkr_litre"].isna()
    table["usd_missing_flag"] = table["usd_lkr_rate"].isna()
    table["inflation_missing_flag"] = table["inflation_rate_pct"].isna()
    splits = assign_split(table["forecast_date"])
    table["split"] = splits["split"]
    table["is_shock"] = splits["is_shock"]
    columns = [
        "crop_id",
        "market_id",
        "forecast_date",
        "horizon_weeks",
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
    mapping = _origin_mapping(origin_map)
    status = {
        "rows": int(len(table)),
        "horizons": list(HORIZONS),
        "cultivation": "joined" if monthly is not None else "missing_no_district_monthly_file",
        "macros": macro_status,
        "target_observed_share": round(float(table["target_is_observed"].mean()), 3),
        "weather_missing_share": round(float(table["weather_missing_flag"].mean()), 3),
    }
    return table, mapping, status


def _write_dictionary(table: pd.DataFrame, status: dict, path: Path) -> None:
    labels = {
        "crop_id": "A identity. Not a future value.",
        "market_id": "A identity. Wholesale market, not the growing district.",
        "forecast_date": "Monday on which inputs are frozen.",
        "horizon_weeks": "How many weeks ahead the label sits. Includes 4, 8, and 12.",
        "target_week": "Future week. Known as a calendar date. Not a price.",
        "target_price_lkr_kg": "Label only. Blank when that future week was not published.",
        "last_observed_price_lkr_kg": "Latest published price strictly before forecast_date.",
        "price_lag_1w": "Published price one week before forecast_date. Blank if that week was not published.",
        "price_change_4w_pct": "Change from price_lag_4w to price_lag_1w. Both must be published.",
        "origin_rainfall_sum_4w_mm": "Origin-district rain over the four complete weeks before forecast_date.",
        "origin_rainfall_anomaly_4w_mm": "That four-week sum minus the median of earlier years at the same week. Blank until three earlier years exist.",
        "diesel_price_lkr_litre": "Empty until a dated diesel series is on disk. Not invented.",
        "usd_lkr_rate": "Empty until a dated exchange-rate series is on disk.",
        "inflation_rate_pct": "Empty until a dated published inflation series is on disk.",
        "origin_target_hectares": "Empty until district-month target hectares exist.",
        "origin_achieved_hectares": "Empty until district-month achieved hectares exist.",
        "cultivation_progress_ratio": "Sum of achieved divided by sum of target. Empty when either is missing.",
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
        "Seasonal Census extent is not copied into the cultivation columns.",
        "",
        "| Column | Non-null share | Role |",
        "| --- | ---: | --- |",
    ]
    for column in table.columns:
        share = round(float(table[column].notna().mean()), 3)
        lines.append(f"| `{column}` | {share} | {labels.get(column, '')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    table, mapping, status = build()
    out = processed_dir() / "latest"
    out.mkdir(parents=True, exist_ok=True)
    table_path = out / "training_table.parquet"
    map_path = out / "supply_origin_mapping.csv"
    table.to_parquet(table_path, index=False)
    mapping.to_csv(map_path, index=False)
    docs = Path(__file__).resolve().parents[3] / "docs" / "Training-Table-Columns-IT23415836.md"
    _write_dictionary(table, status, docs)
    print(f"wrote {table_path} rows={status['rows']} cultivation={status['cultivation']} macros={status['macros']}")
    print(f"wrote {map_path} rows={len(mapping)}")
    print(f"wrote {docs}")


if __name__ == "__main__":
    main()
