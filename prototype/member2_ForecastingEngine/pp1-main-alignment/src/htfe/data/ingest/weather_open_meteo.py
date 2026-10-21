"""Fetch daily ERA5 / ERA5-Land reanalysis from the Open-Meteo Historical Weather
API and build tidy daily, weekly and monthly weather tables.

Run from the ``prototype`` folder::

    python -m htfe.data.ingest.weather_open_meteo                # fetch missing + build
    python -m htfe.data.ingest.weather_open_meteo --build-only   # rebuild CSVs from raw
    python -m htfe.data.ingest.weather_open_meteo --refresh      # re-fetch stale files

API: https://archive-api.open-meteo.com/v1/archive
Docs: https://open-meteo.com/en/docs/historical-weather-api
Licence: data CC BY 4.0 (attribute "Weather data by Open-Meteo.com"); the free
API is for non-commercial use only, limited to 600/min, 5,000/hour and
10,000/day "weighted" calls (all three limits are weighted). A request counts as
(days / 14) x (variables / 10) calls when either factor exceeds 1, so one
location for 2010-2026 with 12 variables costs about 525 calls. This script
tracks the weight it spends in ``raw/open_meteo/_fetch_log.csv`` and waits or
stops before hitting the hourly/daily limits, so it is safe to rerun.

Outputs
-------
raw/open_meteo/<location>__<model>.json        verbatim API response
raw/open_meteo/<location>__<model>.meta.json   request URL, params, status, time
processed/weather_daily_open_meteo.csv
processed/weather_weekly_open_meteo.csv       HARTI weeks (Mon on/after 1 Jan = W1)
processed/weather_weekly_iso_open_meteo.csv   ISO-8601 weeks
processed/weather_monthly_open_meteo.csv
processed/weather_locations.csv
processed/audit/weather_open_meteo_completeness.csv
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from htfe.config import AUDIT_DIR, processed_dir, raw_dir
from htfe.data.ingest.weather_common import (
    Location,
    RateLimited,
    aggregate_periods,
    append_fetch_log,
    completeness_profile,
    locations_frame,
    make_session,
    polite_sleep,
    read_fetch_log,
    request_with_retries,
    select_locations,
)

SOURCE = "open_meteo"
API_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "Asia/Colombo"

# API daily variable -> tidy column name (unit in the name).
DAILY_VARIABLES: dict[str, str] = {
    "temperature_2m_max": "tmax_c",
    "temperature_2m_min": "tmin_c",
    "temperature_2m_mean": "tmean_c",
    "precipitation_sum": "precip_mm",
    "rain_sum": "rain_mm",
    "precipitation_hours": "precip_hours_h",
    "et0_fao_evapotranspiration": "et0_mm",
    "relative_humidity_2m_mean": "rh_mean_pct",
    "wind_speed_10m_max": "wind_max_kmh",
    "shortwave_radiation_sum": "shortwave_mj_m2",
    "soil_moisture_0_to_7cm_mean": "soil_moist_0_7cm_m3m3",
    "soil_moisture_7_to_28cm_mean": "soil_moist_7_28cm_m3m3",
}

AGG_RULES: dict[str, str] = {
    "precip_mm": "sum",
    "rain_mm": "sum",
    "precip_hours_h": "sum",
    "et0_mm": "sum",
    "shortwave_mj_m2": "sum",
    "tmax_c": "mean",
    "tmin_c": "mean",
    "tmean_c": "mean",
    "rh_mean_pct": "mean",
    "wind_max_kmh": "mean",
    "soil_moist_0_7cm_m3m3": "mean",
    "soil_moist_7_28cm_m3m3": "mean",
}
AGG_EXTRA = {
    "precip_max_1d_mm": ("precip_mm", "max"),
    "tmax_max_c": ("tmax_c", "max"),
    "tmin_min_c": ("tmin_c", "min"),
    "wet_days_1mm": ("wet_day", "sum"),
    "heavy_rain_days_50mm": ("heavy_day", "sum"),
}

MINUTELY_LIMIT = 600
HOURLY_LIMIT = 5000
DAILY_LIMIT = 10000


def call_weight(n_days: int, n_vars: int, n_locations: int = 1) -> float:
    """Open-Meteo fractional call weight (see https://open-meteo.com/en/pricing)."""
    return max(1.0, n_days / 14) * max(1.0, n_vars / 10) * n_locations


def raw_paths(loc: Location, model: str):
    folder = raw_dir(SOURCE)
    stem = f"{loc.key}__{model}"
    return folder / f"{stem}.json", folder / f"{stem}.meta.json"


def needs_fetch(loc: Location, model: str, start: date, end: date, force: bool, refresh: bool) -> bool:
    data_path, meta_path = raw_paths(loc, model)
    if force or not data_path.exists() or not meta_path.exists():
        return True
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != 200 or meta["params"]["start_date"] > start.isoformat():
        return True
    return refresh and meta["params"]["end_date"] < end.isoformat()


def wait_for_budget(weight: float, hourly_budget: float, daily_budget: float) -> bool:
    """Sleep until ``weight`` fits in the rolling hourly budget.

    Returns False when the rolling 24 h budget is exhausted (caller stops).
    """
    log_path = raw_dir(SOURCE) / "_fetch_log.csv"
    while True:
        log = read_fetch_log(log_path)
        if log.empty:
            return weight <= daily_budget
        ok = log[(log["source"] == SOURCE) & (log["status"] == 200)]
        now = pd.Timestamp.now(tz="UTC")
        last_hour = ok[ok["fetched_at_utc"] > now - pd.Timedelta(hours=1)]
        last_day = ok[ok["fetched_at_utc"] > now - pd.Timedelta(days=1)]
        used_h = float(last_hour["weight"].sum()) if len(last_hour) else 0.0
        used_d = float(last_day["weight"].sum()) if len(last_day) else 0.0
        if used_d + weight > daily_budget:
            print(f"  daily budget reached ({used_d:.0f} + {weight:.0f} > {daily_budget:.0f}); rerun later")
            return False
        if used_h + weight <= hourly_budget:
            return True
        oldest = last_hour["fetched_at_utc"].min()
        wait = max(60.0, (oldest + pd.Timedelta(hours=1) - now).total_seconds() + 5)
        print(f"  hourly budget {used_h:.0f}/{hourly_budget:.0f} used; sleeping {wait / 60:.1f} min")
        time.sleep(wait)


def fetch_location(session, loc: Location, model: str, start: date, end: date, weight: float) -> bool:
    data_path, meta_path = raw_paths(loc, model)
    log_path = raw_dir(SOURCE) / "_fetch_log.csv"
    params = {
        "latitude": loc.lat,
        "longitude": loc.lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": TIMEZONE,
        "models": model,
    }
    for attempt in range(4):
        try:
            resp = request_with_retries(session, API_URL, params, timeout=300)
        except RateLimited as exc:
            reason = exc.response.text[:300]
            append_fetch_log(log_path, source=SOURCE, location=loc.key, status=429, weight=0,
                             url=exc.response.url, message=reason)
            if "Daily" in reason or "Monthly" in reason:
                print(f"  {loc.key}: 429 daily/monthly limit -> stop. {reason}")
                raise
            wait = 60 if "Minutely" in reason else 15 * 60
            print(f"  {loc.key}: 429 ({reason}); waiting {wait // 60} min (attempt {attempt + 1})")
            time.sleep(wait)
            continue
        break
    else:
        return False

    body = resp.content
    status = resp.status_code
    message = ""
    if status == 200:
        data_path.write_bytes(body)
    else:
        message = resp.text[:300]
    meta = {
        "source": SOURCE,
        "location": loc.key,
        "name": loc.name,
        "url": resp.url,
        "params": params,
        "status": status,
        "bytes": len(body),
        "weight_estimate": round(weight, 1),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "message": message,
    }
    if status == 200 or not data_path.exists():
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    append_fetch_log(log_path, source=SOURCE, location=loc.key, status=status, bytes=len(body),
                     weight=round(weight, 1) if status == 200 else 0, url=resp.url, message=message)
    print(f"  {loc.key}: HTTP {status}, {len(body) / 1024:.0f} KB, weight ~{weight:.0f} {message}")
    return status == 200


def parse_raw(loc: Location, model: str) -> tuple[pd.DataFrame, dict]:
    data_path, _ = raw_paths(loc, model)
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    daily = pd.DataFrame(payload["daily"]).rename(columns={"time": "date", **DAILY_VARIABLES})
    # precipitation_hours is returned as 0 (not null) on days with no data yet.
    daily.loc[daily["precip_mm"].isna(), "precip_hours_h"] = float("nan")
    value_cols = list(DAILY_VARIABLES.values())
    has_value = daily[value_cols].notna().any(axis=1)
    if has_value.any():
        daily = daily.loc[: has_value[has_value].index.max()]
    daily.insert(1, "location", loc.key)
    daily.insert(2, "lat", loc.lat)
    daily.insert(3, "lon", loc.lon)
    grid = {
        "location": loc.key,
        "om_grid_lat": payload.get("latitude"),
        "om_grid_lon": payload.get("longitude"),
        "om_grid_elevation_m": payload.get("elevation"),
        "om_model": model,
    }
    return daily, grid


def build_outputs(locations: list[Location], model: str) -> None:
    out_dir = processed_dir()
    frames, grids = [], []
    for loc in locations:
        data_path, _ = raw_paths(loc, model)
        if not data_path.exists():
            print(f"  (no raw file for {loc.key}; skipped in build)")
            continue
        daily, grid = parse_raw(loc, model)
        frames.append(daily)
        grids.append(grid)
    if not frames:
        print("Nothing to build.")
        return
    daily = pd.concat(frames, ignore_index=True)
    daily.to_csv(out_dir / "weather_daily_open_meteo.csv", index=False)

    agg_input = daily.assign(
        wet_day=(daily["precip_mm"] >= 1.0).where(daily["precip_mm"].notna()).astype(float),
        heavy_day=(daily["precip_mm"] >= 50.0).where(daily["precip_mm"].notna()).astype(float),
    )
    weekly = aggregate_periods(agg_input, AGG_RULES, "harti_week", AGG_EXTRA)
    weekly_iso = aggregate_periods(agg_input, AGG_RULES, "iso_week", AGG_EXTRA)
    monthly = aggregate_periods(agg_input, AGG_RULES, "month", AGG_EXTRA)
    for df, name in (
        (weekly, "weather_weekly_open_meteo.csv"),
        (weekly_iso, "weather_weekly_iso_open_meteo.csv"),
        (monthly, "weather_monthly_open_meteo.csv"),
    ):
        df.round(3).to_csv(out_dir / name, index=False)

    loc_path = out_dir / "weather_locations.csv"
    locs = locations_frame()
    if loc_path.exists():
        prev = pd.read_csv(loc_path)
        keep = [c for c in prev.columns if c.startswith("nasa_")]
        if keep:
            locs = locs.merge(prev[["location", *keep]], on="location", how="left")
    locs = locs.merge(pd.DataFrame(grids), on="location", how="left")
    locs.to_csv(loc_path, index=False)

    today = pd.Timestamp(date.today())
    profile = completeness_profile(daily, list(DAILY_VARIABLES.values()), today)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    profile.to_csv(AUDIT_DIR / "weather_open_meteo_completeness.csv", index=False)

    print(f"daily rows {len(daily):,} | locations {daily['location'].nunique()} | "
          f"{daily['date'].min()} -> {daily['date'].max()}")
    print(f"weekly(harti) {len(weekly):,} | weekly(iso) {len(weekly_iso):,} | monthly {len(monthly):,}")
    cols = ["location", "first_date", "last_date", "missing_dates", "na_precip_mm", "latency_days_precip_mm"]
    print(profile[cols].to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2010-01-01", help="first date (YYYY-MM-DD), default 2010-01-01")
    parser.add_argument("--end", default=None, help="last date requested, default yesterday")
    parser.add_argument("--locations", default=None, help="comma-separated location keys (default core set)")
    parser.add_argument("--include-extra", action="store_true", help="also fetch optional market centroids")
    parser.add_argument("--model", default="era5_seamless",
                        help="Open-Meteo reanalysis model: era5_seamless (default; ERA5-Land temperature/"
                             "humidity/soil + ERA5 rain/wind/solar), era5, era5_land, best_match")
    parser.add_argument("--force", action="store_true", help="re-fetch even if raw files exist")
    parser.add_argument("--refresh", action="store_true", help="re-fetch raw files that end before --end")
    parser.add_argument("--build-only", action="store_true", help="skip network; rebuild CSVs from raw files")
    parser.add_argument("--hourly-budget", type=float, default=4500, help=f"weighted calls/hour (limit {HOURLY_LIMIT})")
    parser.add_argument("--daily-budget", type=float, default=9500, help=f"weighted calls/day (limit {DAILY_LIMIT})")
    args = parser.parse_args(argv)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    locations = select_locations(args.locations, args.include_extra)
    n_days = (end - start).days + 1
    weight = call_weight(n_days, len(DAILY_VARIABLES))
    print(f"Open-Meteo {args.model}: {len(locations)} locations, {start} -> {end} "
          f"({n_days} days x {len(DAILY_VARIABLES)} vars = ~{math.ceil(weight)} weighted calls each)")

    if not args.build_only:
        session = make_session()
        todo = [l for l in locations if needs_fetch(l, args.model, start, end, args.force, args.refresh)]
        print(f"to fetch: {[l.key for l in todo] or 'nothing (raw files present)'}")
        for loc in todo:
            if not wait_for_budget(weight, args.hourly_budget, args.daily_budget):
                break
            try:
                fetch_location(session, loc, args.model, start, end, weight)
            except RateLimited:
                break
            # The 600/minute limit is weighted too, so space heavy requests out.
            polite_sleep(max(2.0, 60.0 * weight / MINUTELY_LIMIT + 5))

    build_outputs(locations, args.model)


if __name__ == "__main__":
    main()
