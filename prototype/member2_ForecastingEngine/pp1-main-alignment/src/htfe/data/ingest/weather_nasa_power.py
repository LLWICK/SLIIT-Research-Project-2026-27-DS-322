"""Fetch daily agro-meteorology from the NASA POWER Daily API (community AG) and
build a tidy daily table.

Run from the ``prototype`` folder::

    python -m htfe.data.ingest.weather_nasa_power                  # fetch missing + build
    python -m htfe.data.ingest.weather_nasa_power --include-extra  # + optional markets
    python -m htfe.data.ingest.weather_nasa_power --build-only     # rebuild CSV from raw

API: https://power.larc.nasa.gov/api/temporal/daily/point
Docs: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
Data are on the source grids (meteorology ~0.5 x 0.625 deg from GEOS-IT/MERRA-2,
solar ~1 deg from CERES/FLASHFlux), so nearby towns can share a cell. Up to 20
parameters per point request; the service asks for one request per cell rather
than repeated calls. Missing values are -999 (converted to NaN here). Daily values
use Local Solar Time (LST) unless ``--time-standard UTC``.

Outputs
-------
raw/nasa_power/<location>.json        verbatim API response
raw/nasa_power/<location>.meta.json   request URL, params, status, time
processed/weather_daily_nasa_power.csv
processed/audit/weather_nasa_power_completeness.csv
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone

import pandas as pd

from htfe.config import AUDIT_DIR, processed_dir, raw_dir
from htfe.data.ingest.weather_common import (
    Location,
    append_fetch_log,
    completeness_profile,
    locations_frame,
    make_session,
    polite_sleep,
    request_with_retries,
    select_locations,
)

SOURCE = "nasa_power"
API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
FILL_VALUE = -999.0

# POWER parameter -> tidy column name (unit in the name).
PARAMETERS: dict[str, str] = {
    "T2M": "tmean_c",
    "T2M_MAX": "tmax_c",
    "T2M_MIN": "tmin_c",
    "PRECTOTCORR": "precip_mm",
    "RH2M": "rh_mean_pct",
    "WS2M": "wind_mean_2m_ms",
    "ALLSKY_SFC_SW_DWN": "shortwave_mj_m2",
    "GWETROOT": "soil_wet_root_frac",
    "GWETTOP": "soil_wet_top_frac",
}


def raw_paths(loc: Location):
    folder = raw_dir(SOURCE)
    return folder / f"{loc.key}.json", folder / f"{loc.key}.meta.json"


def needs_fetch(loc: Location, start: date, end: date, force: bool, refresh: bool) -> bool:
    data_path, meta_path = raw_paths(loc)
    if force or not data_path.exists() or not meta_path.exists():
        return True
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != 200 or meta["params"]["start"] > start.strftime("%Y%m%d"):
        return True
    return refresh and meta["params"]["end"] < end.strftime("%Y%m%d")


def fetch_location(session, loc: Location, start: date, end: date, time_standard: str) -> bool:
    data_path, meta_path = raw_paths(loc)
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "latitude": loc.lat,
        "longitude": loc.lon,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": time_standard,
    }
    log_path = raw_dir(SOURCE) / "_fetch_log.csv"
    try:
        resp = request_with_retries(session, API_URL, params, timeout=600, raise_on_429=False, backoff=20)
    except Exception as exc:  # network failure after retries: record and continue
        append_fetch_log(log_path, source=SOURCE, location=loc.key, status="error", message=repr(exc)[:300])
        print(f"  {loc.key}: failed after retries: {exc!r}")
        return False
    body = resp.content
    message = "" if resp.status_code == 200 else resp.text[:300]
    if resp.status_code == 200:
        data_path.write_bytes(body)
    meta = {
        "source": SOURCE,
        "location": loc.key,
        "name": loc.name,
        "url": resp.url,
        "params": params,
        "status": resp.status_code,
        "bytes": len(body),
        "elapsed_s": round(resp.elapsed.total_seconds(), 1),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "message": message,
    }
    if resp.status_code == 200 or not data_path.exists():
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    append_fetch_log(log_path, source=SOURCE, location=loc.key, status=resp.status_code, bytes=len(body),
                     url=resp.url, message=message)
    print(f"  {loc.key}: HTTP {resp.status_code}, {len(body) / 1024:.0f} KB in {meta['elapsed_s']}s {message}")
    return resp.status_code == 200


def parse_raw(loc: Location) -> tuple[pd.DataFrame, dict]:
    data_path, _ = raw_paths(loc)
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    series = payload["properties"]["parameter"]
    daily = pd.DataFrame({PARAMETERS[p]: pd.Series(v, dtype="float64") for p, v in series.items() if p in PARAMETERS})
    daily = daily.replace(FILL_VALUE, float("nan"))
    daily.index = pd.to_datetime(daily.index, format="%Y%m%d")
    daily = daily.sort_index()
    has_value = daily.notna().any(axis=1)
    if has_value.any():
        daily = daily.loc[: has_value[has_value].index.max()]
    daily = daily.reindex(columns=list(PARAMETERS.values()))
    daily.insert(0, "date", daily.index.date.astype(str))
    daily.insert(1, "location", loc.key)
    daily.insert(2, "lat", loc.lat)
    daily.insert(3, "lon", loc.lon)
    coords = payload.get("geometry", {}).get("coordinates", [None, None, None])
    grid = {
        "location": loc.key,
        "nasa_grid_elevation_m": coords[2] if len(coords) > 2 else None,
        "nasa_sources": "|".join(payload.get("header", {}).get("sources", [])),
        "nasa_api_version": payload.get("header", {}).get("api", {}).get("version"),
    }
    return daily.reset_index(drop=True), grid


def build_outputs(locations: list[Location]) -> None:
    out_dir = processed_dir()
    frames, grids = [], []
    for loc in locations:
        if not raw_paths(loc)[0].exists():
            print(f"  (no raw file for {loc.key}; skipped in build)")
            continue
        daily, grid = parse_raw(loc)
        frames.append(daily)
        grids.append(grid)
    if not frames:
        print("Nothing to build.")
        return
    daily = pd.concat(frames, ignore_index=True)
    daily.to_csv(out_dir / "weather_daily_nasa_power.csv", index=False)

    loc_path = out_dir / "weather_locations.csv"
    locs = pd.read_csv(loc_path) if loc_path.exists() else locations_frame()
    locs = locs.drop(columns=[c for c in locs.columns if c.startswith("nasa_")])
    locs = locs.merge(pd.DataFrame(grids), on="location", how="left")
    locs.to_csv(loc_path, index=False)

    profile = completeness_profile(daily, list(PARAMETERS.values()), pd.Timestamp(date.today()))
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    profile.to_csv(AUDIT_DIR / "weather_nasa_power_completeness.csv", index=False)
    print(f"daily rows {len(daily):,} | locations {daily['location'].nunique()} | "
          f"{daily['date'].min()} -> {daily['date'].max()}")
    cols = ["location", "first_date", "last_date", "missing_dates", "na_precip_mm",
            "latency_days_precip_mm", "latency_days_shortwave_mj_m2"]
    print(profile[cols].to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2010-01-01", help="first date (YYYY-MM-DD), default 2010-01-01")
    parser.add_argument("--end", default=None, help="last date requested, default today (trailing -999 trimmed)")
    parser.add_argument("--locations", default=None, help="comma-separated location keys (default core set)")
    parser.add_argument("--include-extra", action="store_true", help="also fetch optional market centroids")
    parser.add_argument("--time-standard", default="LST", choices=["LST", "UTC"], help="daily boundary, default LST")
    parser.add_argument("--force", action="store_true", help="re-fetch even if raw files exist")
    parser.add_argument("--refresh", action="store_true", help="re-fetch raw files that end before --end")
    parser.add_argument("--build-only", action="store_true", help="skip network; rebuild CSV from raw files")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between requests (default 3)")
    args = parser.parse_args(argv)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    locations = select_locations(args.locations, args.include_extra)
    print(f"NASA POWER daily (AG): {len(locations)} locations, {start} -> {end}")

    if not args.build_only:
        session = make_session()
        todo = [l for l in locations if needs_fetch(l, start, end, args.force, args.refresh)]
        print(f"to fetch: {[l.key for l in todo] or 'nothing (raw files present)'}")
        for loc in todo:
            fetch_location(session, loc, start, end, args.time_standard)
            polite_sleep(args.delay)

    build_outputs(locations)


if __name__ == "__main__":
    main()
