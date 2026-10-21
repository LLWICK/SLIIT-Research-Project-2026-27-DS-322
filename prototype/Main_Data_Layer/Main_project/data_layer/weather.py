"""Download NASA POWER daily weather for the supply districts into local DuckDB."""

import json
from datetime import date

import duckdb
import pandas as pd
import requests

from data_layer.paths import DUCKDB_PATH, EXPORT_DIR, WAREHOUSE_DIR
from data_layer.scope import WEATHER_POINTS

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = ("T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M")
START = date(2015, 1, 1)
END = date(2026, 10, 20)
FILL = -999.0
# The daily PRECTOTCORR value is about 24 times the sum of that day's hourly
# mm/hour values. Checked for Badulla on 2015-01-01: daily 61.68 mm, hourly sum 2.59 mm.
PRECIP_HOUR_FACTOR = 24
CACHE_DIR = WAREHOUSE_DIR / "weather_cache"


def load_weather(refresh: bool = False) -> dict[str, int]:
    frames = []
    points = []
    for code, (name, district, lat, lon) in WEATHER_POINTS.items():
        payload = _fetch(code, lat, lon, refresh)
        frames.append(_daily_frame(code, payload))
        points.append(_point_row(code, name, district, lat, lon, payload))

    daily = pd.concat(frames, ignore_index=True)
    point_table = pd.DataFrame(points)
    _write(daily, point_table)
    return {
        "points": len(point_table),
        "days": len(daily),
        "missing": int(daily["is_missing"].sum()),
    }


def _fetch(code: str, lat: float, lon: float, refresh: bool) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{code}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))

    response = requests.get(
        POWER_URL,
        params={
            "parameters": ",".join(PARAMETERS),
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": START.strftime("%Y%m%d"),
            "end": END.strftime("%Y%m%d"),
            "format": "JSON",
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    cache.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _point_row(code: str, name: str, district: str, lat: float, lon: float, payload: dict) -> dict:
    grid_lon, grid_lat, elevation = payload["geometry"]["coordinates"]
    return {
        "point_code": code,
        "point_name": name,
        "district": district,
        "requested_lat": lat,
        "requested_lon": lon,
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "grid_elevation_m": elevation,
        "source": "nasa_power",
        "precip_divisor": PRECIP_HOUR_FACTOR,
        "time_standard": payload.get("header", {}).get("time_standard"),
        "start_date": START.isoformat(),
        "end_date": END.isoformat(),
    }


def _daily_frame(code: str, payload: dict) -> pd.DataFrame:
    series = payload["properties"]["parameter"]
    dates = sorted(series["T2M"])
    frame = pd.DataFrame({"obs_date": pd.to_datetime(dates, format="%Y%m%d").date, "point_code": code})
    rename = {
        "T2M": "t2m_c",
        "T2M_MAX": "t2m_max_c",
        "T2M_MIN": "t2m_min_c",
        "PRECTOTCORR": "precip_mm",
        "RH2M": "rh2m_pct",
    }
    for source, column in rename.items():
        values = pd.Series([series[source][day] for day in dates], dtype="float64")
        values = values.mask(values == FILL)
        if source == "PRECTOTCORR":
            values = values / PRECIP_HOUR_FACTOR
        frame[column] = values
    value_columns = list(rename.values())
    frame["is_missing"] = frame[value_columns].isna().any(axis=1)
    return frame


def _write(daily: pd.DataFrame, points: pd.DataFrame) -> None:
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS curated")
        con.register("weather_daily", daily)
        con.register("weather_points", points)
        con.execute("CREATE OR REPLACE TABLE curated.dim_weather_point AS SELECT * FROM weather_points")
        con.execute("CREATE OR REPLACE TABLE curated.weather_daily AS SELECT * FROM weather_daily")
        export_path = str(EXPORT_DIR / "weather_daily.parquet").replace("\\", "/")
        con.execute(f"COPY curated.weather_daily TO '{export_path}' (FORMAT PARQUET)")
    finally:
        con.close()


def main() -> None:
    counts = load_weather()
    print(f"warehouse: {DUCKDB_PATH}")
    print(f"points: {counts['points']}  days: {counts['days']:,}  missing: {counts['missing']:,}")


if __name__ == "__main__":
    main()
