"""Load and cache the three prototype datasets supplied in data/v1/.

Original files are read-only. A derived monthly weather pickle may be
written under dashboard/.cache/ so Streamlit does not parse the large
JSON file on every interaction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "v1"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

COMMITMENT_PATH = DATA_DIR / "analysis_ready_commitment.csv"
PRICE_PATH = DATA_DIR / "analysis_ready_price.csv"
WEATHER_PATH = DATA_DIR / "weather_data.json"
WEATHER_CACHE_PATH = CACHE_DIR / "weather_monthly.pkl"

# PROTOTYPE ASSUMPTION: Sri Lankan cultivation seasons mapped to calendar months.
# Maha is treated as Oct–Mar (season year = year of the October start).
# Yala is treated as Apr–Sep of the same calendar year.
YALA_MONTHS = (4, 5, 6, 7, 8, 9)
MAHA_MONTHS = (10, 11, 12, 1, 2, 3)
EXPECTED_SEASON_DAYS = {"Yala": 183, "Maha": 182}


def season_from_month(month: int) -> str:
    return "Yala" if month in YALA_MONTHS else "Maha"


def season_year_from_date(ts: pd.Timestamp) -> int:
    if ts.month in (1, 2, 3):
        return int(ts.year - 1)
    return int(ts.year)


def load_commitment() -> pd.DataFrame:
    df = pd.read_csv(COMMITMENT_PATH)
    df["district"] = df["district"].astype(str).str.strip()
    df["crop_name"] = df["crop_name"].astype(str).str.strip()
    df["season_type"] = df["season_type"].astype(str).str.strip()
    df["season_year"] = pd.to_numeric(df["season_year"], errors="coerce").astype("Int64")
    df["target_hectares"] = pd.to_numeric(df["target_hectares"], errors="coerce")
    df["committed_hectares"] = pd.to_numeric(df["committed_hectares"], errors="coerce")
    df["source"] = df["source"].astype(str).str.strip()
    df["cultivation_intensity"] = df["committed_hectares"] / df["target_hectares"].replace(0, pd.NA)
    return df


def load_price() -> pd.DataFrame:
    df = pd.read_csv(PRICE_PATH)
    df["district"] = df["district"].astype(str).str.strip()
    df["crop"] = df["crop"].astype(str).str.strip()
    df["price_type"] = df["price_type"].astype(str).str.strip()
    df["granularity"] = df["granularity"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")
    df = df.dropna(subset=["price_date", "price"])
    df["year"] = df["price_date"].dt.year
    df["month"] = df["price_date"].dt.month
    df["season_type"] = df["month"].map(season_from_month)
    df["season_year"] = df["price_date"].map(season_year_from_date)
    return df


def _aggregate_weather(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["district"] = raw["district_name_raw"].astype(str).str.strip()
    raw["obs_date"] = pd.to_datetime(raw["obs_date"], errors="coerce")
    raw["rainfall_mm"] = pd.to_numeric(raw["rainfall_mm"], errors="coerce")
    raw["temp_c"] = pd.to_numeric(raw["temp_c"], errors="coerce")
    raw = raw.dropna(subset=["obs_date", "district"])
    raw["year"] = raw["obs_date"].dt.year
    raw["month"] = raw["obs_date"].dt.month
    monthly = (
        raw.groupby(["district", "year", "month"], as_index=False)
        .agg(
            rainfall_mm=("rainfall_mm", "sum"),
            temp_c=("temp_c", "mean"),
            n_days=("obs_date", "count"),
            source=("source", "first"),
        )
    )
    monthly["season_type"] = monthly["month"].map(season_from_month)
    monthly["season_year"] = monthly.apply(
        lambda r: int(r["year"] - 1) if int(r["month"]) in (1, 2, 3) else int(r["year"]),
        axis=1,
    )
    return monthly


def load_weather_monthly(force_reload: bool = False) -> pd.DataFrame:
    """Daily NASA POWER JSON → monthly district summaries, cached on disk."""
    if WEATHER_CACHE_PATH.exists() and not force_reload:
        return pd.read_pickle(WEATHER_CACHE_PATH)

    with open(WEATHER_PATH, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    monthly = _aggregate_weather(pd.DataFrame(records))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    monthly.to_pickle(WEATHER_CACHE_PATH)
    return monthly


def seasonal_weather(monthly: pd.DataFrame) -> pd.DataFrame:
    seasonal = (
        monthly.groupby(["district", "season_year", "season_type"], as_index=False)
        .agg(
            rainfall_mm=("rainfall_mm", "sum"),
            temp_c=("temp_c", "mean"),
            n_days=("n_days", "sum"),
            source=("source", "first"),
        )
    )
    seasonal["expected_days"] = seasonal["season_type"].map(EXPECTED_SEASON_DAYS)
    seasonal["completeness"] = (
        seasonal["n_days"] / seasonal["expected_days"].replace(0, pd.NA)
    ).clip(upper=1.0)
    return seasonal


def weather_completeness_by_district(monthly: pd.DataFrame) -> pd.DataFrame:
    span_days = (monthly.groupby("district")["n_days"].sum()).rename("n_days")
    # Prototype: completeness vs the district with the most daily rows.
    max_days = float(span_days.max()) if len(span_days) else 1.0
    out = span_days.reset_index()
    out["completeness"] = (out["n_days"] / max_days).clip(upper=1.0)
    return out


def price_latest_table(price: pd.DataFrame) -> pd.DataFrame:
    ordered = price.sort_values(["district", "crop", "price_type", "price_date"])
    latest = ordered.groupby(["district", "crop", "price_type"], as_index=False).tail(1)
    return latest[
        ["district", "crop", "price_type", "price_date", "price", "granularity", "source"]
    ].rename(columns={"price": "latest_price", "source": "price_source"})


def price_seasonal_table(price: pd.DataFrame) -> pd.DataFrame:
    return (
        price.groupby(
            ["district", "crop", "price_type", "season_year", "season_type"],
            as_index=False,
        )
        .agg(
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            n_obs=("price", "count"),
            price_source=("source", "first"),
        )
    )


def price_observation_counts(price: pd.DataFrame) -> pd.DataFrame:
    counts = (
        price.groupby(["district", "crop", "price_type"], as_index=False)
        .size()
        .rename(columns={"size": "n_obs"})
    )
    medians = counts.groupby(["crop", "price_type"])["n_obs"].transform("median")
    counts["completeness"] = (counts["n_obs"] / medians.replace(0, pd.NA)).clip(upper=1.0)
    return counts


def filter_commitment(
    commitment: pd.DataFrame,
    year: int,
    season: str,
    crop: str | None = None,
    district: str | None = None,
) -> pd.DataFrame:
    out = commitment[
        (commitment["season_year"] == year) & (commitment["season_type"] == season)
    ]
    if crop:
        out = out[out["crop_name"] == crop]
    if district:
        out = out[out["district"] == district]
    return out.copy()
