"""
Data loading and preprocessing utilities for the Geo-Spatial DSS Dashboard.

Loads, validates, and caches the primary research datasets:
1. data/v1/analysis_ready_price.csv (110,019 weekly price records)
2. data/v1/analysis_ready_commitment.csv (2,875 seasonal commitment records)
3. data/v1/weather_data.json (100,225 daily agro-meteorological records)
4. data/processed/prices/dri_audit_ledger.jsonl (Real Data Readiness Index audits)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

# Optional Streamlit caching decorator fallback
try:
    import streamlit as st
    cache_decorator = st.cache_data
except ImportError:
    def cache_decorator(func):
        return func


def get_project_root() -> Path:
    """Find the root directory of the SLIIT research project repository."""
    # Current file is in <root>/geospatial-dss-dashboard/data_loader.py
    current_dir = Path(__file__).resolve().parent
    if (current_dir.parent / "data").exists():
        return current_dir.parent
    if (current_dir / "data").exists():
        return current_dir
    # Fallback to working directory
    cwd = Path.cwd()
    if (cwd / "data").exists():
        return cwd
    return current_dir.parent


@cache_decorator
def load_price_data() -> pd.DataFrame:
    """
    Load and clean weekly vegetable price dataset.

    Expected Schema:
    - price_date: YYYY-MM-DD (Monday anchor)
    - district: Administrative district
    - crop: Crop name (Carrot, Leek, Cabbage, Tomato, Big Onion (Local))
    - price_type: retail or wholesale
    - granularity: 'Retail - W' or 'Wholesale - W'
    - price: Rs./kg
    - source: 'harti'

    Returns:
        Cleaned pandas DataFrame.
    """
    root = get_project_root()
    price_path = root / "data" / "v1" / "analysis_ready_price.csv"

    if not price_path.exists():
        raise FileNotFoundError(f"Missing required price dataset: {price_path}")

    df = pd.read_csv(price_path)
    required = ["price_date", "district", "crop", "price_type", "price"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Price dataset missing required column '{col}'. Available: {df.columns.tolist()}")

    df["price_date"] = pd.to_datetime(df["price_date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price", "price_date"]).sort_values(by=["price_date", "district", "crop"])

    return df.reset_index(drop=True)


@cache_decorator
def load_commitment_data() -> pd.DataFrame:
    """
    Load seasonal farmer commitment and planning target dataset.

    Expected Schema:
    - district: Administrative district
    - crop_name: Target vegetable crop
    - season_type: Maha or Yala
    - season_year: 2014 to 2025
    - target_hectares: Baseline demand / production quota in hectares
    - committed_hectares: Registered farmer planting intention in hectares
    - source: 'synthetic_proxy'

    Returns:
        Cleaned pandas DataFrame.
    """
    root = get_project_root()
    comm_path = root / "data" / "v1" / "analysis_ready_commitment.csv"

    if not comm_path.exists():
        raise FileNotFoundError(f"Missing required commitment dataset: {comm_path}")

    df = pd.read_csv(comm_path)
    required = ["district", "crop_name", "season_type", "season_year", "target_hectares", "committed_hectares"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Commitment dataset missing required column '{col}'. Available: {df.columns.tolist()}")

    df["target_hectares"] = pd.to_numeric(df["target_hectares"], errors="coerce").fillna(0.0)
    df["committed_hectares"] = pd.to_numeric(df["committed_hectares"], errors="coerce").fillna(0.0)
    df["season_year"] = pd.to_numeric(df["season_year"], errors="coerce").astype(int)

    return df.reset_index(drop=True)


@cache_decorator
def load_weather_data() -> pd.DataFrame:
    """
    Load daily agro-meteorological dataset (NASA POWER).

    Expected Schema:
    - district_name_raw: District name (renamed to 'district')
    - obs_date: Observation date YYYY-MM-DD
    - rainfall_mm: Daily precipitation in mm
    - temp_c: Daily mean temperature in degrees Celsius
    - source: 'nasa_power'

    Returns:
        Cleaned pandas DataFrame.
    """
    root = get_project_root()
    weather_path = root / "data" / "v1" / "weather_data.json"

    if not weather_path.exists():
        raise FileNotFoundError(f"Missing required weather dataset: {weather_path}")

    with open(weather_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if "district_name_raw" in df.columns:
        df = df.rename(columns={"district_name_raw": "district"})

    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce").fillna(0.0)
    df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")

    return df.sort_values(by=["obs_date", "district"]).reset_index(drop=True)


@cache_decorator
def load_dri_audit_ledger() -> List[Dict[str, Any]]:
    """
    Load real Data Readiness Index (DRI) audit evaluations from the price pipeline ledger.

    Returns:
        List of audit record dictionaries.
    """
    root = get_project_root()
    ledger_path = root / "data" / "processed" / "prices" / "dri_audit_ledger.jsonl"

    if not ledger_path.exists():
        return []

    records = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def get_available_seasons(df_comm: pd.DataFrame) -> List[Tuple[str, int]]:
    """Get sorted list of available (season_type, season_year) tuples."""
    seasons = df_comm[["season_type", "season_year"]].drop_duplicates()
    seasons = seasons.sort_values(by=["season_year", "season_type"], ascending=[False, True])
    return list(zip(seasons["season_type"], seasons["season_year"]))
