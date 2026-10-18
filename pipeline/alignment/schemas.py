"""
Data schemas, column requirements, and structural contracts for the weekly temporal alignment pipeline.

These schemas define the input and output boundaries for clean price, weather, DOA,
and DCS datasets without requiring direct imports of unmerged upstream modules.
"""

from typing import List, Set
import pandas as pd


# Official 25 administrative districts of Sri Lanka
OFFICIAL_DISTRICTS: Set[str] = {
    "Ampara", "Anuradhapura", "Badulla", "Batticaloa", "Colombo",
    "Galle", "Gampaha", "Hambantota", "Jaffna", "Kalutara",
    "Kandy", "Kegalle", "Kilinochchi", "Kurunegala", "Mannar",
    "Matale", "Matara", "Monaragala", "Mullaitivu", "Nuwara Eliya",
    "Polonnaruwa", "Puttalam", "Ratnapura", "Trincomalee", "Vavuniya"
}

# Required and optional columns for cleaned daily price data
CLEAN_PRICE_REQUIRED_COLUMNS: List[str] = [
    "obs_date",
    "district",
    "market_name",
    "crop_name",
    "price_type",
    "price_lkr_per_kg",
]

CLEAN_PRICE_OPTIONAL_COLUMNS: List[str] = [
    "source_commodity_name",
    "source_agency",
    "unit",
    "min_price_lkr_per_kg",
    "max_price_lkr_per_kg",
    "modal_price_lkr_per_kg",
    "origin_variety",
    "latitude",
    "longitude",
]

# Required and optional columns for cleaned daily weather data
CLEAN_WEATHER_REQUIRED_COLUMNS: List[str] = [
    "obs_date",
    "district",
    "precipitation_mm",
    "temp_mean_c",
]

CLEAN_WEATHER_OPTIONAL_COLUMNS: List[str] = [
    "temp_max_c",
    "temp_min_c",
    "relative_humidity_pct",
    "wind_speed_ms",
    "latitude",
    "longitude",
    "elevation_m",
]

# Required and optional columns for cleaned seasonal DOA cultivation targets
CLEAN_DOA_REQUIRED_COLUMNS: List[str] = [
    "season",
    "year",
    "district",
    "crop",
    "target_extent_ha",
    "cultivated_extent_ha",
    "expected_production_mt",
]

CLEAN_DOA_OPTIONAL_COLUMNS: List[str] = [
    "period_start",
    "period_end",
    "source_agency",
]

# Required and optional columns for cleaned annual DCS production/yield baselines
CLEAN_DCS_REQUIRED_COLUMNS: List[str] = [
    "crop",
    "district",
    "metric",
    "value",
    "unit",
]

CLEAN_DCS_OPTIONAL_COLUMNS: List[str] = [
    "year",
    "period_type",
    "period_start",
    "period_end",
    "is_estimated",
]


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    schema_name: str = "DataFrame",
    allow_empty: bool = False,
) -> None:
    """
    Validate that a DataFrame contains all required columns and is not empty.

    Args:
        df: Input pandas DataFrame to validate.
        required_columns: List of column names that must be present.
        schema_name: Descriptive name of the schema for error messages.
        allow_empty: If False, raises ValueError when the DataFrame is empty.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If required columns are missing or the DataFrame is empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected {schema_name} to be a pandas DataFrame, got {type(df).__name__}")

    if not allow_empty and df.empty:
        raise ValueError(f"{schema_name} is empty and contains no records.")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{schema_name} is missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
