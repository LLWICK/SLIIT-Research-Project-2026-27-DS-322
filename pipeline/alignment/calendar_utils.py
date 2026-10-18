"""
ISO 8601 calendar utilities and Sri Lankan agricultural seasonal windowing.

Provides functions for deterministic Monday-to-Sunday weekly calculations,
ISO year/week boundary resolution, and agricultural season (Maha / Yala) mapping.
"""

from typing import Union, Tuple, Dict
import datetime
import pandas as pd


def parse_date(date_val: Union[str, datetime.date, datetime.datetime, pd.Timestamp]) -> datetime.date:
    """
    Parse a flexible date input into a standard datetime.date object.

    Args:
        date_val: A string (YYYY-MM-DD), date, datetime, or pandas Timestamp.

    Returns:
        Standard datetime.date.

    Raises:
        ValueError: If the date cannot be parsed.
    """
    if isinstance(date_val, datetime.datetime):
        return date_val.date()
    elif isinstance(date_val, datetime.date):
        return date_val
    elif isinstance(date_val, pd.Timestamp):
        return date_val.date()
    elif isinstance(date_val, str):
        # Strip potential timestamps or whitespace
        clean_str = date_val.strip().split("T")[0].split(" ")[0]
        try:
            return datetime.datetime.strptime(clean_str, "%Y-%m-%d").date()
        except ValueError:
            # Fall back to pandas parser for ISO / unusual formats
            parsed = pd.to_datetime(clean_str)
            return parsed.date()
    else:
        raise ValueError(f"Unsupported date format: {type(date_val)} ({date_val})")


def get_iso_week_metadata(date_val: Union[str, datetime.date, datetime.datetime, pd.Timestamp]) -> Dict[str, Union[str, int]]:
    """
    Calculate the Monday-to-Sunday ISO 8601 week parameters for any date.

    Args:
        date_val: Date to calculate ISO week parameters for.

    Returns:
        Dict containing:
        - "week_start_date": ISO date string of Monday (YYYY-MM-DD).
        - "week_end_date": ISO date string of Sunday (YYYY-MM-DD).
        - "iso_year": ISO 8601 calendar year (int).
        - "iso_week": ISO 8601 calendar week number (1..53) (int).
    """
    d = parse_date(date_val)
    iso_year, iso_week, _ = d.isocalendar()

    # In Python 3.8+, date.fromisocalendar(year, week, day) gives:
    # day 1 = Monday, day 7 = Sunday
    monday = datetime.date.fromisocalendar(iso_year, iso_week, 1)
    sunday = datetime.date.fromisocalendar(iso_year, iso_week, 7)

    return {
        "week_start_date": monday.isoformat(),
        "week_end_date": sunday.isoformat(),
        "iso_year": int(iso_year),
        "iso_week": int(iso_week),
    }


def get_agricultural_season(date_val: Union[str, datetime.date, datetime.datetime, pd.Timestamp]) -> Dict[str, Union[str, int]]:
    """
    Determine the Sri Lankan agricultural season (Maha or Yala) and its planting season year.

    Definitions from Department of Agriculture (DOA):
    - Yala Season: April 1 through September 30 (Southwest monsoon period).
      Season year is the current calendar year.
    - Maha Season: October 1 through March 31 of following year (Northeast monsoon period).
      Season year represents the planting base year (e.g., Oct 2023 - Mar 2024 is Maha 2023/24, base 2023).

    Args:
        date_val: Date to resolve season for.

    Returns:
        Dict containing:
        - "season_type": "Maha" or "Yala"
        - "season_year": The planting season year (int)
        - "season_label": Full descriptive string (e.g. "Maha 2023/2024" or "Yala 2024")
    """
    d = parse_date(date_val)
    month = d.month
    year = d.year

    if 4 <= month <= 9:
        season_type = "Yala"
        season_year = year
        season_label = f"Yala {year}"
    elif month >= 10:
        season_type = "Maha"
        season_year = year
        season_label = f"Maha {year}/{year + 1}"
    else:  # month in [1, 2, 3]
        season_type = "Maha"
        season_year = year - 1
        season_label = f"Maha {year - 1}/{year}"

    return {
        "season_type": season_type,
        "season_year": int(season_year),
        "season_label": season_label,
    }


def add_iso_week_columns(df: pd.DataFrame, date_col: str = "obs_date") -> pd.DataFrame:
    """
    Add ISO week metadata columns (week_start_date, week_end_date, iso_year, iso_week)
    to a DataFrame based on an existing date column.

    Args:
        df: Input DataFrame.
        date_col: Name of the date column to parse.

    Returns:
        New DataFrame with ISO week metadata appended.
    """
    if df.empty or date_col not in df.columns:
        return df.copy()

    res_df = df.copy()
    
    # Map dates to ISO week metadata
    metadata = [get_iso_week_metadata(val) for val in res_df[date_col]]
    
    res_df["week_start_date"] = [m["week_start_date"] for m in metadata]
    res_df["week_end_date"] = [m["week_end_date"] for m in metadata]
    res_df["iso_year"] = [m["iso_year"] for m in metadata]
    res_df["iso_week"] = [m["iso_week"] for m in metadata]

    return res_df
