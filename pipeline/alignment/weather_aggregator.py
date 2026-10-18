"""
Weather aggregation engine for the weekly temporal alignment pipeline.

Aggregates daily agro-meteorological observations into standardized Monday-to-Sunday
ISO weekly features: strictly summing precipitation to capture hydrological load,
averaging temperatures, humidity, and wind speed, and tracking observation counts.
"""

from typing import Optional
import pandas as pd
import numpy as np

from pipeline.alignment.schemas import CLEAN_WEATHER_REQUIRED_COLUMNS, validate_dataframe
from pipeline.alignment.calendar_utils import add_iso_week_columns


class WeatherAggregator:
    """
    Transforms daily district-level weather records into weekly aggregated features.
    """

    def __init__(self, decimals: int = 2):
        """
        Initialize the weather aggregator.

        Args:
            decimals: Number of decimal places to round aggregated metrics to.
        """
        self.decimals = decimals

    def aggregate(self, daily_weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily weather records to weekly features per district and ISO week.

        Aggregation Rules:
        - precipitation_mm: STRICTLY SUMMED (weekly total accumulation)
        - temp_mean_c: ARITHMETIC MEAN
        - temp_min_c: MINIMUM (if present)
        - temp_max_c: MAXIMUM (if present)
        - relative_humidity_pct: ARITHMETIC MEAN (if present)
        - wind_speed_ms: ARITHMETIC MEAN (if present)
        - weather_obs_count: COUNT of daily observations in the week

        Args:
            daily_weather_df: Input DataFrame containing daily weather records.

        Returns:
            Aggregated pandas DataFrame with weekly weather features.
        """
        validate_dataframe(daily_weather_df, CLEAN_WEATHER_REQUIRED_COLUMNS, "CleanWeather")

        # Map daily dates to ISO 8601 week parameters
        df_with_weeks = add_iso_week_columns(daily_weather_df, date_col="obs_date")

        group_cols = [
            "district",
            "iso_year",
            "iso_week",
            "week_start_date",
            "week_end_date",
        ]

        # Prepare aggregation dictionary based on available columns
        agg_spec = {
            "precipitation_mm": [("weekly_precip_sum_mm", "sum")],
            "temp_mean_c": [("weekly_temp_mean_c", "mean")],
            "obs_date": [("weather_obs_count", "count")],
        }

        if "temp_min_c" in df_with_weeks.columns:
            agg_spec["temp_min_c"] = [("weekly_temp_min_c", "min")]
        if "temp_max_c" in df_with_weeks.columns:
            agg_spec["temp_max_c"] = [("weekly_temp_max_c", "max")]
        if "relative_humidity_pct" in df_with_weeks.columns:
            agg_spec["relative_humidity_pct"] = [("weekly_humidity_mean_pct", "mean")]
        if "wind_speed_ms" in df_with_weeks.columns:
            agg_spec["wind_speed_ms"] = [("weekly_wind_speed_mean_ms", "mean")]

        # Group and aggregate
        grouped = df_with_weeks.groupby(group_cols, as_index=False)
        
        # Flatten multi-level aggregation output
        rows = []
        for group_keys, group_df in grouped:
            district, iso_year, iso_week, week_start, week_end = group_keys
            
            row = {
                "district": district,
                "iso_year": iso_year,
                "iso_week": iso_week,
                "week_start_date": week_start,
                "week_end_date": week_end,
                "weekly_precip_sum_mm": round(float(group_df["precipitation_mm"].sum()), self.decimals),
                "weekly_temp_mean_c": round(float(group_df["temp_mean_c"].mean()), self.decimals),
                "weather_obs_count": int(len(group_df)),
            }

            if "temp_min_c" in group_df.columns:
                val = group_df["temp_min_c"].min()
                row["weekly_temp_min_c"] = round(float(val), self.decimals) if pd.notna(val) else None
            if "temp_max_c" in group_df.columns:
                val = group_df["temp_max_c"].max()
                row["weekly_temp_max_c"] = round(float(val), self.decimals) if pd.notna(val) else None
            if "relative_humidity_pct" in group_df.columns:
                val = group_df["relative_humidity_pct"].mean()
                row["weekly_humidity_mean_pct"] = round(float(val), self.decimals) if pd.notna(val) else None
            if "wind_speed_ms" in group_df.columns:
                val = group_df["wind_speed_ms"].mean()
                row["weekly_wind_speed_mean_ms"] = round(float(val), self.decimals) if pd.notna(val) else None

            # Retain coordinates if present
            if "latitude" in group_df.columns:
                lat = group_df["latitude"].dropna()
                if not lat.empty:
                    row["latitude"] = float(lat.iloc[0])
            if "longitude" in group_df.columns:
                lon = group_df["longitude"].dropna()
                if not lon.empty:
                    row["longitude"] = float(lon.iloc[0])

            rows.append(row)

        weekly_weather_df = pd.DataFrame(rows)
        if not weekly_weather_df.empty:
            weekly_weather_df = weekly_weather_df.sort_values(
                by=["district", "iso_year", "iso_week"]
            ).reset_index(drop=True)

        return weekly_weather_df
