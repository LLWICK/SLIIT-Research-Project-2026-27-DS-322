"""Weather data cleaning, standardization, and schema normalization.

Transforms raw NASA POWER JSON payloads into clean, standardized datasets.
Preserves missing values as NaN/null (does NOT substitute with zero).
Aligns observations to standard calendar dates and Sri Lankan seasons.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pipeline.config import (
    MAHA_MONTHS,
    NASA_PARAMETER_METADATA,
    NASA_POWER_FILL_VALUE,
    PROCESSED_WEATHER_DIR,
    YALA_MONTHS,
)

logger = logging.getLogger("transformer.weather")


def season_from_month(month: int) -> str:
    """Return 'Yala' (Apr-Sep) or 'Maha' (Oct-Mar)."""
    return "Yala" if month in YALA_MONTHS else "Maha"


def season_year_from_date(year: int, month: int) -> int:
    """Return cultivation season year.

    For Maha (Oct-Mar), months Jan-Mar belong to the season year of the October start.
    """
    if month in (1, 2, 3):
        return year - 1
    return year


class WeatherCleaner:
    """Cleans and standardizes raw NASA POWER weather responses."""

    def __init__(self, output_dir: Path = PROCESSED_WEATHER_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def clean_raw_payload(
        self,
        raw_json: dict[str, Any],
        district: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> pd.DataFrame:
        """Parse raw NASA POWER JSON and produce a cleaned, standardized DataFrame.

        Strict Rules:
        1. Dates converted to standard 'YYYY-MM-DD' ISO format.
        2. NASA fill values (-999.0) preserved as NaN (never imputed to zero).
        3. Standardized snake_case column names and units.
        4. Elevation extracted from geometry coordinates when present.
        5. Season classification (Yala / Maha) mapped accurately.
        """
        properties = raw_json.get("properties", {})
        parameter_dict = properties.get("parameter", {})
        if not parameter_dict:
            raise ValueError("No 'parameter' object found in NASA POWER response.")

        geometry = raw_json.get("geometry", {})
        coords = geometry.get("coordinates", [])
        lon = coords[0] if len(coords) > 0 else longitude
        lat = coords[1] if len(coords) > 1 else latitude
        elevation = coords[2] if len(coords) > 2 else None

        # Build table from parameter dates
        # Collect all dates across parameters
        all_dates = set()
        for param_data in parameter_dict.values():
            all_dates.update(param_data.keys())

        if not all_dates:
            logger.warning(f"No date entries found for district {district}.")
            return pd.DataFrame()

        sorted_dates = sorted(all_dates)
        records = []
        for d_str in sorted_dates:
            # Parse 'YYYYMMDD'
            dt = pd.to_datetime(d_str, format="%Y%m%d")
            iso_date = dt.strftime("%Y-%m-%d")
            year = dt.year
            month = dt.month

            row: dict[str, Any] = {
                "obs_date": iso_date,
                "district": district,
                "latitude": round(lat, 4) if lat is not None else None,
                "longitude": round(lon, 4) if lon is not None else None,
                "elevation_m": elevation,
                "year": year,
                "month": month,
                "season_type": season_from_month(month),
                "season_year": season_year_from_date(year, month),
            }

            for param_code, param_vals in parameter_dict.items():
                meta = NASA_PARAMETER_METADATA.get(param_code, {})
                col_name = meta.get("standard_name", param_code.lower())
                raw_val = param_vals.get(d_str)

                # Strictly check fill values: -999.0 or -999 indicates missing data
                if raw_val is None or raw_val == NASA_POWER_FILL_VALUE or raw_val == -999:
                    val = np.nan
                else:
                    try:
                        val = float(raw_val)
                    except (ValueError, TypeError):
                        val = np.nan
                row[col_name] = val

            row["source_provenance"] = "NASA_POWER_MERRA2"
            records.append(row)

        df = pd.DataFrame(records)

        # Enforce column order
        lead_cols = [
            "obs_date",
            "district",
            "latitude",
            "longitude",
            "elevation_m",
            "year",
            "month",
            "season_type",
            "season_year",
        ]
        measure_cols = [
            c for c in df.columns if c not in lead_cols and c != "source_provenance"
        ]
        df = df[lead_cols + measure_cols + ["source_provenance"]]

        logger.info(
            f"Cleaned {len(df)} daily observations for {district} "
            f"({df['obs_date'].min()} to {df['obs_date'].max()})"
        )
        return df

    def clean_from_file(self, raw_json_path: Path | str, district: str) -> pd.DataFrame:
        """Load a raw JSON file from disk and return cleaned DataFrame."""
        path = Path(raw_json_path)
        with open(path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)
        return self.clean_raw_payload(raw_json, district=district)

    def save_clean(
        self,
        df: pd.DataFrame,
        district: str,
        start_date: str,
        end_date: str,
        file_format: str = "csv",
    ) -> Path:
        """Save cleaned DataFrame to the processed directory."""
        district_slug = district.lower().replace(" ", "_")
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        
        file_name = f"{district_slug}_{start_fmt}_{end_fmt}_clean.{file_format}"
        out_path = self.output_dir / file_name

        if file_format == "csv":
            df.to_csv(out_path, index=False)
        elif file_format == "parquet":
            df.to_parquet(out_path, index=False)
        else:
            raise ValueError(f"Unsupported format '{file_format}'. Use 'csv' or 'parquet'.")

        logger.info(f"Saved cleaned dataset to {out_path} ({out_path.stat().st_size:,} bytes)")
        return out_path
