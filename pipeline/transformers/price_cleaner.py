"""Vegetable Price Data Cleaner and Harmonizer.

Transforms, normalizes, and enriches raw price observation records extracted from
official sources (CBSL and HARTI).

Guarantees:
1. Preserves original market identities, original crop names, and reported units.
2. Preserves min, max, and modal price values without truncation or synthetic substitution.
3. Preserves explicit nulls/missing values (never fabricates prices or fills with zero).
4. Assigns Sri Lankan agricultural cultivation seasons (Maha / Yala) and administrative districts.
5. Emits normalized tabular DataFrames for storage in the clean warehouse layer.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pipeline.config import (
    CROP_NAME_MAPPINGS,
    MAHA_MONTHS,
    MARKET_CENTERS,
    PROCESSED_PRICE_DIR,
    TARGET_CROPS,
    YALA_MONTHS,
)

logger = logging.getLogger("transformer.price")


class PriceCleaner:
    """Standardizes, normalizes, and enriches vegetable price observation records."""

    def __init__(self, processed_output_dir: Path = PROCESSED_PRICE_DIR) -> None:
        self.processed_output_dir = Path(processed_output_dir)
        self.processed_output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def determine_season(dt: datetime) -> tuple[str, int]:
        """Determine Sri Lankan agricultural season (Maha / Yala) and season year.

        - Maha: October to March. If month is Jan-Mar, season_year is (year - 1).
        - Yala: April to September. season_year is the current year.
        """
        year = dt.year
        month = dt.month

        if month in YALA_MONTHS:
            return "Yala", year
        elif month in (10, 11, 12):
            return "Maha", year
        else:  # Jan, Feb, Mar
            return "Maha", year - 1

    def clean_records(
        self,
        raw_records: list[dict[str, Any]],
        filter_target_crops: bool = True,
    ) -> pd.DataFrame:
        """Clean and normalize a list of raw price observation dictionaries.

        Args:
            raw_records: List of dictionaries from price extractors.
            filter_target_crops: If True, retains only records matching the 5 research target crops.

        Returns:
            Cleaned and normalized pandas DataFrame.
        """
        if not raw_records:
            logger.warning("Empty records list provided to PriceCleaner.")
            return pd.DataFrame()

        df = pd.DataFrame(raw_records)

        # 1. Standardize dates
        date_str_series = df["price_date"].astype(str).str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
        df["price_date"] = pd.to_datetime(date_str_series, errors="coerce")
        invalid_dates = df["price_date"].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Dropped {invalid_dates} rows with unparseable dates.")
            df = df.dropna(subset=["price_date"]).copy()

        df["obs_date"] = df["price_date"].dt.strftime("%Y-%m-%d")
        df["year"] = df["price_date"].dt.year
        df["month"] = df["price_date"].dt.month
        df["day"] = df["price_date"].dt.day
        df["day_of_week"] = df["price_date"].dt.day_name()

        # 2. Assign agricultural seasons
        seasons_info = [self.determine_season(dt) for dt in df["price_date"]]
        df["season_type"] = [s[0] for s in seasons_info]
        df["season_year"] = [s[1] for s in seasons_info]

        # 3. Standardize crop names while strictly preserving original commodity label
        if "source_commodity_name" not in df.columns:
            df["source_commodity_name"] = df["crop_name"]

        # Map to canonical names
        df["crop_name_standardized"] = df["source_commodity_name"].apply(
            lambda x: CROP_NAME_MAPPINGS.get(str(x).lower().strip(), str(x).strip())
        )
        df["crop_name"] = df["crop_name_standardized"]

        # Flag target crops
        df["is_target_crop"] = df["crop_name"].isin(TARGET_CROPS)

        if filter_target_crops:
            initial_count = len(df)
            df = df[df["is_target_crop"]].copy()
            logger.info(
                f"Filtered to research target crops: retained {len(df)} of {initial_count} records."
            )

        # 4. Standardize market identities and enrich district metadata
        if "original_market_name" not in df.columns:
            df["original_market_name"] = df["market_name"]

        def lookup_market_info(mkt_str: str) -> tuple[str, str, Optional[float], Optional[float]]:
            info = MARKET_CENTERS.get(mkt_str)
            if info:
                return (
                    info["market_name"],
                    info["district"],
                    info.get("latitude"),
                    info.get("longitude"),
                )
            return (mkt_str, "Unknown", None, None)

        market_lookups = [lookup_market_info(m) for m in df["original_market_name"]]
        df["market_name"] = [m[0] for m in market_lookups]
        df["district"] = [m[1] for m in market_lookups]
        df["latitude"] = [m[2] for m in market_lookups]
        df["longitude"] = [m[3] for m in market_lookups]

        # 5. Standardize price types
        df["price_type"] = df["price_type"].str.lower().str.strip()

        # 6. Ensure price columns are numeric float, preserving NaN for missing values
        price_cols = [
            "price_lkr_per_kg",
            "min_price_lkr_per_kg",
            "max_price_lkr_per_kg",
            "modal_price_lkr_per_kg",
        ]
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan

        # 7. Ensure units are preserved
        if "unit" not in df.columns:
            df["unit"] = "Rs./kg"
        else:
            df["unit"] = df["unit"].fillna("Rs./kg")

        # 8. Define column ordering for clean warehouse/export schema
        ordered_columns = [
            "obs_date",
            "year",
            "month",
            "day",
            "season_type",
            "season_year",
            "crop_name",
            "source_commodity_name",
            "market_name",
            "original_market_name",
            "district",
            "latitude",
            "longitude",
            "price_type",
            "price_lkr_per_kg",
            "min_price_lkr_per_kg",
            "max_price_lkr_per_kg",
            "modal_price_lkr_per_kg",
            "unit",
            "source_agency",
            "source_url",
            "provenance_sha256",
        ]

        # Retain any auxiliary columns if present
        final_cols = [c for c in ordered_columns if c in df.columns]
        for c in df.columns:
            if c not in final_cols and c not in ("price_date", "crop_name_standardized", "is_target_crop"):
                final_cols.append(c)

        result_df = df[final_cols].sort_values(
            by=["obs_date", "crop_name", "market_name", "price_type"]
        ).reset_index(drop=True)

        return result_df

    def save_processed(
        self,
        df: pd.DataFrame,
        filename_prefix: str,
        source_agency: str,
    ) -> Path:
        """Save cleaned price dataset to processed storage as CSV."""
        if df.empty:
            raise ValueError("Cannot save empty DataFrame to processed storage.")

        out_path = self.processed_output_dir / f"{filename_prefix}_{source_agency.lower()}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(df)} cleaned price records to {out_path}")
        return out_path
