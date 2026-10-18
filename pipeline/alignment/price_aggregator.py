"""
Price aggregation engine for the weekly temporal alignment pipeline.

Aggregates daily market price observations into Monday-to-Sunday ISO weekly features:
calculating price statistics (mean, min, max, modal), tracking trading observation counts,
preserving individual market identities, and isolating Big Onion origin (Local vs. Imported).
"""

from typing import Optional
import pandas as pd
import numpy as np

from pipeline.alignment.schemas import CLEAN_PRICE_REQUIRED_COLUMNS, validate_dataframe
from pipeline.alignment.calendar_utils import add_iso_week_columns
from pipeline.alignment.crop_vocab import CropVocabularyEnforcer


class PriceAggregator:
    """
    Transforms daily market-level price observations into weekly aggregated features.
    """

    def __init__(self, decimals: int = 2, quarantine_sink_path: Optional[str] = None):
        """
        Initialize the price aggregator.

        Args:
            decimals: Number of decimal places to round price metrics to.
            quarantine_sink_path: Optional path to append quarantined ambiguous records.
        """
        self.decimals = decimals
        self.vocab_enforcer = CropVocabularyEnforcer(quarantine_sink_path=quarantine_sink_path)

    def aggregate(self, daily_price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily price records to weekly features per market, crop, and ISO week.

        Aggregation Rules:
        - Filters strictly to the 5 target crops (Carrot, Cabbage, Leek, Big Onion, Tomato).
        - Isolates Big Onion origin variety (Local vs. Imported).
        - Quarantines ambiguous 'Onion' records to prevent bias.
        - Preserves individual market identities (market_name).
        - Computes weekly_price_mean, weekly_price_min, weekly_price_max, weekly_price_modal.
        - Records price_obs_count (number of observed trading days in the week).
        - NEVER fabricates or forward-fills unobserved trading days.

        Args:
            daily_price_df: Input DataFrame containing daily price records.

        Returns:
            Aggregated pandas DataFrame with weekly price features.
        """
        validate_dataframe(daily_price_df, CLEAN_PRICE_REQUIRED_COLUMNS, "CleanPrice")

        # Step 1: Filter and standardize crops using the vocabulary enforcer
        standardized_df = self.vocab_enforcer.filter_and_standardize(
            daily_price_df,
            crop_col="crop_name",
            commodity_col="source_commodity_name" if "source_commodity_name" in daily_price_df.columns else None,
        )

        if standardized_df.empty:
            return pd.DataFrame()

        # Step 2: Add ISO 8601 calendar parameters
        df_with_weeks = add_iso_week_columns(standardized_df, date_col="obs_date")

        # Step 3: Define grouping keys preserving market identities and Big Onion varieties
        group_cols = [
            "district",
            "market_name",
            "crop_name",
            "origin_variety",
            "price_type",
            "iso_year",
            "iso_week",
            "week_start_date",
            "week_end_date",
        ]

        grouped = df_with_weeks.groupby(group_cols, as_index=False, dropna=False)

        rows = []
        for group_keys, group_df in grouped:
            (
                district,
                market_name,
                crop_name,
                origin_variety,
                price_type,
                iso_year,
                iso_week,
                week_start,
                week_end,
            ) = group_keys

            # Valid price values within the group
            valid_prices = group_df["price_lkr_per_kg"].dropna().astype(float)
            if valid_prices.empty:
                continue

            # Compute min price
            if "min_price_lkr_per_kg" in group_df.columns and group_df["min_price_lkr_per_kg"].dropna().any():
                min_p = float(group_df["min_price_lkr_per_kg"].dropna().min())
            else:
                min_p = float(valid_prices.min())

            # Compute max price
            if "max_price_lkr_per_kg" in group_df.columns and group_df["max_price_lkr_per_kg"].dropna().any():
                max_p = float(group_df["max_price_lkr_per_kg"].dropna().max())
            else:
                max_p = float(valid_prices.max())

            # Compute modal price
            if "modal_price_lkr_per_kg" in group_df.columns and group_df["modal_price_lkr_per_kg"].dropna().any():
                modal_p = float(group_df["modal_price_lkr_per_kg"].dropna().mean())
            else:
                modal_p = float(valid_prices.median())

            mean_p = float(valid_prices.mean())

            row = {
                "district": district,
                "market_name": market_name,
                "crop_name": crop_name,
                "origin_variety": origin_variety,
                "price_type": price_type,
                "iso_year": iso_year,
                "iso_week": iso_week,
                "week_start_date": week_start,
                "week_end_date": week_end,
                "weekly_price_mean": round(mean_p, self.decimals),
                "weekly_price_min": round(min_p, self.decimals),
                "weekly_price_max": round(max_p, self.decimals),
                "weekly_price_modal": round(modal_p, self.decimals),
                "price_obs_count": int(len(valid_prices)),
                "unit": "Rs./kg",
            }

            # Retain source agency if consistent
            if "source_agency" in group_df.columns:
                agencies = group_df["source_agency"].dropna().unique()
                row["source_agency"] = ", ".join(agencies) if len(agencies) > 0 else "Unknown"

            rows.append(row)

        weekly_price_df = pd.DataFrame(rows)
        if not weekly_price_df.empty:
            weekly_price_df = weekly_price_df.sort_values(
                by=["district", "market_name", "crop_name", "iso_year", "iso_week"]
            ).reset_index(drop=True)

        return weekly_price_df
