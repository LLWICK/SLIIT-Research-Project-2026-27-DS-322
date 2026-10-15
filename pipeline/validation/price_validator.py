"""Data Quality and Integrity Validator for Agricultural Commodity Prices.

Validates schema conformity, physical economic price bounds, wholesale vs. retail margin logic,
absence of duplicates, and reports completeness metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("validation.price")

# Standard acceptable domain bounds for vegetable prices in Sri Lanka (LKR/kg)
# Lowest wholesale glut ~10-20 LKR/kg, highest retail peak ~3500-5000 LKR/kg during extreme off-season
PRICE_BOUNDS = (5.0, 5000.0)


class PriceValidator:
    """Validates cleaned price datasets against schema, economic sanity rules, and boundary constraints."""

    def __init__(self, price_bounds: tuple[float, float] = PRICE_BOUNDS) -> None:
        self.min_price_bound, self.max_price_bound = price_bounds

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run comprehensive validation checks on cleaned price DataFrame.

        Returns:
            Validation report dictionary with status, errors, warnings, and completeness statistics.
        """
        if df.empty:
            return {
                "is_valid": False,
                "total_rows": 0,
                "errors": ["Price dataset is empty."],
                "warnings": [],
                "completeness": {},
            }

        errors: list[str] = []
        warnings: list[str] = []

        # 1. Schema check: required columns
        required_cols = [
            "obs_date",
            "crop_name",
            "market_name",
            "price_type",
            "unit",
            "source_agency",
            "provenance_sha256",
        ]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing mandatory columns: {missing_cols}")

        # 2. Date format check
        parsed_dates = pd.to_datetime(df["obs_date"], errors="coerce")
        if parsed_dates.isna().any():
            errors.append("Dataset contains unparseable dates in 'obs_date'.")

        # 3. Duplicate observations check
        # Include source_commodity_name (or original_crop_name) to distinguish legitimate crop varieties
        # (e.g. Local vs Imported Big Onion) while catching true duplicate entries.
        dup_cols = ["obs_date", "crop_name", "market_name", "price_type"]
        variety_col = next((c for c in ["source_commodity_name", "original_crop_name", "variety"] if c in df.columns), None)
        if variety_col:
            dup_cols.insert(2, variety_col)
        existing_dup_cols = [c for c in dup_cols if c in df.columns]
        if len(existing_dup_cols) >= 4:
            dups_count = int(df.duplicated(subset=existing_dup_cols).sum())
            if dups_count > 0:
                errors.append(f"Found {dups_count} duplicate price observations on {existing_dup_cols}.")

        # 4. Domain price bounds check
        if "price_lkr_per_kg" in df.columns:
            valid_prices = df["price_lkr_per_kg"].dropna()
            total_recs = len(df)
            non_null_recs = len(valid_prices)
            missing_recs = total_recs - non_null_recs
            completeness_pct = round((non_null_recs / total_recs) * 100.0, 2) if total_recs > 0 else 0.0

            # Negative or zero prices are invalid
            non_positive = int((valid_prices <= 0).sum())
            if non_positive > 0:
                errors.append(f"Found {non_positive} non-positive prices (<= 0 LKR/kg).")

            # Out of bounds
            out_of_bounds = int(
                ((valid_prices < self.min_price_bound) | (valid_prices > self.max_price_bound)).sum()
            )
            if out_of_bounds > 0:
                warnings.append(
                    f"Found {out_of_bounds} prices outside standard economic bounds "
                    f"[{self.min_price_bound}, {self.max_price_bound}] LKR/kg."
                )

        # 5. Min / Max / Modal consistency check
        if "min_price_lkr_per_kg" in df.columns and "max_price_lkr_per_kg" in df.columns:
            both_present = df.dropna(subset=["min_price_lkr_per_kg", "max_price_lkr_per_kg"])
            inverted_bounds = int((both_present["min_price_lkr_per_kg"] > both_present["max_price_lkr_per_kg"]).sum())
            if inverted_bounds > 0:
                errors.append(f"Found {inverted_bounds} rows where min_price > max_price.")

        # 6. Economic sanity check: Retail vs Wholesale margin consistency
        # For the same date, crop, and market: retail price is expected to be >= wholesale price
        if {"obs_date", "crop_name", "market_name", "price_type", "price_lkr_per_kg"}.issubset(df.columns):
            pivoted = df.pivot_table(
                index=["obs_date", "crop_name", "market_name"],
                columns="price_type",
                values="price_lkr_per_kg",
                aggfunc="mean",
            ).reset_index()

            if "retail" in pivoted.columns and "wholesale" in pivoted.columns:
                both_types = pivoted.dropna(subset=["retail", "wholesale"])
                margin_inversions = both_types[both_types["retail"] < both_types["wholesale"]]
                if len(margin_inversions) > 0:
                    warnings.append(
                        f"Detected {len(margin_inversions)} instances where retail price was less than wholesale price "
                        "(possible supply disruption, subsidy, or anomaly)."
                    )

        # 7. Aggregate completeness and crop summaries
        crop_metrics = {}
        if "crop_name" in df.columns and "price_lkr_per_kg" in df.columns:
            for crop, grp in df.groupby("crop_name"):
                p_valid = grp["price_lkr_per_kg"].dropna()
                crop_metrics[str(crop)] = {
                    "total_records": len(grp),
                    "valid_prices": len(p_valid),
                    "missing_prices": int(grp["price_lkr_per_kg"].isna().sum()),
                    "completeness_pct": round((len(p_valid) / len(grp)) * 100.0, 2) if len(grp) else 0.0,
                    "min_price": float(p_valid.min()) if len(p_valid) else None,
                    "max_price": float(p_valid.max()) if len(p_valid) else None,
                    "mean_price": round(float(p_valid.mean()), 2) if len(p_valid) else None,
                }

        is_valid = len(errors) == 0

        report = {
            "is_valid": is_valid,
            "total_rows": len(df),
            "date_range": {
                "start": str(df["obs_date"].min()) if "obs_date" in df.columns and len(df) else None,
                "end": str(df["obs_date"].max()) if "obs_date" in df.columns and len(df) else None,
            },
            "errors": errors,
            "warnings": warnings,
            "crop_summaries": crop_metrics,
        }

        if is_valid:
            logger.info(f"Price validation PASSED ({len(df)} records). Warnings: {len(warnings)}")
        else:
            logger.error(f"Price validation FAILED with {len(errors)} errors: {errors}")

        return report
