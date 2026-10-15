"""Unit tests for PriceValidator.

Validates schema adherence, economic domain bounds, retail vs. wholesale margin logic,
duplicate detection, and data completeness metrics.
"""

from __future__ import annotations

import unittest
import pandas as pd

from pipeline.validation.price_validator import PriceValidator


class TestPriceValidator(unittest.TestCase):
    """Test suite for PriceValidator."""

    def setUp(self) -> None:
        self.validator = PriceValidator()

    def make_valid_df(self) -> pd.DataFrame:
        """Helper to create a minimal valid price DataFrame."""
        return pd.DataFrame([
            {
                "obs_date": "2026-10-14",
                "crop_name": "Carrot",
                "market_name": "Pettah",
                "price_type": "wholesale",
                "price_lkr_per_kg": 220.0,
                "min_price_lkr_per_kg": 200.0,
                "max_price_lkr_per_kg": 240.0,
                "modal_price_lkr_per_kg": 220.0,
                "unit": "Rs./kg",
                "source_agency": "CBSL",
                "provenance_sha256": "a" * 64,
            },
            {
                "obs_date": "2026-10-14",
                "crop_name": "Carrot",
                "market_name": "Pettah",
                "price_type": "retail",
                "price_lkr_per_kg": 270.0,
                "min_price_lkr_per_kg": 250.0,
                "max_price_lkr_per_kg": 290.0,
                "modal_price_lkr_per_kg": 270.0,
                "unit": "Rs./kg",
                "source_agency": "CBSL",
                "provenance_sha256": "a" * 64,
            },
        ])

    def test_valid_data_passes(self) -> None:
        """Verify that a compliant price DataFrame passes validation with zero errors."""
        df = self.make_valid_df()
        report = self.validator.validate(df)
        self.assertTrue(report["is_valid"])
        self.assertEqual(len(report["errors"]), 0)
        self.assertEqual(report["total_rows"], 2)
        self.assertIn("Carrot", report["crop_summaries"])

    def test_missing_mandatory_columns(self) -> None:
        """Verify detection of missing required schema columns."""
        df = self.make_valid_df().drop(columns=["crop_name", "provenance_sha256"])
        report = self.validator.validate(df)
        self.assertFalse(report["is_valid"])
        self.assertTrue(any("Missing mandatory columns" in e for e in report["errors"]))

    def test_non_positive_price_detection(self) -> None:
        """Verify detection of invalid prices (<= 0 LKR/kg)."""
        df = self.make_valid_df()
        df.loc[0, "price_lkr_per_kg"] = -50.0
        report = self.validator.validate(df)
        self.assertFalse(report["is_valid"])
        self.assertTrue(any("non-positive" in e for e in report["errors"]))

    def test_inverted_min_max_detection(self) -> None:
        """Verify detection of min_price > max_price inconsistency."""
        df = self.make_valid_df()
        df.loc[0, "min_price_lkr_per_kg"] = 500.0
        df.loc[0, "max_price_lkr_per_kg"] = 200.0
        report = self.validator.validate(df)
        self.assertFalse(report["is_valid"])
        self.assertTrue(any("min_price > max_price" in e for e in report["errors"]))

    def test_retail_lower_than_wholesale_warning(self) -> None:
        """Verify warning when retail price is lower than wholesale price for same market/crop."""
        df = self.make_valid_df()
        # Set retail (200.0) lower than wholesale (250.0)
        df.loc[0, "price_lkr_per_kg"] = 250.0  # wholesale
        df.loc[1, "price_lkr_per_kg"] = 200.0  # retail
        report = self.validator.validate(df)
        # Should be valid (economic distortion or data quirk) but issue a warning
        self.assertTrue(report["is_valid"])
        self.assertTrue(any("retail price was less than wholesale" in w for w in report["warnings"]))

    def test_duplicate_observations_detection(self) -> None:
        """Verify detection of duplicate (date, crop, variety, market, price_type) records."""
        df = self.make_valid_df()
        # Duplicate row 0 exactly
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        report = self.validator.validate(df)
        self.assertFalse(report["is_valid"])
        self.assertTrue(any("duplicate price observations" in e for e in report["errors"]))

    def test_legitimate_varieties_not_treated_as_duplicates(self) -> None:
        """Verify legitimate varieties (e.g. Local vs Imported Big Onion) are NOT falsely flagged as duplicates."""
        df = pd.DataFrame([
            {
                "obs_date": "2026-10-14",
                "crop_name": "Big onion",
                "source_commodity_name": "Big Onion (Local)",
                "market_name": "Pettah",
                "price_type": "wholesale",
                "price_lkr_per_kg": 270.0,
                "unit": "Rs./kg",
                "source_agency": "CBSL",
                "provenance_sha256": "a" * 64,
            },
            {
                "obs_date": "2026-10-14",
                "crop_name": "Big onion",
                "source_commodity_name": "Big Onion (Imp)",
                "market_name": "Pettah",
                "price_type": "wholesale",
                "price_lkr_per_kg": 265.0,
                "unit": "Rs./kg",
                "source_agency": "CBSL",
                "provenance_sha256": "a" * 64,
            },
        ])
        report = self.validator.validate(df)
        self.assertTrue(report["is_valid"])
        self.assertEqual(len(report["errors"]), 0)

    def test_empty_dataframe(self) -> None:
        """Verify that empty DataFrame fails validation gracefully."""
        report = self.validator.validate(pd.DataFrame())
        self.assertFalse(report["is_valid"])
        self.assertTrue(any("empty" in e for e in report["errors"]))


if __name__ == "__main__":
    unittest.main()
