"""Unit tests for PriceCleaner transformer.

Validates crop name standardization, agricultural season calculation,
market identity preservation, district enrichment, and missing value preservation.
"""

from __future__ import annotations

import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from pipeline.transformers.price_cleaner import PriceCleaner


class TestPriceCleaner(unittest.TestCase):
    """Test suite for PriceCleaner."""

    def setUp(self) -> None:
        self.cleaner = PriceCleaner()

    def test_determine_season(self) -> None:
        """Verify Sri Lankan agricultural season assignment (Maha / Yala)."""
        # Yala: April - September
        s_type, s_year = self.cleaner.determine_season(datetime(2026, 4, 15))
        self.assertEqual(s_type, "Yala")
        self.assertEqual(s_year, 2026)

        s_type, s_year = self.cleaner.determine_season(datetime(2026, 8, 31))
        self.assertEqual(s_type, "Yala")
        self.assertEqual(s_year, 2026)

        # Maha: October - March
        # Q4 (Oct-Dec) has same season_year
        s_type, s_year = self.cleaner.determine_season(datetime(2026, 10, 14))
        self.assertEqual(s_type, "Maha")
        self.assertEqual(s_year, 2026)

        s_type, s_year = self.cleaner.determine_season(datetime(2026, 12, 31))
        self.assertEqual(s_type, "Maha")
        self.assertEqual(s_year, 2026)

        # Q1 (Jan-Mar) belongs to previous year's Maha
        s_type, s_year = self.cleaner.determine_season(datetime(2026, 1, 10))
        self.assertEqual(s_type, "Maha")
        self.assertEqual(s_year, 2025)

        s_type, s_year = self.cleaner.determine_season(datetime(2026, 3, 20))
        self.assertEqual(s_type, "Maha")
        self.assertEqual(s_year, 2025)

    def test_crop_name_normalization_all_target_crops(self) -> None:
        """Verify normalization for all 5 target crops: Carrot, Cabbage, Leek, Big onion, Tomato."""
        sample_records = [
            {"price_date": "2026-10-14", "crop_name": "carrot", "market_name": "Pettah", "price_type": "wholesale", "price_lkr_per_kg": 200.0, "source_agency": "TEST", "provenance_sha256": "a" * 64},
            {"price_date": "2026-10-14", "crop_name": "cabbages", "market_name": "Pettah", "price_type": "wholesale", "price_lkr_per_kg": 300.0, "source_agency": "TEST", "provenance_sha256": "a" * 64},
            {"price_date": "2026-10-14", "crop_name": "leek", "market_name": "Keppetipola", "price_type": "wholesale", "price_lkr_per_kg": 250.0, "source_agency": "TEST", "provenance_sha256": "a" * 64},
            {"price_date": "2026-10-14", "crop_name": "big onion (local)", "market_name": "Dambulla", "price_type": "wholesale", "price_lkr_per_kg": 270.0, "source_agency": "TEST", "provenance_sha256": "a" * 64},
            {"price_date": "2026-10-14", "crop_name": "tomato", "market_name": "Pettah", "price_type": "retail", "price_lkr_per_kg": 600.0, "source_agency": "TEST", "provenance_sha256": "a" * 64},
        ]
        df = self.cleaner.clean_records(sample_records, filter_target_crops=True)
        self.assertEqual(len(df), 5)
        crops = set(df["crop_name"])
        self.assertEqual(crops, {"Carrot", "Cabbage", "Leek", "Big onion", "Tomato"})

        # Verify original crop names preserved
        orig_crops = set(df["source_commodity_name"])
        self.assertIn("cabbages", orig_crops)
        self.assertIn("big onion (local)", orig_crops)

    def test_market_enrichment(self) -> None:
        """Verify district and geographic coordinates are enriched from market registry."""
        sample_records = [
            {"price_date": "2026-10-14", "crop_name": "Carrot", "market_name": "Pettah", "price_type": "wholesale", "price_lkr_per_kg": 200.0, "source_agency": "TEST", "provenance_sha256": "a" * 64},
            {"price_date": "2026-10-14", "crop_name": "Carrot", "market_name": "Dambulla", "price_type": "wholesale", "price_lkr_per_kg": 220.0, "source_agency": "TEST", "provenance_sha256": "a" * 64},
        ]
        df = self.cleaner.clean_records(sample_records)
        pettah = df[df["market_name"] == "Pettah"].iloc[0]
        self.assertEqual(pettah["district"], "Colombo")
        self.assertAlmostEqual(pettah["latitude"], 6.9365, places=3)
        self.assertAlmostEqual(pettah["longitude"], 79.8528, places=3)

        dambulla = df[df["market_name"] == "Dambulla"].iloc[0]
        self.assertEqual(dambulla["district"], "Matale")
        self.assertAlmostEqual(dambulla["latitude"], 7.8606, places=3)

    def test_missing_values_preserved_as_nan(self) -> None:
        """Verify that missing prices remain NaN and are NEVER zero-filled or fabricated."""
        sample_records = [
            {"price_date": "2026-10-14", "crop_name": "Carrot", "market_name": "Pettah", "price_type": "wholesale", "price_lkr_per_kg": None, "source_agency": "TEST", "provenance_sha256": "a" * 64},
            {"price_date": "2026-10-14", "crop_name": "Cabbage", "market_name": "Pettah", "price_type": "retail", "price_lkr_per_kg": np.nan, "source_agency": "TEST", "provenance_sha256": "a" * 64},
        ]
        df = self.cleaner.clean_records(sample_records)
        self.assertEqual(len(df), 2)
        self.assertTrue(df["price_lkr_per_kg"].isna().all())
        # Ensure none were converted to 0.0
        self.assertFalse((df["price_lkr_per_kg"] == 0.0).any())

    def test_target_crop_filter(self) -> None:
        """Verify filter_target_crops excludes non-target commodities like Beans or Pumpkin."""
        sample_records = [
            {"price_date": "2026-10-14", "crop_name": "Carrot", "market_name": "Pettah", "price_type": "wholesale", "price_lkr_per_kg": 200.0, "source_agency": "TEST", "provenance_sha256": "a" * 64},
            {"price_date": "2026-10-14", "crop_name": "Beans", "market_name": "Pettah", "price_type": "wholesale", "price_lkr_per_kg": 500.0, "source_agency": "TEST", "provenance_sha256": "a" * 64},
            {"price_date": "2026-10-14", "crop_name": "Pumpkin", "market_name": "Pettah", "price_type": "wholesale", "price_lkr_per_kg": 150.0, "source_agency": "TEST", "provenance_sha256": "a" * 64},
        ]
        # Filter enabled
        df_filtered = self.cleaner.clean_records(sample_records, filter_target_crops=True)
        self.assertEqual(len(df_filtered), 1)
        self.assertEqual(df_filtered["crop_name"].iloc[0], "Carrot")

        # Filter disabled
        df_all = self.cleaner.clean_records(sample_records, filter_target_crops=False)
        self.assertEqual(len(df_all), 3)

    def test_date_delimiter_robustness(self) -> None:
        """Verify PriceCleaner handles slash and dot delimited dates safely."""
        sample_records = [
            {"price_date": "2026/10/14", "crop_name": "Carrot", "market_name": "Pettah", "price_type": "wholesale", "price_lkr_per_kg": 200.0, "source_agency": "TEST", "provenance_sha256": "a" * 64},
            {"price_date": "2026.10.14", "crop_name": "Carrot", "market_name": "Dambulla", "price_type": "wholesale", "price_lkr_per_kg": 220.0, "source_agency": "TEST", "provenance_sha256": "a" * 64},
        ]
        df = self.cleaner.clean_records(sample_records)
        self.assertEqual(len(df), 2)
        self.assertEqual(df["obs_date"].tolist(), ["2026-10-14", "2026-10-14"])

    def test_empty_records(self) -> None:
        """Verify handling of empty record list."""
        df = self.cleaner.clean_records([])
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
