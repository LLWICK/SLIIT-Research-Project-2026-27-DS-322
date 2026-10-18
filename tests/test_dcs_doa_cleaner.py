"""
Unit tests for DCS and DOA Data Cleaner.
SLIIT Research Project DS-322 (2026/27) - Member 1
"""

import unittest

from pipeline.extractors.dcs_extractor import DCSExtractionRecord
from pipeline.extractors.doa_extractor import DOACultivationRecord
from pipeline.transformers.dcs_doa_cleaner import DCSDOACleaner


class TestDCSDOACleaner(unittest.TestCase):
    """Test suite for DCSDOACleaner normalizations and missing value handling."""

    def setUp(self):
        self.cleaner = DCSDOACleaner()

    def test_crop_normalization(self):
        """Verify standard Sri Lankan vegetable crop name mapping."""
        self.assertEqual(self.cleaner.normalize_crop("potato"), "Potato")
        self.assertEqual(self.cleaner.normalize_crop("Potato (solanum tuberosum)"), "Potato")
        self.assertEqual(self.cleaner.normalize_crop("Big Onion"), "Big Onion")
        self.assertEqual(self.cleaner.normalize_crop("green chili"), "Green Chilli")
        self.assertEqual(self.cleaner.normalize_crop("manioc"), "Cassava")

    def test_district_normalization(self):
        """Verify district name trimming, casing, and alias normalization."""
        self.assertEqual(self.cleaner.normalize_district("Nuwara Eliya "), "Nuwara Eliya")
        self.assertEqual(self.cleaner.normalize_district("nuwaraeliya"), "Nuwara Eliya")
        self.assertEqual(self.cleaner.normalize_district("Badulla"), "Badulla")
        self.assertEqual(self.cleaner.normalize_district("Anuradhapura 1"), "Anuradhapura")
        self.assertEqual(self.cleaner.normalize_district("Other District"), "Other Districts")
        self.assertEqual(self.cleaner.normalize_district("National Total"), "National Total")

    def test_unit_normalization(self):
        """Verify unit string standardization."""
        self.assertEqual(self.cleaner.normalize_unit("Unit : MT ."), "metric_ton")
        self.assertEqual(self.cleaner.normalize_unit("Metric Tons"), "metric_ton")
        self.assertEqual(self.cleaner.normalize_unit("Unit : Ha."), "hectare")
        self.assertEqual(self.cleaner.normalize_unit("Hectares"), "hectare")
        self.assertEqual(self.cleaner.normalize_unit("Kg/Ha"), "kg_per_hectare")

    def test_parse_value_valid_and_estimated(self):
        """Verify numeric parsing, commas stripping, and asterisk footnote tracking."""
        val, is_est, status = self.cleaner.parse_value("15,446")
        self.assertEqual(val, 15446.0)
        self.assertFalse(is_est)
        self.assertEqual(status, "valid")

        val_est, is_est, status = self.cleaner.parse_value("1,250*")
        self.assertEqual(val_est, 1250.0)
        self.assertTrue(is_est)
        self.assertEqual(status, "valid")

        val_dec, is_est, status = self.cleaner.parse_value("15840.3")
        self.assertEqual(val_dec, 15840.3)
        self.assertFalse(is_est)
        self.assertEqual(status, "valid")

    def test_parse_value_nil_and_missing(self):
        """Verify explicit distinction between explicit nil/zero and missing/unparseable values."""
        # Nil (dash or word nil)
        val_dash, is_est, status = self.cleaner.parse_value("-")
        self.assertEqual(val_dash, 0.0)
        self.assertFalse(is_est)
        self.assertEqual(status, "nil")

        # Missing (empty string or N/A)
        val_empty, is_est, status = self.cleaner.parse_value("")
        self.assertIsNone(val_empty)
        self.assertEqual(status, "missing")

        val_na, is_est, status = self.cleaner.parse_value("N/A")
        self.assertIsNone(val_na)
        self.assertEqual(status, "missing")

    def test_parse_time_period(self):
        """Verify annual, partial year, split year, and season time period handling."""
        p_annual = self.cleaner.parse_time_period("2023")
        self.assertEqual(p_annual["period_type"], "annual")
        self.assertEqual(p_annual["year"], 2023)
        self.assertEqual(p_annual["period_start"], "2023-01-01")
        self.assertEqual(p_annual["period_end"], "2023-12-31")

        p_partial = self.cleaner.parse_time_period("2018 upto May")
        self.assertEqual(p_partial["period_type"], "partial_year")
        self.assertEqual(p_partial["year"], 2018)
        self.assertEqual(p_partial["period_start"], "2018-01-01")
        self.assertEqual(p_partial["period_end"], "2018-05-31")

        p_split = self.cleaner.parse_time_period("2006/2007")
        self.assertEqual(p_split["period_type"], "crop_year")
        self.assertEqual(p_split["year"], 2007)
        self.assertEqual(p_split["period_start"], "2006-10-01")
        self.assertEqual(p_split["period_end"], "2007-09-30")

        p_season = self.cleaner.parse_time_period("Yala 2024")
        self.assertEqual(p_season["period_type"], "season")
        self.assertEqual(p_season["season"], "Yala")
        self.assertEqual(p_season["year"], 2024)
        self.assertEqual(p_season["period_start"], "2024-05-01")
        self.assertEqual(p_season["period_end"], "2024-08-31")

    def test_clean_dcs_record_end_to_end(self):
        """Verify full cleaning of a raw DCSExtractionRecord."""
        raw = DCSExtractionRecord(
            crop="Potato",
            metric="production",
            raw_metric="CO3 – 1: Potato Production",
            raw_unit="Unit : MT .",
            district="Nuwara Eliya ",
            period_label="2007",
            raw_value="15,446",
            table_id="CO3 - 1",
            table_title="Potato Production",
            source_url="https://www.statistics.gov.lk/Agriculture/StaticalInformation/Highlandcrops/Potato",
        )

        cleaned = self.cleaner.clean_dcs_record(raw)
        self.assertEqual(cleaned["source_name"], "DCS")
        self.assertEqual(cleaned["crop"], "Potato")
        self.assertEqual(cleaned["district"], "Nuwara Eliya")
        self.assertEqual(cleaned["metric"], "production")
        self.assertEqual(cleaned["unit"], "metric_ton")
        self.assertEqual(cleaned["value"], 15446.0)
        self.assertFalse(cleaned["is_estimated"])
        self.assertEqual(cleaned["raw_status"], "valid")
        self.assertEqual(cleaned["period_type"], "annual")
        self.assertEqual(cleaned["year"], 2007)
        self.assertTrue(cleaned["provenance_hash"])

    def test_clean_doa_cultivation_record_end_to_end(self):
        """Verify full cleaning of a raw DOACultivationRecord with achievement percentage calculation."""
        raw = DOACultivationRecord(
            crop="Big Onion",
            district="Anuradhapura",
            season="Yala",
            year=2024,
            target_extent_ha=2500.0,
            cultivated_extent_ha=2200.0,
            expected_production_mt=33000.0,
            raw_source="DOA_SEPC_CROP_FORECAST_YALA_2024",
        )

        cleaned = self.cleaner.clean_doa_cultivation_record(raw)
        self.assertEqual(cleaned["source_name"], "DOA_SEPC")
        self.assertEqual(cleaned["crop"], "Big Onion")
        self.assertEqual(cleaned["district"], "Anuradhapura")
        self.assertEqual(cleaned["season"], "Yala")
        self.assertEqual(cleaned["year"], 2024)
        self.assertEqual(cleaned["achievement_pct"], 88.0)
        self.assertEqual(cleaned["period_start"], "2024-05-01")
        self.assertEqual(cleaned["period_end"], "2024-08-31")
        self.assertTrue(cleaned["provenance_hash"])


if __name__ == "__main__":
    unittest.main()
