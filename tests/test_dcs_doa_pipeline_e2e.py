"""
End-to-End Pipeline Integration Test for DCS and DOA Agricultural Data.
SLIIT Research Project DS-322 (2026/27) - Member 1
"""

from pathlib import Path
import unittest

from pipeline.extractors.dcs_extractor import DCSExtractor
from pipeline.extractors.doa_extractor import DOAExtractor
from pipeline.transformers.dcs_doa_cleaner import DCSDOACleaner
from pipeline.validation.dcs_doa_validator import DCSDOAValidator


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestDCSDOAPipelineE2E(unittest.TestCase):
    """End-to-end ingestion, cleaning, and validation pipeline test."""

    def setUp(self):
        self.dcs_extractor = DCSExtractor()
        self.doa_extractor = DOAExtractor()
        self.cleaner = DCSDOACleaner()
        self.validator = DCSDOAValidator(strict_districts=True)

    def test_dcs_potato_e2e_pipeline(self):
        """Test full pipeline on DCS Potato HTML: extract -> clean -> validate."""
        raw_records = self.dcs_extractor.extract_from_file(
            FIXTURES_DIR / "dcs_potato_sample.html",
            source_url="https://www.statistics.gov.lk/Agriculture/StaticalInformation/Highlandcrops/Potato",
            crop_hint="Potato",
        )
        self.assertGreater(len(raw_records), 0)

        cleaned_records = [self.cleaner.clean_dcs_record(r) for r in raw_records]
        self.assertEqual(len(cleaned_records), len(raw_records))

        # Validate batch
        summary = self.validator.validate_batch(cleaned_records)
        self.assertEqual(summary["total_records"], len(cleaned_records))
        self.assertGreater(summary["valid_records"], 0)
        # Verify valid records pass rate
        self.assertGreaterEqual(summary["pass_rate_pct"], 90.0)

        # Inspect specific sample records
        sample = next(
            r for r in cleaned_records
            if r["district"] == "Badulla" and r["period_label"] == "2007" and r["metric"] == "production"
        )
        self.assertEqual(sample["value"], 59649.0)
        self.assertEqual(sample["unit"], "metric_ton")
        self.assertEqual(sample["crop"], "Potato")
        self.assertEqual(sample["source_name"], "DCS")
        self.assertIn("https://www.statistics.gov.lk", sample["source_url"])
        self.assertIsNotNone(sample["provenance_hash"])

    def test_doa_cultivation_e2e_pipeline(self):
        """Test full pipeline on DOA cultivation CSV: extract -> clean -> validate."""
        raw_records = self.doa_extractor.extract_cultivation_from_csv(
            FIXTURES_DIR / "doa_crop_forecast_sample.csv"
        )
        self.assertEqual(len(raw_records), 5)

        cleaned_records = [self.cleaner.clean_doa_cultivation_record(r) for r in raw_records]
        self.assertEqual(len(cleaned_records), 5)

        summary = self.validator.validate_batch(cleaned_records)
        self.assertEqual(summary["total_records"], 5)
        self.assertEqual(summary["valid_records"], 5)
        self.assertEqual(summary["pass_rate_pct"], 100.0)

        sample = cleaned_records[0]
        self.assertEqual(sample["crop"], "Big Onion")
        self.assertEqual(sample["district"], "Anuradhapura")
        self.assertEqual(sample["season"], "Yala")
        self.assertEqual(sample["achievement_pct"], 88.0)
        self.assertIsNotNone(sample["provenance_hash"])


if __name__ == "__main__":
    unittest.main()
