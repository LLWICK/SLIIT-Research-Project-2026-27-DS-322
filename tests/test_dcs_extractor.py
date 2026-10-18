"""
Unit tests for DCS Agricultural Statistics Extractor.
SLIIT Research Project DS-322 (2026/27) - Member 1
"""

from pathlib import Path
import unittest

from pipeline.extractors.dcs_extractor import DCSExtractor, DCSExtractionRecord


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestDCSExtractor(unittest.TestCase):
    """Test suite for DCSExtractor with real official DCS page fixtures."""

    def setUp(self):
        self.extractor = DCSExtractor()
        self.potato_fixture = FIXTURES_DIR / "dcs_potato_sample.html"
        self.big_onion_fixture = FIXTURES_DIR / "dcs_big_onion_sample.html"

    def test_extract_potato_production_and_extent(self):
        """Verify extraction from real DCS Potato page fixture (CO3-1 and CO3-2)."""
        records = self.extractor.extract_from_file(
            self.potato_fixture,
            source_url="https://www.statistics.gov.lk/Agriculture/StaticalInformation/Highlandcrops/Potato",
            crop_hint="Potato",
        )

        self.assertGreater(len(records), 0, "Should extract records from potato fixture")

        # Verify crop and metric detection
        crops = {r.crop for r in records}
        self.assertIn("Potato", crops)

        metrics = {r.metric for r in records}
        self.assertIn("production", metrics)
        self.assertIn("extent", metrics)

        # Verify table IDs
        table_ids = {r.table_id for r in records}
        self.assertTrue(any("CO3 - 1" in tid or "C031" in tid for tid in table_ids))
        self.assertTrue(any("CO3 - 2" in tid or "CO32" in tid for tid in table_ids))

        # Check district extraction
        districts = {r.district for r in records}
        self.assertIn("National Total", districts)
        self.assertIn("Nuwara Eliya", districts)
        self.assertIn("Badulla", districts)
        self.assertIn("Other Districts", districts)

        # Check raw values and provenance
        sample_prod = next(
            r for r in records
            if r.district == "Nuwara Eliya" and r.period_label == "2007" and r.metric == "production"
        )
        self.assertEqual(sample_prod.raw_value, "15,446")
        self.assertTrue(sample_prod.source_url.startswith("https://www.statistics.gov.lk"))
        self.assertIsNotNone(sample_prod.extraction_timestamp)

        # Check estimated asterisk preservation
        est_rec = next(
            (r for r in records if "*" in r.raw_value),
            None
        )
        self.assertIsNotNone(est_rec, "Should find record with asterisk footnote")

        # Check nil dash preservation
        dash_rec = next(
            (r for r in records if r.raw_value == "-"),
            None
        )
        self.assertIsNotNone(dash_rec, "Should find record with dash '-' indicating nil/zero")

    def test_extract_big_onion_production(self):
        """Verify extraction from real DCS Big Onion page fixture (Table F07-1)."""
        records = self.extractor.extract_from_file(
            self.big_onion_fixture,
            source_url="https://www.statistics.gov.lk/Agriculture/StaticalInformation/Highlandcrops/BigOnion",
            crop_hint="Big Onion",
        )

        self.assertGreater(len(records), 0)

        crops = {r.crop for r in records}
        self.assertIn("Big Onion", crops)

        metrics = {r.metric for r in records}
        self.assertIn("production", metrics)

        districts = {r.district for r in records}
        self.assertIn("Anuradhapura", districts)
        self.assertIn("Matale", districts)
        self.assertIn("Kurunegala", districts)

        # Check decimal values e.g. 15840.3 for 2022 National Total
        sample_nat = next(
            r for r in records
            if r.district == "National Total" and r.period_label == "2022"
        )
        self.assertEqual(sample_nat.raw_value, "15840.3")

    def test_file_not_found_handling(self):
        """Verify extractor raises FileNotFoundError on missing path."""
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_from_file("non_existent_dcs_file.html")


if __name__ == "__main__":
    unittest.main()
