"""Unit tests for CBSL Daily Price Report Extractor.

Tests execute purely offline using a captured official CBSL price report fixture,
ensuring determinism, exact numeric value verification, and zero network dependency.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from pipeline.extractors.price_cbsl import CbslPriceExtractor

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_PDF_PATH = FIXTURES_DIR / "sample_cbsl_report.pdf"


class TestCbslPriceExtractor(unittest.TestCase):
    """Test suite for CbslPriceExtractor."""

    def setUp(self) -> None:
        self.extractor = CbslPriceExtractor()

    def test_date_formatting(self) -> None:
        """Verify date string conversion to ISO and CBSL token."""
        iso_date, cbsl_token = self.extractor.format_date_str("2026-10-14")
        self.assertEqual(iso_date, "2026-10-14")
        self.assertEqual(cbsl_token, "20261014")

        iso_date2, cbsl_token2 = self.extractor.format_date_str("20261014")
        self.assertEqual(iso_date2, "2026-10-14")
        self.assertEqual(cbsl_token2, "20261014")

    def test_build_pdf_url(self) -> None:
        """Verify official CBSL PDF URL pattern generation."""
        url = self.extractor.build_pdf_url("20261014")
        self.assertEqual(
            url,
            "https://www.cbsl.gov.lk/sites/default/files/cbslweb_documents/statistics/pricerpt/price_report_20261014_e.pdf",
        )

    def test_clean_numeric_price(self) -> None:
        """Verify price parsing and handling of missing values without fabrication."""
        self.assertEqual(self.extractor.clean_numeric_price("220.00"), 220.0)
        self.assertEqual(self.extractor.clean_numeric_price("1,150.00"), 1150.0)
        self.assertIsNone(self.extractor.clean_numeric_price("n.a."))
        self.assertIsNone(self.extractor.clean_numeric_price("NA"))
        self.assertIsNone(self.extractor.clean_numeric_price("-"))
        self.assertIsNone(self.extractor.clean_numeric_price(""))
        self.assertIsNone(self.extractor.clean_numeric_price("invalid_str"))

    def test_extract_missing_file_raises_error(self) -> None:
        """Verify that attempting to extract from a nonexistent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_from_file("nonexistent_report_20261014.pdf")

    @unittest.skipUnless(SAMPLE_PDF_PATH.is_file(), "Offline sample PDF fixture required")
    def test_extract_from_fixture_file(self) -> None:
        """Verify full offline extraction from captured genuine CBSL price report."""
        records = self.extractor.extract_from_file(SAMPLE_PDF_PATH)
        self.assertGreater(len(records), 0)

        # Confirm target crops are identified
        target_records = [r for r in records if r["is_target_crop"]]
        self.assertGreater(len(target_records), 0)

        crops_extracted = {r["crop_name"] for r in target_records}
        self.assertIn("Carrot", crops_extracted)
        self.assertIn("Cabbage", crops_extracted)
        self.assertIn("Tomato", crops_extracted)
        self.assertIn("Big onion", crops_extracted)

        # Confirm expected CBSL limitation: Leek is absent in CBSL reports
        self.assertNotIn("Leek", crops_extracted)

        # Check provenance and fields
        first_rec = target_records[0]
        self.assertEqual(first_rec["price_date"], "2026-10-14")
        self.assertEqual(first_rec["source_agency"], "CBSL")
        self.assertEqual(first_rec["unit"], "Rs./kg")
        self.assertEqual(len(first_rec["provenance_sha256"]), 64)

        # Verify markets present in CBSL report
        markets = {r["market_name"] for r in target_records}
        self.assertIn("Pettah", markets)
        self.assertIn("Dambulla", markets)
        self.assertIn("Narahenpita", markets)

        # Verify wholesale and retail distinctions
        price_types = {r["price_type"] for r in target_records}
        self.assertEqual(price_types, {"wholesale", "retail"})

        # Exact value verification against genuine CBSL published report
        # Helper to lookup price
        def get_price(crop: str, market: str, ptype: str) -> float | None:
            matches = [
                r["price_lkr_per_kg"]
                for r in target_records
                if r["crop_name"] == crop and r["market_name"] == market and r["price_type"] == ptype
            ]
            return matches[0] if matches else None

        # Carrot
        self.assertEqual(get_price("Carrot", "Pettah", "wholesale"), 220.0)
        self.assertEqual(get_price("Carrot", "Dambulla", "wholesale"), 225.0)
        self.assertEqual(get_price("Carrot", "Pettah", "retail"), 270.0)
        self.assertEqual(get_price("Carrot", "Dambulla", "retail"), 255.0)
        self.assertEqual(get_price("Carrot", "Narahenpita", "retail"), 320.0)

        # Cabbage
        self.assertEqual(get_price("Cabbage", "Pettah", "wholesale"), 400.0)
        self.assertEqual(get_price("Cabbage", "Dambulla", "wholesale"), 315.0)
        self.assertEqual(get_price("Cabbage", "Pettah", "retail"), 450.0)
        self.assertEqual(get_price("Cabbage", "Dambulla", "retail"), 345.0)
        self.assertEqual(get_price("Cabbage", "Narahenpita", "retail"), 400.0)

        # Tomato
        self.assertEqual(get_price("Tomato", "Pettah", "wholesale"), 600.0)
        self.assertEqual(get_price("Tomato", "Dambulla", "wholesale"), 475.0)
        self.assertEqual(get_price("Tomato", "Pettah", "retail"), 650.0)
        self.assertEqual(get_price("Tomato", "Dambulla", "retail"), 505.0)
        self.assertEqual(get_price("Tomato", "Narahenpita", "retail"), 800.0)

        # Big Onion (Local)
        self.assertEqual(get_price("Big onion", "Pettah", "wholesale"), 270.0)
        self.assertEqual(get_price("Big onion", "Dambulla", "wholesale"), 245.0)
        self.assertEqual(get_price("Big onion", "Pettah", "retail"), 290.0)
        self.assertEqual(get_price("Big onion", "Dambulla", "retail"), 265.0)
        self.assertEqual(get_price("Big onion", "Narahenpita", "retail"), 380.0)

        # Big Onion (Imported) Narahenpita retail is 'n.a.' in report -> preserved as None
        imp_narahenpita = [
            r["price_lkr_per_kg"]
            for r in target_records
            if r["crop_name"] == "Big onion (Imported)" and r["market_name"] == "Narahenpita"
        ]
        if imp_narahenpita:
            self.assertIsNone(imp_narahenpita[0])


if __name__ == "__main__":
    unittest.main()
