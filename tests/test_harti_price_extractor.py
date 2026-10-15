"""Unit tests for HARTI Daily Food Price Bulletin Extractor.

Tests live-safe offline fallback bulletin ingestion, tabular record normalization,
preservation of Min/Max/Modal prices across research crops (including Leek),
and SHA-256 provenance generation.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.extractors.price_harti import HartiPriceExtractor


class TestHartiPriceExtractor(unittest.TestCase):
    """Test suite for HartiPriceExtractor."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.extractor = HartiPriceExtractor(raw_output_dir=Path(self.temp_dir.name))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_format_date_tokens(self) -> None:
        """Verify date string conversion to canonical HARTI tokens."""
        iso_date, harti_token, year, month = self.extractor.format_date_tokens("2024-12-31")
        self.assertEqual(iso_date, "2024-12-31")
        self.assertEqual(harti_token, "31-12-2024")
        self.assertEqual(year, "2024")
        self.assertEqual(month, "December")

        # Test dot-delimited format as used in recent 2026 government portals
        iso_date2, harti_token2, year2, month2 = self.extractor.format_date_tokens("2026.10.07")
        self.assertEqual(iso_date2, "2026-10-07")
        self.assertEqual(harti_token2, "07-10-2026")
        self.assertEqual(year2, "2026")
        self.assertEqual(month2, "October")

    def test_parse_date_from_filename(self) -> None:
        """Verify date extraction across verified historical and modern 2026 filename conventions."""
        # 1. Verified 2026 portal format: Vegetable Pricenew ex1(2026.10.07).pdf
        dt1 = self.extractor.parse_date_from_filename("Vegetable Pricenew ex1(2026.10.07).pdf")
        self.assertIsNotNone(dt1)
        self.assertEqual(dt1.strftime("%Y-%m-%d"), "2026-10-07")

        # 2. Alternative 2026 format: Vegetables Wholesale Prices (2026.10.04).pdf
        dt2 = self.extractor.parse_date_from_filename("Vegetables Wholesale Prices (2026.10.04).pdf")
        self.assertIsNotNone(dt2)
        self.assertEqual(dt2.strftime("%Y-%m-%d"), "2026-10-04")

        # 3. Canonical archive format: daily_31-12-2024.pdf
        dt3 = self.extractor.parse_date_from_filename("daily_31-12-2024.pdf")
        self.assertIsNotNone(dt3)
        self.assertEqual(dt3.strftime("%Y-%m-%d"), "2024-12-31")

        # 4. Unrecognized format returns None safely without exception
        dt_none = self.extractor.parse_date_from_filename("unrecognized_bulletin_title.pdf")
        self.assertIsNone(dt_none)

    def test_build_pdf_url(self) -> None:
        """Verify official HARTI URL pattern formatting."""
        url = self.extractor.build_pdf_url("2024", "December", "31-12-2024")
        self.assertEqual(
            url,
            "https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2024/December/daily_31-12-2024.pdf",
        )

    def test_ingest_offline_bulletin_manifest(self) -> None:
        """Verify offline fallback ingestion records raw bytes, SHA-256 hash, and provenance manifest."""
        # Create a mock captured bulletin file
        mock_file = Path(self.temp_dir.name) / "daily_31-12-2024.pdf"
        dummy_content = b"%PDF-1.4 Mock HARTI bulletin content for testing provenance"
        with open(mock_file, "wb") as f:
            f.write(dummy_content)

        payload, manifest = self.extractor.ingest_offline_bulletin(
            file_path=mock_file,
            report_date="2024-12-31",
            save_raw=True,
        )

        self.assertEqual(payload, dummy_content)
        self.assertEqual(manifest["source_name"], "HARTI")
        self.assertEqual(manifest["parameters"]["date"], "2024-12-31")
        self.assertEqual(len(manifest["raw_sha256"]), 64)
        self.assertEqual(manifest["payload_size_bytes"], len(dummy_content))
        self.assertTrue(manifest.get("offline_ingestion"))

    def test_ingest_missing_offline_bulletin_raises_error(self) -> None:
        """Verify FileNotFoundError when specified offline bulletin does not exist."""
        with self.assertRaises(FileNotFoundError):
            self.extractor.ingest_offline_bulletin("nonexistent_bulletin.pdf")

    def test_ingest_tabular_records_all_research_crops(self) -> None:
        """Verify normalization of structured HARTI records preserving Min/Max/Modal prices.

        HARTI reports all 5 target crops including Leek, and provides Min, Max, Modal distributions.
        """
        raw_records = [
            {
                "crop_name": "Carrot",
                "market_name": "Pettah",
                "price_type": "wholesale",
                "min_price": 200.0,
                "max_price": 240.0,
                "modal_price": 220.0,
                "unit": "Rs./kg",
            },
            {
                "crop_name": "Cabbage",
                "market_name": "Dambulla",
                "price_type": "wholesale",
                "min_price": 300.0,
                "max_price": 350.0,
                "modal_price": 320.0,
                "unit": "Rs./kg",
            },
            {
                "crop_name": "Leek",
                "market_name": "Keppetipola",
                "price_type": "wholesale",
                "min_price": 180.0,
                "max_price": 220.0,
                "modal_price": 200.0,
                "unit": "Rs./kg",
            },
            {
                "crop_name": "Big onion",
                "market_name": "Dambulla",
                "price_type": "wholesale",
                "min_price": 250.0,
                "max_price": 280.0,
                "modal_price": 265.0,
                "unit": "Rs./kg",
            },
            {
                "crop_name": "Tomato",
                "market_name": "Kandy",
                "price_type": "retail",
                "min_price": 450.0,
                "max_price": 550.0,
                "modal_price": 500.0,
                "unit": "Rs./kg",
            },
        ]

        normalized = self.extractor.ingest_tabular_records(
            raw_records=raw_records,
            report_date="2024-12-31",
            source_reference="HARTI_OFFLINE_FIXTURE",
        )

        self.assertEqual(len(normalized), 5)

        crops_present = {r["crop_name"] for r in normalized}
        self.assertEqual(crops_present, {"Carrot", "Cabbage", "Leek", "Big onion", "Tomato"})

        # Verify Leek is preserved from HARTI
        leek_rec = [r for r in normalized if r["crop_name"] == "Leek"][0]
        self.assertEqual(leek_rec["market_name"], "Keppetipola")
        self.assertEqual(leek_rec["min_price_lkr_per_kg"], 180.0)
        self.assertEqual(leek_rec["max_price_lkr_per_kg"], 220.0)
        self.assertEqual(leek_rec["modal_price_lkr_per_kg"], 200.0)
        self.assertEqual(leek_rec["price_lkr_per_kg"], 200.0)  # Primary price is modal
        self.assertEqual(leek_rec["source_agency"], "HARTI")
        self.assertEqual(len(leek_rec["provenance_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
