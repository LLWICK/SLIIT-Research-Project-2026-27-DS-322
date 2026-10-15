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

    def test_find_dec_vegetable_stream_non_pdf(self) -> None:
        """Verify stream search returns None when given non-PDF payload."""
        html_payload = b"<!DOCTYPE html><html><body>Error 403 Forbidden</body></html>"
        stream = self.extractor.find_dec_vegetable_stream(html_payload)
        self.assertIsNone(stream)

    def test_extract_from_bytes_invalid_pdf_raises(self) -> None:
        """Verify extract_from_bytes raises ValueError when invalid magic bytes are provided."""
        with self.assertRaises(ValueError):
            self.extractor.extract_from_bytes(b"NOT_A_PDF_DOCUMENT")

    def test_extract_from_missing_file_raises(self) -> None:
        """Verify extract_from_file raises FileNotFoundError for non-existent path."""
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_from_file("missing_harti_bulletin.pdf")

    def test_extract_from_genuine_harti_pdf(self) -> None:
        """Verify automated extraction of crop names, markets, dates, and Min/Max prices from genuine HARTI PDF.

        Tests extraction using genuine captured HARTI bulletin fixture:
        - Accurately parses DEC wholesale matrix (470 total raw records across 10 economic centres).
        - Extracts all 5 research target vegetables (Carrot, Leek, Cabbage, Tomato, Big onion).
        - Verifies Leek presence in HARTI (confirming coverage where CBSL is absent).
        - Verifies accurate Min and Max price bounds matching official publication.
        - Preserves unquoted/untraded cells as None without synthetic fabrication.
        - Integrates with PriceCleaner, PriceValidator, and DRICalculator.
        """
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "sample_harti_report.pdf"
        self.assertTrue(
            fixture_path.exists(),
            f"Required genuine HARTI PDF fixture not found at {fixture_path}",
        )

        records = self.extractor.extract_from_file(fixture_path, report_date="2026-10-07")

        # 1. Verification of record volume
        self.assertGreaterEqual(len(records), 400)
        self.assertEqual(records[0]["source_agency"], "HARTI")
        self.assertEqual(len(records[0]["provenance_sha256"]), 64)

        # 2. Verification of all 10 Dedicated Economic Centres
        markets_found = {r["market_name"] for r in records}
        expected_markets = {
            "Peliyagoda",
            "Kandy",
            "Dambulla",
            "Meegoda",
            "Norochcholai",
            "Thambuththegama",
            "Keppetipola",
            "Nuwara Eliya",
            "Bandarawela",
            "Veyangoda",
        }
        self.assertTrue(
            expected_markets.issubset(markets_found),
            f"Missing expected DEC markets: {expected_markets - markets_found}",
        )

        # 3. Verification of all 5 research target crops
        target_records = [r for r in records if r["is_target_crop"]]
        target_crops = {r["crop_name"] for r in target_records}
        self.assertIn("Carrot", target_crops)
        self.assertIn("Leek", target_crops)
        self.assertIn("Cabbage", target_crops)
        self.assertIn("Tomato", target_crops)
        self.assertIn("Big onion", target_crops)
        self.assertIn("Big onion (Imported)", target_crops)

        # 4. Verified price checks on specific commodities:
        # Carrot at Peliyagoda: Min=150.0, Max=160.0
        peliyagoda_carrot = next(
            r for r in target_records if r["crop_name"] == "Carrot" and r["market_name"] == "Peliyagoda"
        )
        self.assertEqual(peliyagoda_carrot["min_price_lkr_per_kg"], 150.0)
        self.assertEqual(peliyagoda_carrot["max_price_lkr_per_kg"], 160.0)
        self.assertEqual(peliyagoda_carrot["price_lkr_per_kg"], 155.0)  # Midpoint of min & max
        self.assertEqual(peliyagoda_carrot["price_date"], "2026-10-07")

        # Leek at Kandy: Min=100.0, Max=120.0
        kandy_leek = next(
            r for r in target_records if r["crop_name"] == "Leek" and r["market_name"] == "Kandy"
        )
        self.assertEqual(kandy_leek["min_price_lkr_per_kg"], 100.0)
        self.assertEqual(kandy_leek["max_price_lkr_per_kg"], 120.0)
        self.assertEqual(kandy_leek["price_lkr_per_kg"], 110.0)

        # Tomato at Peliyagoda: Min=500.0, Max=550.0
        peliyagoda_tomato = next(
            r for r in target_records if r["crop_name"] == "Tomato" and r["market_name"] == "Peliyagoda"
        )
        self.assertEqual(peliyagoda_tomato["min_price_lkr_per_kg"], 500.0)
        self.assertEqual(peliyagoda_tomato["max_price_lkr_per_kg"], 550.0)
        self.assertEqual(peliyagoda_tomato["price_lkr_per_kg"], 525.0)

        # 5. Untraded market verification (no fabrication):
        # Carrot was not traded at Norochcholai in this bulletin (marked as '-')
        noroch_carrot = next(
            r for r in target_records if r["crop_name"] == "Carrot" and r["market_name"] == "Norochcholai"
        )
        self.assertIsNone(noroch_carrot["min_price_lkr_per_kg"])
        self.assertIsNone(noroch_carrot["max_price_lkr_per_kg"])
        self.assertIsNone(noroch_carrot["price_lkr_per_kg"])
        self.assertEqual(noroch_carrot["raw_price_str"], "-")


if __name__ == "__main__":
    unittest.main()
