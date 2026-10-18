"""
Unit tests for DOA SEPC Extractor.
SLIIT Research Project DS-322 (2026/27) - Member 1
"""

from pathlib import Path
import unittest

from pipeline.extractors.doa_extractor import (
    DOAExtractor,
    DOAPublicationRecord,
    DOACultivationRecord,
    UnsupportedSourceFormatError,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestDOAExtractor(unittest.TestCase):
    """Test suite for DOAExtractor."""

    def setUp(self):
        self.extractor = DOAExtractor()
        self.sepc_html_fixture = FIXTURES_DIR / "doa_sepc_downloads_sample.html"
        self.cultivation_csv_fixture = FIXTURES_DIR / "doa_crop_forecast_sample.csv"

    def test_extract_catalog_from_html(self):
        """Verify parsing of publications from DOA SEPC downloads portal."""
        content = self.sepc_html_fixture.read_text(encoding="utf-8")
        catalog = self.extractor.extract_catalog_from_html(content)

        self.assertGreater(len(catalog), 0, "Should extract publication records")

        categories = {p.category for p in catalog}
        self.assertIn("Agstat Booklet", categories)
        self.assertIn("Crop Forecast Monthly Report", categories)

        # Check languages
        languages = {p.language for p in catalog}
        self.assertIn("English", languages)
        self.assertIn("Sinhala", languages)
        self.assertIn("Tamil", languages)

        # Check download links point to Google Drive
        for p in catalog:
            self.assertTrue(p.download_url.startswith("https://drive.google.com/file/d/"))
            self.assertEqual(p.source_page_url, "https://doa.gov.lk/sepc-downloads_en/")

        # Verify season detection for Crop Forecast
        crop_forecast = [p for p in catalog if p.category == "Crop Forecast Monthly Report"]
        seasons = {p.season for p in crop_forecast if p.season}
        self.assertTrue("Yala" in seasons or "Maha" in seasons)

    def test_extract_cultivation_from_csv(self):
        """Verify structured ingestion of DOA cultivation records from CSV fixture."""
        records = self.extractor.extract_cultivation_from_csv(self.cultivation_csv_fixture)

        self.assertEqual(len(records), 5)

        first = records[0]
        self.assertEqual(first.crop, "Big Onion")
        self.assertEqual(first.district, "Anuradhapura")
        self.assertEqual(first.season, "Yala")
        self.assertEqual(first.year, 2024)
        self.assertEqual(first.target_extent_ha, 2500.0)
        self.assertEqual(first.cultivated_extent_ha, 2200.0)
        self.assertEqual(first.expected_production_mt, 33000.0)
        self.assertEqual(first.raw_source, "DOA_SEPC_CROP_FORECAST_YALA_2024")

    def test_unsupported_pdf_publication_format_handling(self):
        """Verify that attempting to parse Google Drive PDF booklets raises UnsupportedSourceFormatError."""
        pub_rec = DOAPublicationRecord(
            category="Agstat Booklet",
            title="Agstat Booklet - 2025",
            year=2025,
            season=None,
            language="English",
            download_url="https://drive.google.com/file/d/1vchIx64sHA3hKp2U-g9VPsCmp1Bo_I6W/view?usp=sharing",
        )

        with self.assertRaises(UnsupportedSourceFormatError) as ctx:
            self.extractor.extract_pdf_publication(pub_rec)

        self.assertIn("Google Drive", str(ctx.exception))
        self.assertIn("pdfplumber", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
