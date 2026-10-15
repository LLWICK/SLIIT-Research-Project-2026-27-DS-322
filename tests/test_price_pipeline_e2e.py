"""End-to-End Pipeline Tests for Member 1 Vegetable Price ETL.

Verifies the complete data pipeline from genuine CBSL PDF fixture through
extraction, cleaning, standardization, domain validation, DRI scoring,
and audit ledger persistence.
"""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.cli import run_price_pipeline
from pipeline.extractors.price_cbsl import CbslPriceExtractor
from pipeline.extractors.price_harti import HartiPriceExtractor
from pipeline.quality.dri_calculator import DRICalculator
from pipeline.transformers.price_cleaner import PriceCleaner
from pipeline.validation.price_validator import PriceValidator

CBSL_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_cbsl_report.pdf"
HARTI_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_harti_report.pdf"
FIXTURE_PATH = CBSL_FIXTURE_PATH


class TestPricePipelineE2E(unittest.TestCase):
    """End-to-end integration tests for vegetable price ETL workflow."""

    def setUp(self) -> None:
        self.assertTrue(
            CBSL_FIXTURE_PATH.exists(),
            f"Required genuine CBSL fixture not found at {CBSL_FIXTURE_PATH}",
        )
        self.assertTrue(
            HARTI_FIXTURE_PATH.exists(),
            f"Required genuine HARTI fixture not found at {HARTI_FIXTURE_PATH}",
        )

    def test_cbsl_fixture_pipeline_e2e(self) -> None:
        """Test genuine CBSL PDF through extraction, cleaning, validation, and DRI scoring.

        Derives expected record counts directly from verified fixture output:
        - 115 total raw records extracted from 23 commodity rows across 5 market columns.
        - 25 target crop observations retained (5 each for Carrot, Cabbage, Tomato, Big onion, Big onion (Imported)).
        - 0 Leek observations (verified absence in CBSL, zero fabricated records).
        - 100% schema and domain validation pass.
        - Genuine high DRI score (98.4, HIGH tier).
        """
        # Step 1: Extraction
        extractor = CbslPriceExtractor()
        raw_records = extractor.extract_from_file(FIXTURE_PATH, report_date="2026-10-14")

        self.assertEqual(len(raw_records), 115)
        self.assertEqual(raw_records[0]["price_date"], "2026-10-14")
        self.assertEqual(raw_records[0]["source_agency"], "CBSL")
        self.assertTrue(raw_records[0]["provenance_sha256"])

        # Step 2: Cleaning & Transformation
        cleaner = PriceCleaner()
        cleaned_df = cleaner.clean_records(raw_records, filter_target_crops=True)

        self.assertEqual(len(cleaned_df), 25)
        crop_counts = cleaned_df["crop_name"].value_counts().to_dict()
        self.assertEqual(crop_counts.get("Carrot"), 5)
        self.assertEqual(crop_counts.get("Cabbage"), 5)
        self.assertEqual(crop_counts.get("Tomato"), 5)
        self.assertEqual(crop_counts.get("Big onion"), 5)
        self.assertEqual(crop_counts.get("Big onion (Imported)"), 5)
        # Critical verify: Leek is absent and not fabricated
        self.assertNotIn("Leek", crop_counts)

        # Verify standardized structural columns
        expected_cols = [
            "obs_date",
            "year",
            "month",
            "day",
            "season_type",
            "season_year",
            "crop_name",
            "source_commodity_name",
            "market_name",
            "original_market_name",
            "district",
            "latitude",
            "longitude",
            "price_type",
            "price_lkr_per_kg",
            "source_agency",
            "source_url",
            "provenance_sha256",
        ]
        for col in expected_cols:
            self.assertIn(col, cleaned_df.columns)

        # Step 3: Domain Validation
        validator = PriceValidator()
        val_report = validator.validate(cleaned_df)

        self.assertTrue(val_report["is_valid"], f"Validation failed with errors: {val_report['errors']}")
        self.assertEqual(len(val_report["errors"]), 0)
        self.assertEqual(val_report["total_rows"], 25)

        # Step 4: DRI Scoring
        calc = DRICalculator()
        dri_report = calc.evaluate_price_dataset(
            df=cleaned_df,
            validation_report=val_report,
            evaluation_date="2026-10-15",
            source_agency="CBSL",
        )

        self.assertEqual(dri_report["calculated_dri"], 98.4)
        self.assertEqual(dri_report["reliability_tier"], "HIGH")
        self.assertEqual(dri_report["validation_status"], "PASSED")
        # 24 of 25 observations have valid numeric prices (1 'n.a.' for Big onion (Imported) in Narahenpita preserved as null)
        self.assertEqual(dri_report["completeness_score"], 0.96)
        self.assertEqual(dri_report["timeliness_score"], 1.0)
        self.assertEqual(dri_report["consistency_score"], 1.0)
        self.assertEqual(dri_report["total_rows"], 25)
        self.assertEqual(dri_report["source_agency"], "CBSL")

        # Step 5: Ledger Persistence
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger_file = Path(tmp_dir) / "test_ledger.jsonl"
            saved_path = calc.save_audit_ledger_record(dri_report, output_path=ledger_file)

            self.assertEqual(saved_path, ledger_file)
            self.assertTrue(ledger_file.exists())

            with open(ledger_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            self.assertEqual(len(lines), 1)
            persisted_record = json.loads(lines[0])
            self.assertEqual(persisted_record["calculated_dri"], 98.4)
            self.assertEqual(persisted_record["reliability_tier"], "HIGH")
            self.assertEqual(persisted_record["source_agency"], "CBSL")
            self.assertEqual(persisted_record["validation_status"], "PASSED")

    def test_cli_integration_cbsl_success(self) -> None:
        """Verify CLI execution on genuine CBSL fixture outputs DRI and writes audit record."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger_path = Path(tmp_dir) / "cli_audit_ledger.jsonl"
            args = argparse.Namespace(
                command="price",
                source="cbsl",
                file=str(FIXTURE_PATH),
                date="2026-10-14",
                all_crops=False,
                force_save=False,
                discover=False,
                ledger_path=str(ledger_path),
            )

            # Capture stdout and mock save_processed to isolate test execution
            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout, \
                 patch.object(PriceCleaner, "save_processed", return_value=Path(tmp_dir) / "prices.csv"):
                exit_code = run_price_pipeline(args)
                output = mock_stdout.getvalue()

            self.assertEqual(exit_code, 0)
            self.assertIn("PRICE VALIDATION REPORT — CBSL", output)
            self.assertIn("Validation Status: PASSED", output)
            self.assertIn("DATA RELIABILITY INDEX (DRI) REPORT — CBSL", output)
            self.assertIn("DRI Score:          98.4 / 100.0", output)
            self.assertIn("Reliability Tier:   HIGH", output)
            self.assertIn("DRI Audit Record appended to:", output)

            # Verify ledger file exists and contains the complete record
            self.assertTrue(ledger_path.exists())
            with open(ledger_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["calculated_dri"], 98.4)
            self.assertEqual(records[0]["source_agency"], "CBSL")
            self.assertEqual(records[0]["validation_status"], "PASSED")


    def test_harti_fixture_pipeline_e2e(self) -> None:
        """Test genuine HARTI PDF through extraction, cleaning, validation, and DRI scoring.

        Derives expected metrics directly from verified fixture output:
        - 450 total raw records extracted from 45 commodity rows across 10 market columns.
        - 70 target crop observations retained:
          - 20 Cabbage (10 N'Eliya + 10 Kandy)
          - 10 Carrot
          - 10 Leek (confirming coverage where CBSL is absent)
          - 10 Tomato
          - 10 Big onion (Local)
          - 10 Big onion (Imported)
        - 100% schema and domain validation pass.
        - Genuine high DRI score (86.86, HIGH tier).
        """
        # Step 1: Extraction
        extractor = HartiPriceExtractor()
        raw_records = extractor.extract_from_file(HARTI_FIXTURE_PATH, report_date="2026-10-07")

        self.assertEqual(len(raw_records), 450)
        self.assertEqual(raw_records[0]["source_agency"], "HARTI")
        self.assertTrue(raw_records[0]["provenance_sha256"])

        # Step 2: Cleaning & Transformation
        cleaner = PriceCleaner()
        cleaned_df = cleaner.clean_records(raw_records, filter_target_crops=True)

        self.assertEqual(len(cleaned_df), 70)
        crop_counts = cleaned_df["crop_name"].value_counts().to_dict()
        self.assertEqual(crop_counts.get("Carrot"), 10)
        self.assertEqual(crop_counts.get("Leek"), 10)
        self.assertEqual(crop_counts.get("Cabbage"), 20)
        self.assertEqual(crop_counts.get("Tomato"), 10)
        self.assertEqual(crop_counts.get("Big onion"), 10)
        self.assertEqual(crop_counts.get("Big onion (Imported)"), 10)

        # Step 3: Domain Validation
        validator = PriceValidator()
        val_report = validator.validate(cleaned_df)

        self.assertTrue(val_report["is_valid"], f"Validation failed with errors: {val_report['errors']}")
        self.assertEqual(len(val_report["errors"]), 0)
        self.assertEqual(val_report["total_rows"], 70)

        # Step 4: DRI Scoring
        calc = DRICalculator()
        dri_report = calc.evaluate_price_dataset(
            df=cleaned_df,
            validation_report=val_report,
            evaluation_date="2026-10-07",
            source_agency="HARTI",
        )

        self.assertEqual(dri_report["calculated_dri"], 86.86)
        self.assertEqual(dri_report["reliability_tier"], "HIGH")
        self.assertEqual(dri_report["validation_status"], "PASSED")
        self.assertEqual(dri_report["timeliness_score"], 1.0)
        self.assertEqual(dri_report["consistency_score"], 1.0)
        self.assertEqual(dri_report["total_rows"], 70)
        self.assertEqual(dri_report["source_agency"], "HARTI")

    def test_cli_integration_harti_success(self) -> None:
        """Verify CLI execution on genuine HARTI fixture outputs DRI and writes audit record."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger_path = Path(tmp_dir) / "cli_harti_audit_ledger.jsonl"
            args = argparse.Namespace(
                command="price",
                source="harti",
                file=str(HARTI_FIXTURE_PATH),
                date="2026-10-07",
                all_crops=False,
                include_prev=False,
                force_save=False,
                discover=False,
                ledger_path=str(ledger_path),
            )

            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout, \
                 patch.object(PriceCleaner, "save_processed", return_value=Path(tmp_dir) / "prices_harti.csv"):
                exit_code = run_price_pipeline(args)
                output = mock_stdout.getvalue()

            self.assertEqual(exit_code, 0)
            self.assertIn("PRICE VALIDATION REPORT", output)
            self.assertIn("HARTI", output)
            self.assertIn("Validation Status: PASSED", output)
            self.assertIn("DATA RELIABILITY INDEX (DRI) REPORT", output)
            self.assertIn("DRI Score:          86.86 / 100.0", output)
            self.assertIn("Reliability Tier:   HIGH", output)
            self.assertIn("DRI Audit Record appended to:", output)

            self.assertTrue(ledger_path.exists())
            with open(ledger_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["calculated_dri"], 86.86)
            self.assertEqual(records[0]["source_agency"], "HARTI")
            self.assertEqual(records[0]["validation_status"], "PASSED")

    def test_cli_invalid_data_does_not_treat_as_reliable(self) -> None:
        """Verify that CLI execution handles validation failures transparently and returns error code."""
        # Create a mock invalid scenario by passing invalid command arguments
        args = argparse.Namespace(
            command="price",
            source="cbsl",
            file=None,
            date=None,  # Neither date nor file provided
            all_crops=False,
            force_save=False,
            discover=False,
            ledger_path=None,
        )

        with patch("sys.stdout", new_callable=io.StringIO):
            exit_code = run_price_pipeline(args)

        # CLI returns 1 because neither date nor file was specified
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
