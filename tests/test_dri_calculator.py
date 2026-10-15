"""Unit tests for Data Reliability Index (DRI) Calculator.

Tests completeness, timeliness/recency decay, consistency scoring,
configurable weights, classification tiers, and edge cases completely offline.
"""

from __future__ import annotations

import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from pipeline.quality.dri_calculator import DRICalculator, DRIConfig


class TestDRICalculator(unittest.TestCase):
    """Test suite for DRICalculator and DRIConfig."""

    def setUp(self) -> None:
        self.calc = DRICalculator()

    def test_config_weight_validation(self) -> None:
        """Verify DRIConfig raises ValueError if weights do not sum to 1.0."""
        # Valid weights sum to 1.0
        cfg = DRIConfig(weight_completeness=0.5, weight_timeliness=0.3, weight_consistency=0.2)
        self.assertEqual(cfg.weight_completeness, 0.5)

        # Invalid weights sum != 1.0
        with self.assertRaises(ValueError):
            DRIConfig(weight_completeness=0.5, weight_timeliness=0.5, weight_consistency=0.5)

    def test_completeness_calculation(self) -> None:
        """Verify completeness scoring with valid, missing, and non-positive prices."""
        # 100% valid
        df_full = pd.DataFrame({"price_lkr_per_kg": [100.0, 200.0, 300.0, 400.0]})
        score, meta = self.calc.compute_completeness(df_full, target_col="price_lkr_per_kg")
        self.assertEqual(score, 1.0)
        self.assertEqual(meta["valid_observations"], 4)

        # 50% missing (NaN and None)
        df_half = pd.DataFrame({"price_lkr_per_kg": [100.0, np.nan, 300.0, None]})
        score2, meta2 = self.calc.compute_completeness(df_half, target_col="price_lkr_per_kg")
        self.assertEqual(score2, 0.5)
        self.assertEqual(meta2["valid_observations"], 2)

        # Non-positive prices (e.g. 0.0 or negative) are treated as invalid
        df_invalid = pd.DataFrame({"price_lkr_per_kg": [100.0, 0.0, -20.0, 400.0]})
        score3, meta3 = self.calc.compute_completeness(df_invalid, target_col="price_lkr_per_kg")
        self.assertEqual(score3, 0.5)

    def test_timeliness_calculation(self) -> None:
        """Verify linear recency decay based on latency from evaluation date."""
        eval_date = datetime(2026, 10, 15)

        # 1. Fresh observation (same day / 1 day delay) -> Score 1.0
        df_fresh = pd.DataFrame({"obs_date": ["2026-10-14", "2026-10-15"]})
        score_fresh, meta_fresh = self.calc.compute_timeliness(df_fresh, evaluation_date=eval_date)
        self.assertEqual(score_fresh, 1.0)
        self.assertEqual(meta_fresh["latency_days"], 0)

        # 2. 15-day delay against 30-day window -> Score ~0.5
        df_mid = pd.DataFrame({"obs_date": ["2026-09-30"]})
        score_mid, meta_mid = self.calc.compute_timeliness(df_mid, evaluation_date=eval_date)
        self.assertAlmostEqual(score_mid, 0.5, places=1)
        self.assertEqual(meta_mid["latency_days"], 15)

        # 3. 35-day delay (exceeds 30-day window) -> Score 0.0
        df_stale = pd.DataFrame({"obs_date": ["2026-09-01"]})
        score_stale, meta_stale = self.calc.compute_timeliness(df_stale, evaluation_date=eval_date)
        self.assertEqual(score_stale, 0.0)

    def test_consistency_calculation(self) -> None:
        """Verify consistency scoring with clean and anomalous data."""
        # Clean data -> Score 1.0
        df_clean = pd.DataFrame({
            "price_lkr_per_kg": [100.0, 200.0, 300.0],
            "min_price_lkr_per_kg": [80.0, 180.0, 280.0],
            "max_price_lkr_per_kg": [120.0, 220.0, 320.0],
        })
        score_clean, meta_clean = self.calc.compute_consistency(df_clean)
        self.assertEqual(score_clean, 1.0)

        # Anomalous data (out-of-bounds price and min > max)
        df_bad = pd.DataFrame({
            "price_lkr_per_kg": [100.0, 99999.0],  # 99999 is out of bounds
            "min_price_lkr_per_kg": [80.0, 500.0],
            "max_price_lkr_per_kg": [120.0, 200.0],  # min > max
        })
        score_bad, meta_bad = self.calc.compute_consistency(df_bad)
        self.assertLess(score_bad, 1.0)

    def test_evaluate_price_dataset_full_flow(self) -> None:
        """Verify full DRI evaluation flow and high tier classification."""
        df = pd.DataFrame({
            "obs_date": ["2026-10-14"] * 6,
            "crop_name": ["Carrot"] * 6,
            "price_lkr_per_kg": [220.0, 225.0, 270.0, 255.0, 320.0, 250.0],
            "min_price_lkr_per_kg": [200.0] * 6,
            "max_price_lkr_per_kg": [350.0] * 6,
            "season_year": [2026] * 6,
            "season_type": ["Maha"] * 6,
        })
        report = self.calc.evaluate_price_dataset(
            df=df,
            evaluation_date="2026-10-15",
            crop="Carrot",
            district="Colombo",
        )
        self.assertGreaterEqual(report["calculated_dri"], 80.0)
        self.assertEqual(report["reliability_tier"], "HIGH")
        self.assertEqual(report["total_rows"], 6)
    def test_small_sample_penalty(self) -> None:
        """Verify the exact 0.7 multiplier for fewer than five observations and resulting audit note."""
        # 1. Dataset with 3 observations (< 5 min required)
        df_three = pd.DataFrame({
            "obs_date": ["2026-10-15"] * 3,
            "crop_name": ["Carrot"] * 3,
            "price_lkr_per_kg": [220.0, 230.0, 240.0],
            "season_year": [2026] * 3,
            "season_type": ["Maha"] * 3,
        })
        # Unpenalized score would be 100.0 (1.0 completeness, 1.0 timeliness, 1.0 consistency)
        report_three = self.calc.evaluate_price_dataset(
            df=df_three,
            evaluation_date="2026-10-15",
            crop="Carrot",
        )
        # Expected: exactly 100.0 * 0.7 = 70.0
        self.assertEqual(report_three["calculated_dri"], 70.0)
        self.assertIn("Small sample penalty applied: 3 rows < 5 required.", report_three["audit_notes"])

        # 2. Dataset with 5 observations (>= 5 min required) -> no penalty applied
        df_five = pd.DataFrame({
            "obs_date": ["2026-10-15"] * 5,
            "crop_name": ["Carrot"] * 5,
            "price_lkr_per_kg": [220.0, 230.0, 240.0, 250.0, 260.0],
            "season_year": [2026] * 5,
            "season_type": ["Maha"] * 5,
        })
        report_five = self.calc.evaluate_price_dataset(
            df=df_five,
            evaluation_date="2026-10-15",
            crop="Carrot",
        )
        self.assertEqual(report_five["calculated_dri"], 100.0)
        self.assertNotIn("Small sample penalty applied", report_five["audit_notes"])

    def test_save_audit_ledger_record_appends_and_preserves(self) -> None:
        """Verify audit ledger appends new records without overwriting existing entries."""
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger_path = Path(tmp_dir) / "test_dri_audit_ledger.jsonl"
            record1 = {
                "audit_id": "test_1",
                "calculated_dri": 95.5,
                "reliability_tier": "HIGH",
                "validation_status": "PASSED",
            }
            record2 = {
                "audit_id": "test_2",
                "calculated_dri": 72.0,
                "reliability_tier": "MODERATE",
                "validation_status": "PASSED",
            }

            # Write first record
            out1 = self.calc.save_audit_ledger_record(record1, output_path=ledger_path)
            self.assertEqual(out1, ledger_path)
            self.assertTrue(ledger_path.exists())

            # Write second record
            out2 = self.calc.save_audit_ledger_record(record2, output_path=ledger_path)
            self.assertEqual(out2, ledger_path)

            # Verify contents
            with open(ledger_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            self.assertEqual(len(lines), 2)
            parsed_1 = json.loads(lines[0])
            parsed_2 = json.loads(lines[1])
            self.assertEqual(parsed_1["audit_id"], "test_1")
            self.assertEqual(parsed_1["calculated_dri"], 95.5)
            self.assertEqual(parsed_2["audit_id"], "test_2")
            self.assertEqual(parsed_2["calculated_dri"], 72.0)

    def test_save_audit_ledger_creates_parent_directories(self) -> None:
        """Verify ledger writer creates parent directories when they do not exist."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_ledger = Path(tmp_dir) / "deep" / "nested" / "prices" / "audit.jsonl"
            self.assertFalse(nested_ledger.parent.exists())

            record = {"calculated_dri": 85.0, "reliability_tier": "HIGH"}
            saved_path = self.calc.save_audit_ledger_record(record, output_path=nested_ledger)

            self.assertTrue(nested_ledger.parent.exists())
            self.assertTrue(saved_path.exists())

    def test_save_audit_ledger_invalid_path(self) -> None:
        """Verify error handling when output path is an invalid target (e.g. existing directory)."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_target = Path(tmp_dir)
            with self.assertRaises(IsADirectoryError):
                self.calc.save_audit_ledger_record({"record": 1}, output_path=dir_target)

    def test_invalid_validation_does_not_silently_produce_high_tier(self) -> None:
        """Verify that datasets failing validation are not silently marked as HIGH reliability."""
        df = pd.DataFrame({
            "obs_date": ["2026-10-15"] * 6,
            "crop_name": ["Carrot"] * 6,
            "price_lkr_per_kg": [220.0] * 6,
            "season_year": [2026] * 6,
            "season_type": ["Maha"] * 6,
        })
        # Mock a failed validation report
        failed_val = {
            "is_valid": False,
            "errors": ["Critical domain sanity check failed"],
            "warnings": [],
        }
        report = self.calc.evaluate_price_dataset(
            df=df,
            validation_report=failed_val,
            evaluation_date="2026-10-15",
            crop="Carrot",
        )
        self.assertEqual(report["validation_status"], "FAILED")
        self.assertNotEqual(report["reliability_tier"], "HIGH")
    def test_empty_dataset_handling(self) -> None:
        """Verify transparent handling of empty datasets."""
        report = self.calc.evaluate_price_dataset(pd.DataFrame())
        self.assertEqual(report["calculated_dri"], 0.0)
        self.assertEqual(report["reliability_tier"], "INSUFFICIENT_DATA")
        self.assertEqual(report["total_rows"], 0)


if __name__ == "__main__":
    unittest.main()