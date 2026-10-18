"""
Unit tests for DCS & DOA Data Validator.
SLIIT Research Project DS-322 (2026/27) - Member 1
"""

import unittest

from pipeline.validation.dcs_doa_validator import DCSDOAValidator


class TestDCSDOAValidator(unittest.TestCase):
    """Test suite for DCSDOAValidator."""

    def setUp(self):
        self.validator = DCSDOAValidator(strict_districts=True)

    def test_valid_dcs_record(self):
        """Verify a clean DCS record passes validation."""
        rec = {
            "source_name": "DCS",
            "crop": "Potato",
            "district": "Nuwara Eliya",
            "metric": "production",
            "unit": "metric_ton",
            "value": 15446.0,
            "raw_status": "valid",
            "period_start": "2007-01-01",
            "period_end": "2007-12-31",
            "provenance_hash": "a1b2c3d4e5f67890",
        }
        is_valid, issues = self.validator.validate_record(rec)
        self.assertTrue(is_valid, f"Expected valid record, got issues: {issues}")
        self.assertEqual(len(issues), 0)

    def test_valid_doa_record(self):
        """Verify a clean DOA record passes validation."""
        rec = {
            "source_name": "DOA_SEPC",
            "crop": "Big Onion",
            "district": "Anuradhapura",
            "season": "Yala",
            "year": 2024,
            "target_extent_ha": 2500.0,
            "cultivated_extent_ha": 2200.0,
            "expected_production_mt": 33000.0,
            "period_start": "2024-05-01",
            "period_end": "2024-08-31",
            "provenance_hash": "b2c3d4e5f67890a1",
        }
        is_valid, issues = self.validator.validate_record(rec)
        self.assertTrue(is_valid, f"Expected valid record, got issues: {issues}")

    def test_unrecognized_district(self):
        """Verify unrecognized district is flagged as an issue."""
        rec = {
            "source_name": "DCS",
            "crop": "Potato",
            "district": "Atlantis",
            "metric": "production",
            "unit": "metric_ton",
            "value": 100.0,
            "raw_status": "valid",
            "period_start": "2023-01-01",
            "period_end": "2023-12-31",
            "provenance_hash": "c3d4e5f67890a1b2",
        }
        is_valid, issues = self.validator.validate_record(rec)
        self.assertFalse(is_valid)
        self.assertTrue(any("Atlantis" in msg for msg in issues))

    def test_negative_value_rejected(self):
        """Verify negative production/extent values are flagged."""
        rec = {
            "source_name": "DCS",
            "crop": "Potato",
            "district": "Badulla",
            "metric": "production",
            "unit": "metric_ton",
            "value": -50.0,
            "raw_status": "valid",
            "period_start": "2023-01-01",
            "period_end": "2023-12-31",
            "provenance_hash": "d4e5f67890a1b2c3",
        }
        is_valid, issues = self.validator.validate_record(rec)
        self.assertFalse(is_valid)
        self.assertTrue(any("Negative" in msg for msg in issues))

    def test_inverted_time_period(self):
        """Verify period_start after period_end is flagged."""
        rec = {
            "source_name": "DCS",
            "crop": "Potato",
            "district": "Badulla",
            "metric": "production",
            "unit": "metric_ton",
            "value": 500.0,
            "raw_status": "valid",
            "period_start": "2023-12-31",
            "period_end": "2023-01-01",
            "provenance_hash": "e5f67890a1b2c3d4",
        }
        is_valid, issues = self.validator.validate_record(rec)
        self.assertFalse(is_valid)
        self.assertTrue(any("Inverted period" in msg for msg in issues))

    def test_batch_validation(self):
        """Verify batch validation summary statistics."""
        batch = [
            {
                "source_name": "DCS",
                "crop": "Potato",
                "district": "Nuwara Eliya",
                "metric": "production",
                "unit": "metric_ton",
                "value": 15000.0,
                "raw_status": "valid",
                "period_start": "2023-01-01",
                "period_end": "2023-12-31",
                "provenance_hash": "hash1",
            },
            {
                "source_name": "DCS",
                "crop": "Unknown",
                "district": "InvalidDistrict",
                "metric": "production",
                "unit": "metric_ton",
                "value": 100.0,
                "raw_status": "valid",
                "period_start": "2023-01-01",
                "period_end": "2023-12-31",
                "provenance_hash": "hash2",
            },
        ]
        summary = self.validator.validate_batch(batch)
        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["valid_records"], 1)
        self.assertEqual(summary["invalid_records"], 1)
        self.assertEqual(summary["pass_rate_pct"], 50.0)
        self.assertEqual(len(summary["issues"]), 1)


if __name__ == "__main__":
    unittest.main()
