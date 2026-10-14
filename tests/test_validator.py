"""Unit tests for weather data validator using standard unittest."""

import unittest
import numpy as np
import pandas as pd

from pipeline.validation.validator import WeatherValidator


class TestWeatherValidator(unittest.TestCase):

    def setUp(self):
        self.validator = WeatherValidator()

    def test_validator_passes_valid_data(self):
        df = pd.DataFrame(
            {
                "obs_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "district": ["Nuwara Eliya", "Nuwara Eliya", "Nuwara Eliya"],
                "year": [2024, 2024, 2024],
                "month": [1, 1, 1],
                "season_type": ["Maha", "Maha", "Maha"],
                "season_year": [2023, 2023, 2023],
                "precipitation_mm": [5.2, 0.0, 12.1],
                "temp_mean_c": [18.5, 19.0, 18.2],
                "temp_min_c": [14.0, 15.0, 14.5],
                "temp_max_c": [22.0, 23.5, 21.0],
                "relative_humidity_pct": [85.0, 80.0, 90.0],
                "wind_speed_ms": [2.5, 3.1, 1.8],
            }
        )

        report = self.validator.validate(df)
        self.assertTrue(report["is_valid"])
        self.assertEqual(len(report["errors"]), 0)
        self.assertEqual(report["total_rows"], 3)

    def test_validator_catches_duplicate_dates(self):
        df = pd.DataFrame(
            {
                "obs_date": ["2024-01-01", "2024-01-01"],  # Duplicate
                "district": ["Badulla", "Badulla"],
                "year": [2024, 2024],
                "month": [1, 1],
                "season_type": ["Maha", "Maha"],
                "season_year": [2023, 2023],
            }
        )
        report = self.validator.validate(df)
        self.assertFalse(report["is_valid"])
        self.assertTrue(any("duplicate" in err.lower() for err in report["errors"]))

    def test_validator_catches_temp_inversion(self):
        df = pd.DataFrame(
            {
                "obs_date": ["2024-01-01"],
                "district": ["Kandy"],
                "year": [2024],
                "month": [1],
                "season_type": ["Maha"],
                "season_year": [2023],
                "temp_min_c": [25.0],
                "temp_max_c": [20.0],  # Inversion: max < min
            }
        )
        report = self.validator.validate(df)
        self.assertFalse(report["is_valid"])
        self.assertTrue(any("temp_max_c < temp_min_c" in err for err in report["errors"]))


if __name__ == "__main__":
    unittest.main()
