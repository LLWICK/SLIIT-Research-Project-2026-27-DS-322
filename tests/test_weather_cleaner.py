"""Unit tests for weather cleaner and fill value preservation using standard unittest."""

import unittest
import numpy as np
import pandas as pd

from pipeline.transformers.weather_cleaner import (
    WeatherCleaner,
    season_from_month,
    season_year_from_date,
)


def sample_nasa_payload():
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [80.7891, 6.9497, 1868.0],
        },
        "properties": {
            "parameter": {
                "PRECTOTCORR": {
                    "20240101": 15.2,
                    "20240102": 0.0,
                    "20240103": -999.0,  # Missing value indicator
                },
                "T2M": {
                    "20240101": 21.5,
                    "20240102": -999.0,  # Missing value
                    "20240103": 22.0,
                },
                "T2M_MAX": {
                    "20240101": 25.0,
                    "20240102": 24.5,
                    "20240103": 26.0,
                },
                "T2M_MIN": {
                    "20240101": 18.0,
                    "20240102": 17.5,
                    "20240103": 18.2,
                },
            }
        },
    }


class TestWeatherCleaner(unittest.TestCase):

    def setUp(self):
        self.cleaner = WeatherCleaner()

    def test_clean_raw_payload_structure(self):
        df = self.cleaner.clean_raw_payload(sample_nasa_payload(), district="Nuwara Eliya")
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df["obs_date"]), ["2024-01-01", "2024-01-02", "2024-01-03"])
        self.assertEqual(df["district"].iloc[0], "Nuwara Eliya")
        self.assertEqual(df["elevation_m"].iloc[0], 1868.0)

    def test_fill_value_preserved_as_nan_not_zero(self):
        """CRITICAL RULE: Missing values (-999.0) must NEVER be replaced by zero."""
        df = self.cleaner.clean_raw_payload(sample_nasa_payload(), district="Nuwara Eliya")

        # Day 1: true rain = 15.2
        self.assertEqual(
            df.loc[df["obs_date"] == "2024-01-01", "precipitation_mm"].iloc[0], 15.2
        )

        # Day 2: genuine zero rain = 0.0
        self.assertEqual(
            df.loc[df["obs_date"] == "2024-01-02", "precipitation_mm"].iloc[0], 0.0
        )

        # Day 3: -999.0 MUST be NaN, NOT 0.0!
        day3_rain = df.loc[df["obs_date"] == "2024-01-03", "precipitation_mm"].iloc[0]
        self.assertTrue(pd.isna(day3_rain))

        # Day 2: -999.0 temperature MUST be NaN
        day2_temp = df.loc[df["obs_date"] == "2024-01-02", "temp_mean_c"].iloc[0]
        self.assertTrue(pd.isna(day2_temp))

    def test_seasonal_mapping(self):
        # January belongs to Maha, with season year = year - 1
        self.assertEqual(season_from_month(1), "Maha")
        self.assertEqual(season_year_from_date(2024, 1), 2023)

        # May belongs to Yala, with season year = same year
        self.assertEqual(season_from_month(5), "Yala")
        self.assertEqual(season_year_from_date(2024, 5), 2024)

        # October belongs to Maha, with season year = same year
        self.assertEqual(season_from_month(10), "Maha")
        self.assertEqual(season_year_from_date(2024, 10), 2024)


if __name__ == "__main__":
    unittest.main()
