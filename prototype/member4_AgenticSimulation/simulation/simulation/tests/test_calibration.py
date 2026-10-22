"""The cobweb comparison recovers a known ratio and does not invent one."""

from __future__ import annotations

import unittest

import pandas as pd

from data_layer.paths import DUCKDB_PATH

from simulation.calibration.compare import compare_with_harti, fit_crop, series_moments
from simulation.config.settings import default_settings, season_ordinal, season_sequence


def _season_frame(prices: list[float], crop_code: str = "carrot") -> pd.DataFrame:
    seasons = season_sequence(2015, len(prices))
    rows = []
    for ordinal, (price, (code, label)) in enumerate(zip(prices, seasons, strict=True)):
        rows.append(
            {
                "crop_code": crop_code,
                "season_code": code,
                "season_label": label,
                "price": float(price),
                "n_weeks": 20,
                "ordinal": ordinal,
            }
        )
    return pd.DataFrame(rows)


class SeasonOrdinalTest(unittest.TestCase):
    def test_ordinal_follows_the_season_sequence(self) -> None:
        for ordinal, (code, label) in enumerate(season_sequence(2015, 6)):
            self.assertEqual(season_ordinal(code, label), ordinal)


class FitTest(unittest.TestCase):
    def test_a_pure_cycle_selects_ratio_one(self) -> None:
        settings = default_settings()
        frame = _season_frame([120, 80, 120, 80, 120, 80, 120, 80])
        result = fit_crop(settings.crop("carrot"), frame, settings)
        self.assertEqual(result.best_ratio, 1.0)
        self.assertAlmostEqual(result.cobweb_rmse, 0.0, places=6)
        self.assertEqual(result.closest, "cobweb")

    def test_a_rising_series_is_closer_to_persistence(self) -> None:
        settings = default_settings()
        frame = _season_frame([100, 110, 120, 130, 140, 150, 160, 170])
        result = fit_crop(settings.crop("carrot"), frame, settings)
        self.assertEqual(result.closest, "persist")
        self.assertEqual(result.best_ratio, 0.05)

    def test_maha_yala_level_gap_is_removed_before_the_lag(self) -> None:
        # Maha rises 100, 110, 120. Yala rises 50, 55, 60. The raw lag flips sign
        # because Yala is cheaper. The within-season deviations both rise.
        frame = _season_frame([100, 50, 110, 55, 120, 60])
        moments = series_moments(frame, drop_fraction=0.25)
        self.assertIsNotNone(moments.lag_corr)
        self.assertIsNotNone(moments.lag_corr_deseasoned)
        self.assertLess(moments.lag_corr, 0)
        self.assertGreater(moments.lag_corr_deseasoned, 0)
        self.assertEqual(moments.crashes, 3)

    def test_a_calendar_gap_is_not_a_pair(self) -> None:
        frame = _season_frame([100, 70, 100])
        frame.loc[2, "ordinal"] = 3
        moments = series_moments(frame, drop_fraction=0.25)
        self.assertEqual(moments.n_pairs, 1)
        self.assertEqual(moments.crashes, 1)


class WarehouseComparisonTest(unittest.TestCase):
    def test_colombo_wholesale_seasons_are_scored(self) -> None:
        if not DUCKDB_PATH.exists():
            self.skipTest("local warehouse is not built")
        results = compare_with_harti()
        self.assertEqual([item.crop_code for item in results], ["carrot", "leeks", "tomato"])
        closest = {"carrot": "flat", "leeks": "flat", "tomato": "persist"}
        for result in results:
            self.assertGreaterEqual(result.moments.n_seasons, 10)
            self.assertGreater(result.moments.lag_corr_deseasoned, 0)
            self.assertEqual(result.best_ratio, 0.05)
            self.assertEqual(result.closest, closest[result.crop_code])


if __name__ == "__main__":
    unittest.main()
