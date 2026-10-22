"""The persistent price stays on last season when persistence is 1."""

from __future__ import annotations

import unittest

import pandas as pd

from simulation.market.price_process import combine_price, fit_price_process


class PriceProcessTest(unittest.TestCase):
    def test_full_persistence_repeats_last_price_when_there_is_no_shock(self) -> None:
        self.assertEqual(combine_price(180.0, 40.0, 1.0, 0.0), 180.0)
        self.assertEqual(combine_price(180.0, 40.0, 0.0, 0.0), 40.0)

    def test_a_negative_fit_is_clipped_to_zero(self) -> None:
        rows = pd.DataFrame(
            {
                "crop_code": ["carrot"] * 6,
                "ordinal": [0, 1, 2, 3, 4, 5],
                "season_code": ["Maha", "Yala"] * 3,
                "price": [100.0, 40.0, 110.0, 30.0, 120.0, 20.0],
            }
        )
        process = fit_price_process(rows)
        self.assertEqual(process.persistence, 0.0)
        self.assertGreater(process.shock_std, 0.0)

    def test_a_persistent_series_keeps_a_positive_coefficient(self) -> None:
        prices = [200.0]
        for _ in range(11):
            prices.append(0.8 * prices[-1] + 20.0)
        rows = pd.DataFrame(
            {
                "crop_code": ["carrot"] * len(prices),
                "ordinal": list(range(len(prices))),
                "season_code": ["Maha" if index % 2 == 0 else "Yala" for index in range(len(prices))],
                "price": prices,
            }
        )
        process = fit_price_process(rows)
        self.assertGreater(process.persistence, 0.5)
        self.assertLessEqual(process.persistence, 1.0)


if __name__ == "__main__":
    unittest.main()
