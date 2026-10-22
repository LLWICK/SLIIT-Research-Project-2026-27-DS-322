"""Tests for the baseline season: planting, clearing, crashes, and the slope ratio."""

from __future__ import annotations

import unittest
from datetime import date

from data_layer.season import season_for

from simulation.agents.farmer import aggregate_area, equal_farmers, plant_area
from simulation.config.settings import (
    default_settings,
    demand_intercept_rs,
    demand_slope_rs_per_ha,
    season_sequence,
    supply_intercept_ha,
    supply_slope_ha_per_rs,
)
from simulation.market.clearing import clear_price
from simulation.metrics.crash import crash_counts, season_is_crash
from simulation.scenarios.baseline import run_baseline


class PlantingAndClearingTest(unittest.TestCase):
    def test_area_rises_with_last_price_and_stays_inside_bounds(self) -> None:
        low = plant_area(100.0, intercept_ha=50.0, supply_slope=0.2, land_ha=80.0)
        high = plant_area(200.0, intercept_ha=50.0, supply_slope=0.2, land_ha=80.0)
        self.assertAlmostEqual(low, 70.0)
        self.assertAlmostEqual(high, 80.0)
        self.assertEqual(plant_area(0.0, intercept_ha=-5.0, supply_slope=0.0, land_ha=10.0), 0.0)

    def test_identical_farmers_sum_to_the_national_line(self) -> None:
        farmers = equal_farmers(10, land_ha=1_000.0)
        total = aggregate_area(farmers, last_price=150.0, intercept_ha=1_000.0, supply_slope=2.0)
        self.assertAlmostEqual(total, 1_300.0)

    def test_price_falls_as_area_rises(self) -> None:
        scarce = clear_price(100.0, demand_intercept=500.0, demand_slope=1.0)
        flooded = clear_price(400.0, demand_intercept=500.0, demand_slope=1.0)
        self.assertAlmostEqual(scarce, 400.0)
        self.assertAlmostEqual(flooded, 100.0)
        self.assertEqual(clear_price(600.0, demand_intercept=500.0, demand_slope=1.0), 0.0)


class CrashRuleTest(unittest.TestCase):
    def test_a_quarter_drop_is_a_crash_and_a_smaller_drop_is_not(self) -> None:
        self.assertTrue(season_is_crash(75.0, 100.0, 0.25))
        self.assertTrue(season_is_crash(74.0, 100.0, 0.25))
        self.assertFalse(season_is_crash(76.0, 100.0, 0.25))
        self.assertFalse(season_is_crash(75.0, None, 0.25))
        self.assertFalse(season_is_crash(0.0, 0.0, 0.25))

    def test_drop_fraction_outside_the_unit_interval_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            season_is_crash(50.0, 100.0, 0.0)


class SeasonLabelTest(unittest.TestCase):
    def test_labels_match_the_department_calendar(self) -> None:
        seasons = season_sequence(2015, 4)
        self.assertEqual(seasons[0], season_for(date(2015, 10, 1)))
        self.assertEqual(seasons[1], season_for(date(2016, 4, 1)))
        self.assertEqual(seasons[2], season_for(date(2016, 10, 1)))
        self.assertEqual(seasons[3], season_for(date(2017, 4, 1)))
        self.assertEqual(seasons[0][1], "2015/16 Maha")
        self.assertEqual(seasons[1][1], "2016 Yala")


class BaselinePathTest(unittest.TestCase):
    def test_slope_below_one_returns_to_the_reference_price(self) -> None:
        settings = default_settings()
        crop = settings.crop("carrot")
        initial = {item.crop_code: item.reference_price for item in settings.crops}
        initial["carrot"] = crop.reference_price * 1.2
        frame = run_baseline(settings, n_seasons=30, initial_prices=initial, slope_ratio=0.5)
        carrot = frame[frame["crop_code"] == "carrot"]
        last = float(carrot["price"].iloc[-1])
        self.assertAlmostEqual(last, crop.reference_price, delta=crop.reference_price * 0.01)
        self.assertTrue((carrot["price"] >= 0).all())
        self.assertTrue((carrot["area_ha"] >= 0).all())
        self.assertFalse(bool(carrot["is_crash"].iloc[0]))

    def test_slope_above_one_grows_away_from_the_reference_price(self) -> None:
        settings = default_settings()
        crop = settings.crop("carrot")
        initial = {item.crop_code: item.reference_price for item in settings.crops}
        initial["carrot"] = crop.reference_price * 1.1
        frame = run_baseline(settings, n_seasons=4, initial_prices=initial, slope_ratio=1.5)
        gaps = (frame.loc[frame["crop_code"] == "carrot", "price"] - crop.reference_price).abs()
        gaps = list(gaps)
        self.assertGreater(gaps[1], gaps[0])
        self.assertGreater(gaps[2], gaps[1])
        self.assertGreater(gaps[3], gaps[2])

    def test_reference_point_is_the_equilibrium(self) -> None:
        settings = default_settings()
        crop = settings.crop("leeks")
        slope = supply_slope_ha_per_rs(crop, settings.price_response_share)
        intercept = supply_intercept_ha(crop, slope)
        demand_slope = demand_slope_rs_per_ha(slope, 0.8)
        demand_intercept = demand_intercept_rs(crop, demand_slope)
        area = aggregate_area(
            equal_farmers(settings.n_farmers, 10_000.0),
            crop.reference_price,
            intercept,
            slope,
        )
        price = clear_price(area, demand_intercept, demand_slope)
        self.assertAlmostEqual(area, crop.reference_area_ha, places=6)
        self.assertAlmostEqual(price, crop.reference_price, places=6)

    def test_tomato_area_is_marked_as_an_assumption(self) -> None:
        self.assertTrue(default_settings().crop("tomato").area_is_assumed)
        self.assertFalse(default_settings().crop("carrot").area_is_assumed)

    def test_crash_count_follows_the_rule(self) -> None:
        settings = default_settings()
        crop = settings.crop("carrot")
        # A 20% start is a small gap at ratio 0.5, so the first drop is about 10%.
        initial = {item.crop_code: item.reference_price for item in settings.crops}
        initial["carrot"] = crop.reference_price * 2.0
        frame = run_baseline(settings, n_seasons=6, initial_prices=initial, slope_ratio=0.5)
        carrot = frame[frame["crop_code"] == "carrot"].reset_index(drop=True)
        # Season 0 price is reference - 0.5 * (2*reference - reference) = 0.5 * reference.
        self.assertAlmostEqual(float(carrot.loc[0, "price"]), crop.reference_price * 0.5, places=5)
        self.assertTrue(bool(carrot.loc[0, "is_crash"]) is False)
        self.assertGreater(crash_counts(frame)["carrot"], 0)


if __name__ == "__main__":
    unittest.main()
