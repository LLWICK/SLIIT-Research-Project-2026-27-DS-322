"""District shares, a fixed adopter set, and the effect of last season's score."""

from __future__ import annotations

import unittest

from simulation.agents.population import build_population, choose_adopters, farmer_counts, plant_hectares
from simulation.config.districts import district_land_ha, district_targets, fair_share_ha
from simulation.config.settings import default_settings
from simulation.market.msrs import saturation_score
from simulation.scenarios.advisory import run_advisory


class DistrictShareTest(unittest.TestCase):
    def test_targets_match_the_checked_national_totals(self) -> None:
        settings = default_settings()
        for crop_code in ("carrot", "leeks"):
            total = sum(target.target_ha for target in district_targets(crop_code))
            self.assertAlmostEqual(total, settings.crop(crop_code).reference_area_ha, places=6)

    def test_nuwaraeliya_is_not_an_equal_slice(self) -> None:
        self.assertEqual(fair_share_ha("carrot", "Nuwaraeliya"), 1224.0)
        equal_slice = default_settings().crop("carrot").reference_area_ha / 13
        self.assertGreater(fair_share_ha("carrot", "Nuwaraeliya"), equal_slice * 5)
        self.assertEqual(fair_share_ha("carrot", "Kurunegala"), 1.0)
        self.assertEqual(fair_share_ha("tomato", "Nuwaraeliya"), 0.0)

    def test_saturation_uses_member3_pressure_thresholds(self) -> None:
        # Member 3: 0 at 80% of capacity, 0.5 at 100%, 1 at 120%.
        self.assertEqual(saturation_score(0.8 * 1224.0, 1224.0), 0.0)
        self.assertEqual(saturation_score(1224.0, 1224.0), 0.5)
        self.assertEqual(saturation_score(1.2 * 1224.0, 1224.0), 1.0)
        self.assertEqual(saturation_score(600.0, 1224.0), 0.0)
        self.assertEqual(saturation_score(3000.0, 1224.0), 1.0)


class PopulationTest(unittest.TestCase):
    def test_every_district_is_present_and_land_is_conserved(self) -> None:
        population = build_population(n_farmers=100, adoption_rate=0.3, seed=1)
        self.assertEqual(len(population), 100)
        self.assertEqual(set(farmer.district for farmer in population), set(district_land_ha()))
        self.assertAlmostEqual(sum(farmer.land_ha for farmer in population), sum(district_land_ha().values()), places=6)
        self.assertEqual(sum(farmer_counts(100).values()), 100)

    def test_the_adopter_set_depends_on_the_seed_and_not_on_the_season(self) -> None:
        self.assertEqual(choose_adopters(100, 0.0, seed=2), frozenset())
        self.assertEqual(choose_adopters(100, 1.0, seed=2), frozenset(range(100)))
        self.assertEqual(len(choose_adopters(100, 0.3, seed=2)), 30)
        self.assertEqual(choose_adopters(100, 0.3, seed=2), choose_adopters(100, 0.3, seed=2))
        self.assertNotEqual(choose_adopters(100, 0.3, seed=2), choose_adopters(100, 0.3, seed=3))
        first = build_population(100, 0.3, seed=2)
        second = build_population(100, 0.3, seed=2)
        self.assertEqual(
            tuple(farmer.adopter for farmer in first),
            tuple(farmer.adopter for farmer in second),
        )


class PlantingTest(unittest.TestCase):
    def test_reference_prices_reproduce_the_district_targets(self) -> None:
        settings = default_settings()
        farmer = next(item for item in build_population(100, 0.0, seed=0) if item.district == "Nuwaraeliya")
        prices = {crop.crop_code: crop.reference_price for crop in settings.crops}
        references = dict(prices)
        planted = plant_hectares(farmer, prices, references, msrs={})
        self.assertAlmostEqual(sum(planted.values()), farmer.land_ha, places=6)
        carrot_target = next(hectares for crop, hectares in farmer.crop_targets if crop == "carrot")
        self.assertAlmostEqual(planted["carrot"], carrot_target, places=6)

    def test_only_an_adopter_moves_land_off_a_saturated_crop(self) -> None:
        settings = default_settings()
        population = build_population(100, 1.0, seed=0)
        farmer = next(item for item in population if item.district == "Nuwaraeliya")
        self.assertTrue(farmer.adopter)
        prices = {"carrot": settings.crop("carrot").reference_price, "leeks": settings.crop("leeks").reference_price}
        saturated = {("Nuwaraeliya", "carrot"): 1.0}
        planted = plant_hectares(farmer, prices, prices, saturated)
        self.assertAlmostEqual(planted["carrot"], 0.0, places=6)
        self.assertAlmostEqual(planted["leeks"], farmer.land_ha, places=6)

        bystander = build_population(100, 0.0, seed=0)
        same = next(item for item in bystander if item.farmer_id == farmer.farmer_id)
        self.assertFalse(same.adopter)
        ignored = plant_hectares(same, prices, prices, saturated)
        self.assertGreater(ignored["carrot"], 0.0)

    def test_adopters_plant_less_carrot_after_a_crowded_season(self) -> None:
        settings = default_settings()
        initial = {
            "carrot": settings.crop("carrot").reference_price * 1.2,
            "leeks": settings.crop("leeks").reference_price,
        }
        without = run_advisory(0.0, seed=0, n_seasons=2, initial_prices=initial)
        with_score = run_advisory(1.0, seed=0, n_seasons=2, initial_prices=initial)
        carrot_without = without[without["crop_code"] == "carrot"]["area_ha"].tolist()
        carrot_with = with_score[with_score["crop_code"] == "carrot"]["area_ha"].tolist()
        self.assertAlmostEqual(carrot_without[0], carrot_with[0], places=4)
        self.assertLess(carrot_with[1], carrot_without[1])


if __name__ == "__main__":
    unittest.main()
