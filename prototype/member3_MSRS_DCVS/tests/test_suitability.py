import pytest

from member3.suitability import (
    PLANTING_AREAS,
    calculate_suitability,
    get_area_suitability_map,
)


def test_planting_areas_have_required_environmental_factors():
    assert "nuwara_eliya" in PLANTING_AREAS
    assert "dambulla" in PLANTING_AREAS
    nuwara = PLANTING_AREAS["nuwara_eliya"]
    assert nuwara.altitude_m > 1500
    assert nuwara.mean_temp_c < 20
    assert nuwara.seasonal_rainfall_mm > 1500


def test_crop_suitability_nuwara_eliya_favors_temperate_crops():
    # Nuwara Eliya is cool, high altitude -> Carrot and Leek should score high, Tomato should score lower
    carrot = calculate_suitability("carrot", "nuwara_eliya")
    leek = calculate_suitability("leek", "nuwara_eliya")
    tomato = calculate_suitability("tomato", "nuwara_eliya")

    assert carrot["composite_suitability"] >= 0.80
    assert leek["composite_suitability"] >= 0.80
    assert tomato["composite_suitability"] < carrot["composite_suitability"]
    assert "sub_scores" in carrot
    assert "environmental_factors" in carrot


def test_crop_suitability_dambulla_favors_tomato():
    # Dambulla is warm, low elevation -> Tomato should score high, Carrot and Leek lower
    carrot = calculate_suitability("carrot", "dambulla")
    tomato = calculate_suitability("tomato", "dambulla")

    assert tomato["composite_suitability"] > carrot["composite_suitability"]


def test_get_area_suitability_map():
    suit_map = get_area_suitability_map("badulla")
    assert "carrot" in suit_map
    assert "leek" in suit_map
    assert "tomato" in suit_map
    for val in suit_map.values():
        assert 0.0 <= val <= 1.0
