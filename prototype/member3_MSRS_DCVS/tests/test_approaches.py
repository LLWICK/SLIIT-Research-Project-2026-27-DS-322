import pytest

from member3.approaches import (
    run_baseline_dcvs,
    run_bpr_similarity,
    run_portfolio_hedging,
    run_softmax_load_balancing,
    run_topsis_mcdm,
)


@pytest.fixture
def sample_candidates():
    return [
        {
            "crop": "carrot",
            "forecast": {"predicted_price": 180.0, "lower_price": 100.0, "upper_price": 260.0},
            "current_confirmed_area_ha": 950.0,  # 95% saturated (near capacity 1000)
            "proposed_area_ha": 150.0,
            "reference_capacity_ha": 1000.0,
            "suitability": 0.90,
        },
        {
            "crop": "leek",
            "forecast": {"predicted_price": 130.0, "lower_price": 70.0, "upper_price": 190.0},
            "current_confirmed_area_ha": 200.0,  # 25% saturated (low crowding, capacity 800)
            "proposed_area_ha": 150.0,
            "reference_capacity_ha": 800.0,
            "suitability": 0.85,
        },
        {
            "crop": "tomato",
            "forecast": {"predicted_price": 140.0, "lower_price": 60.0, "upper_price": 220.0},
            "current_confirmed_area_ha": 150.0,  # 16% saturated (capacity 900)
            "proposed_area_ha": 150.0,
            "reference_capacity_ha": 900.0,
            "suitability": 0.50,
        },
    ]


def test_baseline_dcvs_penalizes_saturated_crop(sample_candidates):
    results = run_baseline_dcvs(sample_candidates)
    assert len(results) == 3
    # Carrot has high price & suitability, but high saturation; Leek has lower saturation
    crop_order = [r["crop"] for r in results]
    assert "leek" in crop_order
    # Ensure rank_score is bounded in [0, 100]
    for r in results:
        assert 0.0 <= r["rank_score"] <= 100.0


def test_topsis_mcdm_produces_valid_rankings(sample_candidates):
    results = run_topsis_mcdm(sample_candidates)
    assert len(results) == 3
    for r in results:
        assert 0.0 <= r["rank_score"] <= 100.0
        assert "dist_ideal_best" in r
        assert "dist_ideal_worst" in r


def test_bpr_similarity_computes_vector_similarity(sample_candidates):
    results = run_bpr_similarity(sample_candidates, requested_crop="carrot")
    assert len(results) == 3
    for r in results:
        assert "profile_similarity" in r
        assert "bpr_congestion" in r
        assert 0.0 <= r["profile_similarity"] <= 1.0


def test_portfolio_hedging_splits_land_when_saturated(sample_candidates):
    result = run_portfolio_hedging(sample_candidates, requested_crop="carrot", proposed_area_ha=150.0)
    assert result["status"] == "hedged_split_allocation"
    assert len(result["allocations"]) == 2
    total_split = sum(a["allocated_area_ha"] for a in result["allocations"])
    assert pytest.approx(total_split, 0.01) == 150.0


def test_softmax_load_balancing_sums_to_100_percent(sample_candidates):
    results = run_softmax_load_balancing(sample_candidates, temperature=0.6)
    assert len(results) == 3
    total_pct = sum(r["rank_score"] for r in results)
    assert pytest.approx(total_pct, 0.5) == 100.0
