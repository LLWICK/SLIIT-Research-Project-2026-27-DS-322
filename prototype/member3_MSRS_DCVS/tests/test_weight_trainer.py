import pytest

from member3.weight_trainer import (
    derive_critic_weights,
    derive_entropy_weights,
    extract_historical_price_stats,
    fit_bpr_elasticity_parameters,
    generate_mock_historical_dataset,
    optimize_weights_simulation,
    train_all_models,
    train_supervised_profit_weights,
)


def test_extract_historical_price_stats():
    stats = extract_historical_price_stats()
    assert "carrot" in stats
    assert "leek" in stats
    assert "tomato" in stats
    assert stats["carrot"]["mean"] > 100.0
    assert stats["carrot"]["stdev"] > 0.0


def test_generate_mock_historical_dataset():
    data = generate_mock_historical_dataset(num_seasons=10)
    assert len(data) == 30  # 10 seasons * 3 crops
    sample = data[0]
    assert "forecast_price" in sample
    assert "harvest_price" in sample
    assert "suitability" in sample
    assert "pressure_risk" in sample
    assert "realized_profit_lkr" in sample


def test_train_supervised_profit_weights():
    res = train_supervised_profit_weights()
    assert res.model_name.startswith("Supervised")
    assert 0.0 <= res.price_weight <= 1.0
    assert 0.0 <= res.suitability_weight <= 1.0
    assert 0.0 <= res.pressure_weight <= 1.0
    assert pytest.approx(res.price_weight + res.suitability_weight + res.pressure_weight, 0.05) == 1.0


def test_derive_entropy_and_critic_weights():
    entropy = derive_entropy_weights()
    critic = derive_critic_weights()

    for res in [entropy, critic]:
        total = res.price_weight + res.suitability_weight + res.pressure_weight
        assert pytest.approx(total, 0.05) == 1.0


def test_optimize_weights_simulation():
    res = optimize_weights_simulation(num_iterations=10)
    total = res.price_weight + res.suitability_weight + res.pressure_weight
    assert pytest.approx(total, 0.05) == 1.0


def test_fit_bpr_elasticity_parameters():
    alpha, beta = fit_bpr_elasticity_parameters()
    assert 0.05 <= alpha <= 0.5
    assert 2.0 <= beta <= 6.0


def test_train_all_models_runs_cleanly():
    report = train_all_models()
    assert "dataset_summary" in report
    assert "derived_models" in report
    assert "supervised_profit" in report["derived_models"]
    assert "shannon_entropy" in report["derived_models"]
