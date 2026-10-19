"""Recommendation orchestration: aggregates, forecasts, candidate scores, explanations."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .approaches import (
    run_baseline_dcvs,
    run_bpr_similarity,
    run_portfolio_hedging,
    run_softmax_load_balancing,
    run_topsis_mcdm,
)
from .config import DEMO_CAPACITY_HA, ScoringConfig
from .domain import calculate_dcvs, calculate_pressure_risk, min_max_normalize
from .forecast_adapter import ForecastAdapter, canonical_crop
from .repository import CommitmentRepository
from .suitability import calculate_suitability, get_area_suitability_map


class RecommendationService:
    def __init__(self, repository: CommitmentRepository, forecasts: ForecastAdapter, config: ScoringConfig = ScoringConfig()):
        self.repository = repository
        self.forecasts = forecasts
        self.config = config

    def assess(self, request: dict) -> dict:
        requested_crop = canonical_crop(request["crop"])
        market, horizon = request["market"], request["horizon_weeks"]
        district = request.get("district", "nuwara_eliya")

        # Resolve suitability: use provided values or auto-compute from environmental profile
        user_suitability = request.get("suitability") or {}
        auto_suitability = get_area_suitability_map(district)
        suitability_map = {crop: user_suitability.get(crop, auto_suitability.get(crop, 0.7)) for crop in self.forecasts.supported_crops()}

        if requested_crop not in suitability_map:
            raise ValueError(f"Missing suitability for requested crop: {requested_crop}")

        crops = self.forecasts.supported_crops()
        forecasts = {crop: self.forecasts.find(crop, market, horizon) for crop in crops}
        normalized_prices = min_max_normalize({crop: item["predicted_price"] for crop, item in forecasts.items()})

        recommendations = []
        candidates_for_approaches = []

        for crop in crops:
            current = self.repository.aggregate_area(request["district"], market, crop, request["season"])
            proposed = request["proposed_area_ha"] if crop == requested_crop else request["proposed_area_ha"]
            pressure = calculate_pressure_risk(current + proposed, DEMO_CAPACITY_HA[crop], self.config)
            suitability = suitability_map[crop]

            dcvs = calculate_dcvs(normalized_prices[crop], suitability, pressure.score, self.config)
            rec_item = {
                "crop": crop,
                "forecast": forecasts[crop],
                "current_confirmed_area_ha": current,
                "proposed_total_area_ha": round(current + proposed, 2),
                "proposed_area_ha": proposed,
                "reference_capacity_ha": DEMO_CAPACITY_HA[crop],
                "reference_capacity_status": "demo_placeholder_not_validated_demand",
                "pressure": asdict(pressure),
                "suitability": suitability,
                "price_attractiveness": round(normalized_prices[crop], 4),
                "dcvs": dcvs,
                "explanation": f"Forecast price is compared across available crops; pressure is {pressure.label} after the proposed area.",
            }
            recommendations.append(rec_item)
            candidates_for_approaches.append(rec_item)

        recommendations.sort(key=lambda item: item["dcvs"], reverse=True)
        requested = next(item for item in recommendations if item["crop"] == requested_crop)

        # Run multi-paradigm recommendation approaches
        approaches = {
            "baseline_dcvs": run_baseline_dcvs(candidates_for_approaches, self.config),
            "topsis_mcdm": run_topsis_mcdm(candidates_for_approaches),
            "bpr_similarity": run_bpr_similarity(candidates_for_approaches, requested_crop),
            "portfolio_hedging": run_portfolio_hedging(candidates_for_approaches, requested_crop, request["proposed_area_ha"]),
            "softmax_load_balancing": run_softmax_load_balancing(candidates_for_approaches),
        }

        # Environmental suitability breakdown for the requested district
        env_breakdown = {crop: calculate_suitability(crop, district) for crop in crops}

        return {
            "request": request,
            "requested_crop_assessment": requested,
            "recommendations": recommendations,
            "scoring_status": "experimental_candidate_not_harti_validated",
            "forecast_provenance": self.forecasts.metadata,
            "environmental_profile": env_breakdown[requested_crop],
            "district_environmental_profiles": env_breakdown,
            "approaches": approaches,
        }
