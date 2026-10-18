"""Recommendation orchestration: aggregates, forecasts, candidate scores, explanations."""

from __future__ import annotations

from dataclasses import asdict

from .config import DEMO_CAPACITY_HA, ScoringConfig
from .domain import calculate_dcvs, calculate_pressure_risk, min_max_normalize
from .forecast_adapter import ForecastAdapter, canonical_crop
from .repository import CommitmentRepository


class RecommendationService:
    def __init__(self, repository: CommitmentRepository, forecasts: ForecastAdapter, config: ScoringConfig = ScoringConfig()):
        self.repository = repository
        self.forecasts = forecasts
        self.config = config

    def assess(self, request: dict) -> dict:
        requested_crop = canonical_crop(request["crop"])
        market, horizon = request["market"], request["horizon_weeks"]
        if requested_crop not in request["suitability"]:
            raise ValueError(f"Missing suitability for requested crop: {requested_crop}")
        crops = self.forecasts.supported_crops()
        forecasts = {crop: self.forecasts.find(crop, market, horizon) for crop in crops}
        normalized_prices = min_max_normalize({crop: item["predicted_price"] for crop, item in forecasts.items()})

        recommendations = []
        for crop in crops:
            current = self.repository.aggregate_area(request["district"], market, crop, request["season"])
            proposed = request["proposed_area_ha"] if crop == requested_crop else request["proposed_area_ha"]
            pressure = calculate_pressure_risk(current + proposed, DEMO_CAPACITY_HA[crop], self.config)
            suitability = request["suitability"].get(crop)
            if suitability is None:
                continue
            dcvs = calculate_dcvs(normalized_prices[crop], suitability, pressure.score, self.config)
            recommendations.append({
                "crop": crop,
                "forecast": forecasts[crop],
                "current_confirmed_area_ha": current,
                "proposed_total_area_ha": round(current + proposed, 2),
                "reference_capacity_ha": DEMO_CAPACITY_HA[crop],
                "reference_capacity_status": "demo_placeholder_not_validated_demand",
                "pressure": asdict(pressure),
                "suitability": suitability,
                "price_attractiveness": round(normalized_prices[crop], 4),
                "dcvs": dcvs,
                "explanation": f"Forecast price is compared across available crops; pressure is {pressure.label} after the proposed area.",
            })
        recommendations.sort(key=lambda item: item["dcvs"], reverse=True)
        requested = next(item for item in recommendations if item["crop"] == requested_crop)
        return {
            "request": request,
            "requested_crop_assessment": requested,
            "recommendations": recommendations,
            "scoring_status": "experimental_candidate_not_harti_validated",
            "forecast_provenance": self.forecasts.metadata,
        }
