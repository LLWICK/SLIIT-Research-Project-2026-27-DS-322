"""FastAPI surface for the Member 3 prototype."""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .forecast_adapter import ForecastAdapter, canonical_crop
from .repository import CommitmentRepository
from .service import RecommendationService
from .suitability import PLANTING_AREAS, calculate_suitability, get_area_suitability_map

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Member3Data"
DB_PATH = Path(os.getenv("MEMBER3_DB_PATH", ROOT / "member3_commitments.sqlite3"))


class AssessmentRequest(BaseModel):
    district: str = Field(min_length=1)
    market: str = Field(min_length=1)
    crop: str = Field(min_length=1)
    season: str = Field(pattern="^(Maha|Yala)$")
    proposed_area_ha: float = Field(gt=0)
    horizon_weeks: int = Field(default=4, ge=1)
    suitability: dict[str, float] = Field(default_factory=dict)

    @field_validator("suitability")
    @classmethod
    def validate_suitability(cls, values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        normalized = {canonical_crop(crop): value for crop, value in values.items()}
        if any(not 0 <= value <= 1 for value in normalized.values()):
            raise ValueError("suitability values must be in [0, 1]")
        return normalized


class CommitmentCreate(AssessmentRequest):
    idempotency_key: str = Field(min_length=8)
    farmer_reference: str = Field(min_length=1)
    planting_date: date
    harvest_date: date
    confirmed: bool


class AreaUpdate(BaseModel):
    area_ha: float = Field(gt=0)


def build_app(database_path: Path = DB_PATH) -> FastAPI:
    repository = CommitmentRepository(database_path)
    adapter = ForecastAdapter(DATA_DIR / "member3_forecast_c.json")
    service = RecommendationService(repository, adapter)
    app = FastAPI(title="Member 3 Crowding-Aware Crop Recommendation Prototype", version="0.2.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "forecast_source": "Member3Data/member3_forecast_c.json",
            "supported_approaches": [
                "baseline_dcvs",
                "topsis_mcdm",
                "bpr_similarity",
                "portfolio_hedging",
                "softmax_load_balancing",
            ],
        }

    @app.get("/planting-areas")
    def list_planting_areas() -> dict:
        """List all planting areas with environmental profiles and crop suitabilities."""
        areas = {}
        for key, profile in PLANTING_AREAS.items():
            area_dict = asdict(profile)
            area_dict["crop_suitabilities"] = get_area_suitability_map(key)
            areas[key] = area_dict
        return {"planting_areas": areas}

    @app.get("/planting-areas/{district}")
    def get_planting_area(district: str) -> dict:
        """Get environmental profile and crop suitability breakdown for a specific district."""
        key = district.strip().lower().replace(" ", "_")
        profile = PLANTING_AREAS.get(key)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Planting area '{district}' not found")
        crops = adapter.supported_crops()
        breakdown = {crop: calculate_suitability(crop, key) for crop in crops}
        return {
            "profile": asdict(profile),
            "suitability_breakdown": breakdown,
        }

    @app.post("/recommendations/assess")
    def assess(request: AssessmentRequest) -> dict:
        try:
            return service.assess(request.model_dump())
        except (LookupError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/commitments", status_code=201)
    def confirm_commitment(request: CommitmentCreate) -> dict:
        if not request.confirmed:
            raise HTTPException(status_code=400, detail="A proposal is not a commitment. Use /recommendations/assess until the farmer confirms.")
        payload = request.model_dump()
        payload["crop"] = canonical_crop(payload["crop"])
        payload["planting_date"] = str(payload["planting_date"])
        payload["harvest_date"] = str(payload["harvest_date"])
        payload["status"] = "confirmed"
        payload["area_ha"] = payload.pop("proposed_area_ha")
        payload.pop("confirmed")
        payload.pop("suitability", None)
        payload.pop("horizon_weeks", None)
        return repository.create(payload)

    @app.patch("/commitments/{commitment_id}")
    def update_commitment(commitment_id: int, update: AreaUpdate) -> dict:
        record = repository.update_area(commitment_id, update.area_ha)
        if not record:
            raise HTTPException(status_code=404, detail="Confirmed commitment not found")
        return record

    @app.delete("/commitments/{commitment_id}")
    def cancel_commitment(commitment_id: int) -> dict:
        record = repository.cancel(commitment_id)
        if not record:
            raise HTTPException(status_code=404, detail="Commitment not found")
        return record

    @app.get("/market-state/{district}/{market}/{crop}/{season}")
    def market_state(district: str, market: str, crop: str, season: str) -> dict:
        crop_key = canonical_crop(crop)
        return {"confirmed_area_ha": repository.aggregate_area(district, market, crop_key, season)}

    return app


app = build_app()
