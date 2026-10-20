"""Adapter for Member 2's supplied Experiment C forecast artifact only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CROP_ALIASES = {"leeks": "leek", "leek": "leek", "carrot": "carrot", "tomato": "tomato"}


def canonical_crop(crop: str) -> str:
    normalized = crop.strip().lower().replace(" ", "_")
    if normalized not in CROP_ALIASES:
        raise ValueError(f"Unsupported crop: {crop}")
    return CROP_ALIASES[normalized]


class ForecastAdapter:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("from_member") != 2 or payload.get("to_member") != 3:
            raise ValueError("forecast artifact is not a Member 2 to Member 3 payload")
        self.metadata = {key: value for key, value in payload.items() if key != "forecasts"}
        self.forecasts: list[dict[str, Any]] = payload.get("forecasts", [])
        required = {
            "crop", "market", "forecast_issue_week", "forecast_week", "horizon_weeks",
            "predicted_price", "lower_price", "upper_price", "model_version",
        }
        if not self.forecasts or not required.issubset(self.forecasts[0]):
            raise ValueError("forecast artifact is missing required fields")

    def supported_crops(self) -> list[str]:
        return sorted({canonical_crop(record["crop"]) for record in self.forecasts})

    def supported_markets(self) -> list[str]:
        return sorted({str(record["market"]) for record in self.forecasts})

    def find(self, crop: str, market: str, horizon_weeks: int) -> dict[str, Any]:
        crop_key = canonical_crop(crop)
        matches = [
            record for record in self.forecasts
            if canonical_crop(record["crop"]) == crop_key
            and record["market"] == market
            and record["horizon_weeks"] == horizon_weeks
        ]
        if len(matches) != 1:
            raise LookupError(f"No unique forecast for {crop_key}/{market}/horizon {horizon_weeks}")
        return dict(matches[0])
