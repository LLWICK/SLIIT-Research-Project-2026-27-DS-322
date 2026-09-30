"""Price lookup shared by the studio. The slider does not invent a supply response."""
from __future__ import annotations

from typing import Any

ANCHOR = 0.85


def match_forecast(live: list[dict[str, Any]], crop: str, market: str, intensity: float) -> dict[str, Any] | None:
    pool = [row for row in live if row["crop"] == crop and row["market"] == market]
    if not pool:
        return None
    base = dict(pool[-1])
    base["requested_cultivation_progress"] = intensity
    if base.get("progress_in_model"):
        base["applied_cultivation_update"] = False
        base["note"] = base.get("note") or "Cultivation progress is a model feature. This screen shows the stored forecast."
        return base
    base["applied_cultivation_update"] = False
    base["note"] = (
        "Monthly district cultivation progress is not in the trained model. "
        "Moving this control does not change the price."
    )
    return base


def handoff_request(match: dict[str, Any], intensity: float) -> dict[str, Any]:
    return {
        "crop": str(match["crop"]).upper(),
        "market": str(match["market"]).upper(),
        "forecast_week": match["forecast_week"],
        "latest_cultivation_information": {
            "cultivation_progress": intensity if match.get("progress_in_model") else None,
        },
    }


def handoff_packet(match: dict[str, Any]) -> dict[str, Any]:
    progress = match.get("cultivation_progress")
    return {
        "crop": str(match["crop"]).upper(),
        "market": str(match["market"]).upper(),
        "forecast_week": match["forecast_week"],
        "predicted_price": match["predicted_price"],
        "lower_price": match["lower_price"],
        "upper_price": match["upper_price"],
        "coverage_level": match.get("coverage_level", 0.9),
        "cultivation_progress": progress,
        "cultivation_intensity": match.get("cultivation_intensity", progress),
        "commitment_source": match.get("commitment_source"),
        "model_version": match.get("model_version"),
        "applied_cultivation_update": False,
    }
