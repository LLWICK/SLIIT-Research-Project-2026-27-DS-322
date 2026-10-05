"""Price lookup shared by the studio. The slider writes a commitment record. It does not scale the price."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ANCHOR = 0.85


def _score_commitment(crop: str, market: str, intensity: float, issue_week: str) -> dict[str, Any]:
    engine = Path(__file__).resolve().parents[2]
    if str(engine) not in sys.path:
        sys.path.insert(0, str(engine))
    from service.runtime import score_progress

    return score_progress(
        crop,
        market,
        achieved_ha=round(float(intensity) * 100.0, 4),
        target_ha=100.0,
        report_date=issue_week,
        source="studio",
    )


def match_forecast(live: list[dict[str, Any]], crop: str, market: str, intensity: float) -> dict[str, Any] | None:
    pool = [row for row in live if row["crop"] == crop and row["market"] == market]
    if not pool:
        return None
    base = dict(pool[-1])
    base["requested_cultivation_progress"] = intensity
    if base.get("progress_in_model"):
        issue = base.get("forecast_issue_week")
        if not issue:
            base["applied_cultivation_update"] = False
            base["note"] = "This forecast has no issue date, so a cultivation record was not applied."
            return base
        try:
            scored = _score_commitment(crop, market, intensity, str(issue))
        except Exception as exc:
            base["applied_cultivation_update"] = False
            base["note"] = f"Runtime model could not be scored ({exc})."
            return base
        base["predicted_price"] = scored["predicted_price"]
        base["lower_price"] = scored["lower_price"]
        base["upper_price"] = scored["upper_price"]
        base["cultivation_progress"] = scored["cultivation_progress"]
        base["applied_cultivation_update"] = True
        base["commitment_source"] = "commitment_record"
        base["note"] = scored["note"]
        base["model_version"] = scored["model_version"]
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
        "applied_cultivation_update": bool(match.get("applied_cultivation_update")),
    }
