"""Same intensity interpolation Member 3 will call later."""

from __future__ import annotations

from typing import Any

ANCHOR = 0.85


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def interpolate(pool: list[dict[str, Any]], intensity: float) -> dict[str, Any]:
    sorted_rows = sorted(pool, key=lambda row: row["cultivation_intensity"])
    if intensity <= sorted_rows[0]["cultivation_intensity"]:
        return dict(sorted_rows[0])
    last = sorted_rows[-1]
    if intensity >= last["cultivation_intensity"]:
        return dict(last)
    right = next(i for i, row in enumerate(sorted_rows) if row["cultivation_intensity"] >= intensity)
    hi = sorted_rows[right]
    lo = sorted_rows[right - 1]
    span = hi["cultivation_intensity"] - lo["cultivation_intensity"] or 1
    t = (intensity - lo["cultivation_intensity"]) / span
    out = dict(lo)
    out["cultivation_intensity"] = intensity
    out["predicted_price"] = _lerp(lo["predicted_price"], hi["predicted_price"], t)
    out["lower_price"] = _lerp(lo["lower_price"], hi["lower_price"], t)
    out["upper_price"] = _lerp(lo["upper_price"], hi["upper_price"], t)
    return out


def with_planting_pressure(base: dict[str, Any], intensity: float) -> dict[str, Any]:
    # Extra committed land means more harvest later, so the expected price eases.
    delta = intensity - ANCHOR
    factor = 1 - 0.18 * delta
    point = base["predicted_price"] * factor
    down = (base["predicted_price"] - base["lower_price"]) * (1 + 0.12 * abs(delta))
    up = (base["upper_price"] - base["predicted_price"]) * (1 + 0.12 * abs(delta))
    out = dict(base)
    out["cultivation_intensity"] = intensity
    out["predicted_price"] = round(point)
    out["lower_price"] = round(max(1, point - down))
    out["upper_price"] = round(point + up)
    return out


def match_forecast(live: list[dict[str, Any]], crop: str, market: str, intensity: float) -> dict[str, Any] | None:
    pool = [row for row in live if row["crop"] == crop and row["market"] == market]
    if not pool:
        return None
    return with_planting_pressure(interpolate(pool, ANCHOR), intensity)


def handoff_request(match: dict[str, Any], intensity: float) -> dict[str, Any]:
    return {
        "crop": str(match["crop"]).upper(),
        "market": str(match["market"]).upper(),
        "forecast_week": match["forecast_week"],
        "current_commitment_hectares": round(intensity * 1000),
    }


def handoff_packet(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "crop": str(match["crop"]).upper(),
        "market": str(match["market"]).upper(),
        "forecast_week": match["forecast_week"],
        "predicted_price": match["predicted_price"],
        "lower_price": match["lower_price"],
        "upper_price": match["upper_price"],
        "coverage_level": match["coverage_level"],
        "cultivation_intensity": match["cultivation_intensity"],
        "commitment_source": match["commitment_source"],
        "model_version": match["model_version"],
    }
