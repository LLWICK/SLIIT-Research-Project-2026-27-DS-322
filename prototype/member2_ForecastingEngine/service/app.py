"""Forecast service. It returns a price interval. It does not recommend what to plant."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

STUDIO = Path(__file__).resolve().parents[1] / "htfe-studio" / "data"
MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "methodology" / "manifest.json"

app = FastAPI(title="HTFE forecast", version="methodology")


class CultivationUpdate(BaseModel):
    achieved_ha: float | None = None
    target_ha: float | None = None
    cultivation_progress: float | None = None


class ForecastRequest(BaseModel):
    crop: str
    market: str
    forecast_week: str | None = None
    latest_cultivation_information: CultivationUpdate | None = None


class ForecastResponse(BaseModel):
    crop: str
    market: str
    forecast_week: str | None = None
    predicted_price: float | None = None
    lower_price: float | None = None
    upper_price: float | None = None
    coverage_level: float = 0.9
    cultivation_progress: float | None = None
    model_version: str | None = None
    applied_cultivation_update: bool = False
    note: str = ""


def _live() -> list[dict]:
    path = STUDIO / "live_forecast.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest() -> dict:
    if not MANIFEST.exists():
        return {"progress_in_model": False, "note": "Methodology manifest is missing."}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _progress_from(info: CultivationUpdate | None) -> float | None:
    if info is None:
        return None
    if info.cultivation_progress is not None:
        return info.cultivation_progress
    if info.achieved_ha is not None and info.target_ha not in (None, 0):
        return info.achieved_ha / info.target_ha
    return None


def _vectors() -> list[dict]:
    path = MANIFEST.parent / "live_vectors.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _from_model(match: dict, requested: float | None, manifest: dict) -> tuple[dict, bool, str]:
    """Call the saved horizon-1 LightGBM boosters. Change cultivation_progress only when that column exists."""
    import math

    import joblib
    import pandas as pd

    vector = dict(match["features"])
    applied = False
    note = ""
    if requested is not None and "cultivation_progress" in vector and manifest.get("progress_in_model"):
        vector["cultivation_progress"] = requested
        applied = True
        note = "The saved LightGBM model was scored again with the supplied cultivation progress. It was not retrained."
    elif requested is not None:
        note = (
            "latest_cultivation_information was received, but cultivation_progress is not a column in the trained model, "
            "so the price was not changed."
        )
    else:
        note = "Price is the saved horizon-1 LightGBM forecast."
    model_dir = Path(manifest["model_dir"])
    feature_set = manifest["feature_set"]
    origin = float(match["origin_price"])
    prices = {}
    for label, key in (("q50", "predicted_price"), ("q05", "lower_price"), ("q95", "upper_price")):
        bundle = joblib.load(model_dir / f"lightgbm_{feature_set}_h1_{label}.joblib")
        columns = bundle["columns"]
        row = pd.DataFrame([{name: vector.get(name) for name in columns}])
        raw = float(bundle["model"].predict(row)[0])
        prices[key] = origin * math.exp(raw)
    qhat = 0.0
    cqr_path = MANIFEST.parent / "cqr.json"
    if cqr_path.exists():
        cqr = json.loads(cqr_path.read_text(encoding="utf-8"))
        qhat = float((cqr.get("qhat_by_horizon") or {}).get("1") or 0.0)
    prices["lower_price"] = max(0.01, prices["lower_price"] - qhat)
    prices["upper_price"] = prices["upper_price"] + qhat
    prices["cultivation_progress"] = vector.get("cultivation_progress")
    return prices, applied, note


def forecast_one(request: ForecastRequest) -> ForecastResponse:
    manifest = _manifest()
    crop = request.crop.strip().lower()
    market = request.market.strip().lower().replace(" ", "_")
    requested = _progress_from(request.latest_cultivation_information)
    vector_row = next(
        (row for row in _vectors() if str(row["crop"]).lower() == crop and str(row["market"]).lower() == market),
        None,
    )
    stored = next(
        (row for row in _live() if str(row["crop"]).lower() == crop and str(row["market"]).lower() == market),
        None,
    )
    if vector_row is not None and manifest.get("model_dir"):
        try:
            prices, applied, note = _from_model(vector_row, requested, manifest)
            progress = prices.get("cultivation_progress")
            return ForecastResponse(
                crop=crop,
                market=market,
                forecast_week=str(vector_row.get("forecast_week") or request.forecast_week),
                predicted_price=round(prices["predicted_price"], 2),
                lower_price=round(prices["lower_price"], 2),
                upper_price=round(prices["upper_price"], 2),
                coverage_level=0.9,
                cultivation_progress=None if progress is None else float(progress),
                model_version=(stored or {}).get("model_version"),
                applied_cultivation_update=applied,
                note=note,
            )
        except Exception as exc:
            note = f"Saved model could not be scored ({exc})."
            if stored is None:
                return ForecastResponse(crop=crop, market=market, forecast_week=request.forecast_week, note=note)
    if stored is None:
        return ForecastResponse(
            crop=crop,
            market=market,
            forecast_week=request.forecast_week,
            note="No trained forecast is stored for this crop and market.",
        )
    note = str(stored.get("note") or manifest.get("note") or "")
    if requested is not None and not manifest.get("progress_in_model"):
        note = "latest_cultivation_information was not applied. " + note
    progress = stored.get("cultivation_progress")
    return ForecastResponse(
        crop=str(stored["crop"]),
        market=str(stored["market"]),
        forecast_week=str(stored.get("forecast_week") or request.forecast_week),
        predicted_price=stored.get("predicted_price"),
        lower_price=stored.get("lower_price"),
        upper_price=stored.get("upper_price"),
        coverage_level=float(stored.get("coverage_level") or 0.9),
        cultivation_progress=None if progress is None else float(progress),
        model_version=stored.get("model_version"),
        applied_cultivation_update=False,
        note=note,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "progress_in_model": bool(_manifest().get("progress_in_model"))}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    return forecast_one(request)


@app.post("/forecast/batch", response_model=list[ForecastResponse])
def forecast_batch(items: list[ForecastRequest]) -> list[ForecastResponse]:
    return [forecast_one(item) for item in items]
