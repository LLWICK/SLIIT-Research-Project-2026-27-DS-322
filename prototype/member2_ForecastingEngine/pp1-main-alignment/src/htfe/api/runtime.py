"""Score the saved runtime model. A new record changes only cultivation progress."""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import pandas as pd

from htfe.training.config import ARTIFACTS

OUT = ARTIFACTS / "runtime_model"
STORE = OUT / "commitment_records.csv"
VECTORS = OUT / "live_vectors.json"
MANIFEST = OUT / "service_manifest.json"
PROGRESS = "cultivation_progress"
COLUMNS = ["crop", "market", "report_date", "target_ha", "achieved_ha", "source"]


def _manifest() -> dict:
    if not MANIFEST.exists():
        return {"progress_in_model": False}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _vectors() -> list[dict]:
    if not VECTORS.exists():
        return []
    return json.loads(VECTORS.read_text(encoding="utf-8"))


def _find(crop: str, market: str) -> dict | None:
    crop_id = crop.strip().lower()
    market_id = market.strip().lower().replace(" ", "_")
    return next(
        (row for row in _vectors() if row["crop"] == crop_id and row["market"] == market_id),
        None,
    )


def read_records() -> pd.DataFrame:
    if not STORE.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(STORE)
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    frame["report_date"] = pd.to_datetime(frame["report_date"])
    frame["target_ha"] = pd.to_numeric(frame["target_ha"], errors="coerce")
    frame["achieved_ha"] = pd.to_numeric(frame["achieved_ha"], errors="coerce")
    return frame


def write_record(crop: str, market: str, report_date: str, target_ha: float, achieved_ha: float, source: str) -> None:
    frame = read_records()
    crop_id = crop.strip().lower()
    market_id = market.strip().lower().replace(" ", "_")
    if source == "studio" and not frame.empty:
        frame = frame.loc[
            ~(
                frame["crop"].astype(str).str.lower().eq(crop_id)
                & frame["market"].astype(str).str.lower().str.replace(" ", "_", regex=False).eq(market_id)
                & frame["source"].astype(str).eq("studio")
            )
        ]
    row = {
        "crop": crop_id,
        "market": market_id,
        "report_date": pd.Timestamp(report_date),
        "target_ha": float(target_ha),
        "achieved_ha": float(achieved_ha),
        "source": source,
    }
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(STORE, index=False)


def raw_progress(achieved_ha: float, target_ha: float) -> tuple[float, bool]:
    if target_ha == 0:
        raise ValueError("target hectares cannot be zero")
    progress = float(achieved_ha) / float(target_ha)
    return progress, progress > 1


def latest_progress(crop: str, market: str, issue_date: str, records: pd.DataFrame | None = None) -> dict:
    """Use the newest record dated on or before the forecast Monday. Do not clip ratios above 1."""
    frame = read_records() if records is None else records.copy()
    if frame.empty:
        return {"progress": None, "target_exceeded": None, "report_date": None, "used": False}
    frame["crop"] = frame["crop"].astype(str).str.lower()
    frame["market"] = frame["market"].astype(str).str.lower().str.replace(" ", "_", regex=False)
    frame["report_date"] = pd.to_datetime(frame["report_date"])
    issue = pd.Timestamp(issue_date)
    part = frame[
        frame["crop"].eq(crop.strip().lower())
        & frame["market"].eq(market.strip().lower().replace(" ", "_"))
        & frame["report_date"].le(issue)
    ]
    if part.empty:
        return {"progress": None, "target_exceeded": None, "report_date": None, "used": False}
    row = part.sort_values("report_date").iloc[-1]
    progress, exceeded = raw_progress(float(row["achieved_ha"]), float(row["target_ha"]))
    return {
        "progress": progress,
        "target_exceeded": exceeded,
        "report_date": pd.Timestamp(row["report_date"]).strftime("%Y-%m-%d"),
        "used": True,
    }


def feature_row(vector: dict, progress: float | None) -> dict:
    """Copy the stored feature row and replace only cultivation progress."""
    features = dict(vector["features"])
    features[PROGRESS] = None if progress is None or (isinstance(progress, float) and math.isnan(progress)) else float(progress)
    changed = [name for name in features if features[name] != vector["features"].get(name)]
    if changed not in ([], [PROGRESS]):
        raise AssertionError(f"feature builder changed {changed}")
    return features


def score_vector(vector: dict, progress: float | None) -> dict:
    """Return raw quantile prices. The caller applies the 2023 conformal adjustment."""
    manifest = _manifest()
    model_dir = Path(manifest["model_dir"])
    feature_set = manifest["feature_set"]
    features = feature_row(vector, progress if progress is not None else vector["features"].get(PROGRESS))
    origin = float(vector["origin_price"])
    prices = {}
    for label, key in (("q50", "predicted_price"), ("q05", "lower_price"), ("q95", "upper_price")):
        bundle = joblib.load(model_dir / f"lightgbm_{feature_set}_h1_{label}.joblib")
        row = pd.DataFrame([{name: features.get(name) for name in bundle["columns"]}])
        row = row.apply(pd.to_numeric, errors="coerce")
        raw = float(bundle["model"].predict(row)[0])
        prices[key] = origin * math.exp(raw)
    prices[PROGRESS] = features.get(PROGRESS)
    return prices


def _qhat() -> float:
    path = OUT / "cqr.json"
    if not path.exists():
        return 0.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float((payload.get("qhat_by_horizon") or {}).get("1") or 0.0)


APPLIED_NOTE = (
    "Cultivation progress is the raw achieved/target ratio from the latest commitment record "
    "dated on or before the forecast Monday. The saved model was scored again and was not retrained. "
    "That column was trained on the labelled synthetic calendar curve, so this is the update mechanism, "
    "not the Experiment A/B accuracy result."
)


def score_progress(
    crop: str,
    market: str,
    achieved_ha: float | None = None,
    target_ha: float | None = None,
    report_date: str | None = None,
    progress: float | None = None,
    source: str = "runtime",
) -> dict:
    vector = _find(crop, market)
    if vector is None:
        raise KeyError(f"no runtime vector for {crop} {market}")
    issue = vector["forecast_issue_week"]
    dated_after = report_date is not None and pd.Timestamp(report_date) > pd.Timestamp(issue)
    if achieved_ha is not None and target_ha is not None and not dated_after:
        write_record(crop, market, report_date or issue, target_ha, achieved_ha, source)
    selected = latest_progress(crop, market, issue)
    exceeded = selected["target_exceeded"]
    used_date = selected["report_date"]
    if selected["used"]:
        chosen = float(selected["progress"])
        applied = True
        note = APPLIED_NOTE
        if exceeded:
            note = "The ratio is above 1 and was kept, not clipped. " + note
    elif progress is not None and not dated_after:
        chosen = float(progress)
        exceeded = chosen > 1
        used_date = report_date or issue
        applied = True
        note = APPLIED_NOTE
    else:
        chosen = vector["features"].get(PROGRESS)
        applied = False
        note = "Price is the saved runtime LightGBM forecast. No commitment record is dated on or before this forecast Monday."
    if dated_after:
        note = "A commitment record dated after the forecast Monday was not used. " + note
    prices = score_vector(vector, None if chosen is None or (isinstance(chosen, float) and math.isnan(chosen)) else float(chosen))
    qhat = _qhat()
    return {
        "predicted_price": round(prices["predicted_price"], 2),
        "lower_price": round(max(0.01, prices["lower_price"] - qhat), 2),
        "upper_price": round(prices["upper_price"] + qhat, 2),
        "cultivation_progress": None if chosen is None or (isinstance(chosen, float) and math.isnan(chosen)) else float(chosen),
        "cultivation_target_exceeded": exceeded,
        "report_date": used_date,
        "applied_cultivation_update": applied,
        "forecast_week": vector["forecast_week"],
        "forecast_issue_week": issue,
        "note": note,
        "model_version": "runtime_synthetic_progress",
    }


def _drop_check_rows() -> None:
    frame = read_records()
    if frame.empty:
        return
    frame = frame.loc[~frame["source"].astype(str).eq("check")]
    frame.to_csv(STORE, index=False)


def mechanism_check() -> dict:
    """Post two carrot/Dambulla records. The later one must not change the feature. 1.4 stays 1.4."""
    vector = _find("carrot", "dambulla")
    if vector is None:
        raise AssertionError("carrot dambulla runtime vector is missing")
    issue = pd.Timestamp(vector["forecast_issue_week"])
    issue_text = issue.strftime("%Y-%m-%d")
    future = (issue + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    earlier = (issue - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    stored = vector["features"].get(PROGRESS)
    try:
        write_record("carrot", "dambulla", future, 100, 10, "check")
        future_only = latest_progress("carrot", "dambulla", issue_text)
        if future_only["used"]:
            raise AssertionError("a commitment dated after the forecast Monday changed the feature")
        future_features = feature_row(vector, stored)
        if future_features[PROGRESS] != stored and not (future_features[PROGRESS] is None and stored is None):
            raise AssertionError("the future record replaced cultivation_progress")
        write_record("carrot", "dambulla", earlier, 100, 140, "check")
        selected = latest_progress("carrot", "dambulla", issue_text)
        if selected["report_date"] != earlier:
            raise AssertionError("the latest allowed record was not the one dated on or before the forecast Monday")
        if abs(float(selected["progress"]) - 1.4) > 1e-9 or not selected["target_exceeded"]:
            raise AssertionError("ratio above 1 was clipped or not flagged")
        kept = feature_row(vector, selected["progress"])
        if abs(float(kept[PROGRESS]) - 1.4) > 1e-9:
            raise AssertionError("feature builder did not keep the raw ratio")
        low = score_vector(vector, 0.2)
        high = score_vector(vector, 1.4)
        if abs(low["predicted_price"] - high["predicted_price"]) < 1e-6:
            raise AssertionError("changing cultivation progress did not change the saved model's price")
    finally:
        _drop_check_rows()
    return {
        "future_ignored": True,
        "ratio_above_one": 1.4,
        "price_changed": True,
        "low_progress_price": round(low["predicted_price"], 2),
        "high_progress_price": round(high["predicted_price"], 2),
        "forecast_issue_week": issue_text,
        "not_a_mape": True,
    }
