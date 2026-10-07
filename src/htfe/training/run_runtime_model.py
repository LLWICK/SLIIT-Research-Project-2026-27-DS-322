"""Runtime forecast: one saved model, one cultivation-progress column.

The column is trained on the labelled synthetic calendar curve so a later
commitment record can change it. This is not Experiment C and it is not
written into the A/B accuracy files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from htfe.training.align_experiment_a import _cqr
from htfe.training.config import ARTIFACTS, EVAL_MARKETS, STUDIO_DATA
from htfe.features.sets import feature_sets
from htfe.training.run_experiment_a import EVAL_IDS, _design, _eligible, _score_frame
from htfe.training.run_experiment_b import OUT as B_DIR
from htfe.training.run_experiment_b import _prepare_table
from htfe.models.trees import _fit_lgbm

OUT = ARTIFACTS / "runtime_model"
MODEL_DIR = OUT / "models"
SYNTHETIC_NAME = "cultivation_synthetic.parquet"
PROGRESS = "cultivation_progress"
NOTE = (
    "Runtime model. cultivation_progress was trained on the labelled synthetic calendar curve "
    "so a new achieved/target record can be scored without retraining. "
    "This is the update mechanism, not the Experiment A/B accuracy result, and it is not Experiment C."
)


def _origin_map() -> dict:
    from htfe.config import origin_map_path
    path = origin_map_path()
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _market_id(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _synthetic_progress(table: pd.DataFrame) -> pd.DataFrame:
    from htfe.config import built_file

    path = built_file(SYNTHETIC_NAME)
    synthetic = pd.read_parquet(path)
    synthetic = synthetic[synthetic["cultivation_scenario"].eq("typical")].copy()
    if not bool(synthetic["cultivation_is_synthetic"].astype(bool).all()):
        raise AssertionError("runtime training rows must be marked synthetic")
    synthetic["crop_key"] = synthetic["crop"].astype(str).str.strip().str.lower()
    synthetic["district_key"] = synthetic["origin_district"].astype(str).str.strip().str.casefold()
    synthetic["week_start"] = pd.to_datetime(synthetic["week_start"])
    if synthetic.duplicated(["crop_key", "district_key", "week_start"]).any():
        raise AssertionError("typical synthetic progress is not unique by crop, district, and week")
    links = []
    for crop, markets in _origin_map().items():
        for market, spec in markets.items():
            for district in spec.get("supply_districts") or []:
                links.append(
                    {
                        "crop_key": str(crop).strip().lower(),
                        "market_id": _market_id(market),
                        "district_key": str(district).strip().casefold(),
                    }
                )
    joined = synthetic.merge(pd.DataFrame(links), on=["crop_key", "district_key"], how="inner")
    grouped = joined.groupby(["crop_key", "market_id", "week_start"], as_index=False).agg(
        area=("cultivation_area_asof", "sum"),
        reference=("cultivation_reference_area", "sum"),
    )
    grouped[PROGRESS] = grouped["area"] / grouped["reference"].replace(0, np.nan)
    grouped["cultivation_is_synthetic"] = True
    return grouped


def _with_progress(table: pd.DataFrame) -> pd.DataFrame:
    progress = _synthetic_progress(table).rename(columns={"market_id": "market_key", "week_start": "join_week"})
    frame = table.copy()
    before = len(frame)
    frame["crop_key"] = frame["crop_id"].astype(str).str.strip().str.lower()
    frame["market_key"] = frame["market_id"].astype(str).str.strip().str.lower()
    frame["join_week"] = pd.to_datetime(frame["forecast_issue_week"])
    for name in (PROGRESS, "cultivation_is_synthetic"):
        if name in frame.columns:
            frame = frame.drop(columns=[name])
    merged = frame.merge(
        progress[["crop_key", "market_key", "join_week", PROGRESS, "cultivation_is_synthetic"]],
        on=["crop_key", "market_key", "join_week"],
        how="left",
    )
    if len(merged) != before:
        raise AssertionError("synthetic progress join duplicated training rows")
    merged["cultivation_is_synthetic"] = merged["cultivation_is_synthetic"].fillna(False).astype(bool)
    return merged


def _params() -> dict:
    manifest = json.loads((B_DIR / "manifest.json").read_text(encoding="utf-8"))
    return dict(manifest["optuna"]["lightgbm"])


def _fit(table: pd.DataFrame, columns: list[str], params: dict) -> tuple[pd.DataFrame, list[dict]]:
    parts = []
    live = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for horizon in (1, 2, 4, 8, 12):
        subset = _eligible(table, horizon)
        design, names = _design(subset, columns)
        if PROGRESS not in names:
            raise RuntimeError("cultivation_progress was not in the runtime design matrix")
        y = subset["y_logret"].to_numpy(dtype=float)
        train = subset["split"].eq("train").to_numpy()
        cal = subset["split"].eq("calibrate").to_numpy()
        x_train, y_train = design.iloc[np.flatnonzero(train)], y[train]
        x_cal, y_cal = design.iloc[np.flatnonzero(cal)], y[cal]
        score_mask = subset["market_id"].isin(EVAL_IDS) & subset["split"].isin(["calibrate", "test"])
        score = subset.loc[score_mask]
        x_score = design.loc[score_mask]
        origin = score["last_observed_price_lkr_kg"].to_numpy(dtype=float)
        fitted = {}
        for alpha, label in ((0.05, "q05"), (0.50, "q50"), (0.95, "q95")):
            model, _curve = _fit_lgbm(x_train, y_train, x_cal, y_cal, alpha, params)
            fitted[label] = model
            joblib.dump(
                {
                    "model": model,
                    "columns": names,
                    "horizon": horizon,
                    "alpha": alpha,
                    "feature_set": "runtime",
                    "cultivation_is_synthetic": True,
                },
                MODEL_DIR / f"lightgbm_runtime_h{horizon}_{label}.joblib",
            )
        point = np.maximum(origin * np.exp(np.asarray(fitted["q50"].predict(x_score), dtype=float)), 0.01)
        low = np.maximum(origin * np.exp(np.asarray(fitted["q05"].predict(x_score), dtype=float)), 0.01)
        high = np.maximum(origin * np.exp(np.asarray(fitted["q95"].predict(x_score), dtype=float)), 0.01)
        frame = _score_frame(score, point, low, high, "lightgbm")
        frame["feature_set"] = "runtime"
        parts.append(frame)
        if horizon == 1:
            indexed = score.copy()
            indexed["_pos"] = np.arange(len(indexed))
            last_rows = (
                indexed[indexed["split"].eq("test")]
                .sort_values("forecast_issue_week")
                .groupby(["crop_id", "market_id"], observed=True)
                .tail(1)
            )
            for _, row in last_rows.iterrows():
                if str(row["market_id"]) not in {_market_id(market) for market in EVAL_MARKETS}:
                    continue
                position = int(row["_pos"])
                vector = {}
                for name in names:
                    value = x_score.iloc[position][name]
                    vector[name] = None if pd.isna(value) else float(value)
                live.append(
                    {
                        "crop": str(row["crop_id"]),
                        "market": str(row["market_id"]),
                        "origin_price": float(row["last_observed_price_lkr_kg"]),
                        "forecast_week": pd.Timestamp(row["target_week"]).strftime("%Y-%m-%d"),
                        "forecast_issue_week": pd.Timestamp(row["forecast_issue_week"]).strftime("%Y-%m-%d"),
                        "features": vector,
                    }
                )
    return pd.concat(parts, ignore_index=True), live


def _same_base_row(live: list[dict]) -> None:
    """The stored Experiment B feature values stay as they are. Only cultivation_progress is added."""
    stored = json.loads((B_DIR / "live_vectors.json").read_text(encoding="utf-8"))
    for row in live:
        match = next(item for item in stored if item["crop"] == row["crop"] and item["market"] == row["market"])
        for key, value in match["features"].items():
            got = row["features"][key]
            if value is None or got is None:
                if not (value is None and got is None):
                    raise AssertionError(f"{row['crop']} {row['market']} rebuilt {key}")
            elif abs(float(value) - float(got)) > 1e-5:
                raise AssertionError(f"{row['crop']} {row['market']} rebuilt {key}")
        if PROGRESS not in row["features"]:
            raise AssertionError("runtime feature row has no cultivation_progress column")


def _accuracy_fingerprint() -> dict[str, str]:
    paths = [
        ARTIFACTS / "experiment_a" / "metrics_by_model_horizon.csv",
        ARTIFACTS / "experiment_b" / "metrics_by_model_horizon.csv",
        ARTIFACTS / "experiment_b" / "ab_comparison.csv",
        STUDIO_DATA / "evaluation.json",
    ]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _qhat_payload(predictions: pd.DataFrame) -> dict:
    _rows, payload = _cqr(predictions)
    light = payload["qhat_by_model"]["lightgbm"]
    payload["qhat_by_horizon"] = light
    payload["model"] = "runtime_lightgbm"
    payload["not_an_accuracy_result"] = True
    payload["by_model_horizon"] = []
    payload["predictions_file_unchanged"] = False
    payload["note"] = NOTE + " qhat is this model's own 2023 adjustment. Do not quote these rows as Experiment A, B, or C accuracy."
    return payload


def main() -> None:
    before = _accuracy_fingerprint()
    table = _with_progress(_prepare_table())
    if not bool(table.loc[table[PROGRESS].notna(), "cultivation_is_synthetic"].all()):
        raise AssertionError("a non-null runtime progress value was not marked synthetic")
    base = feature_sets(table)["B"]
    if not base:
        raise RuntimeError("Experiment B columns are not available, so the runtime model was not trained")
    columns = list(base) + [PROGRESS]
    filled = int(table[PROGRESS].notna().sum())
    if filled == 0:
        raise AssertionError("the typical synthetic curve did not match any training week")
    print(f"runtime rows with synthetic progress {filled}", flush=True)
    predictions, live = _fit(table, columns, _params())
    _same_base_row(live)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    cqr = _qhat_payload(predictions)
    (OUT / "cqr.json").write_text(json.dumps(cqr, indent=2), encoding="utf-8")
    (OUT / "live_vectors.json").write_text(json.dumps(live, indent=2), encoding="utf-8")
    manifest = {
        "progress_in_model": True,
        "model_dir": str(MODEL_DIR),
        "feature_set": "runtime",
        "model_version": "runtime_synthetic_progress",
        "note": NOTE,
        "cultivation_source": "synthetic_calendar",
        "cultivation_is_synthetic": True,
        "not_experiment_c": True,
    }
    (OUT / "service_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    store = OUT / "commitment_records.csv"
    if not store.exists():
        pd.DataFrame(columns=["crop", "market", "report_date", "target_ha", "achieved_ha", "source"]).to_csv(store, index=False)
    from htfe.api.runtime import mechanism_check

    report = mechanism_check()
    (OUT / "mechanism_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _studio(live, cqr)
    if _accuracy_fingerprint() != before:
        raise AssertionError("Experiment A/B accuracy files changed while training the runtime model")
    print(f"wrote {OUT / 'service_manifest.json'}", flush=True)
    print(json.dumps({key: report[key] for key in ("future_ignored", "ratio_above_one", "price_changed")}, indent=2), flush=True)


def _studio(live: list[dict], cqr: dict) -> None:
    from htfe.api.runtime import score_vector

    qhat = float(cqr["qhat_by_horizon"]["1"])
    cards = []
    for row in sorted(live, key=lambda item: (item["crop"], item["market"])):
        prices = score_vector(row, progress=row["features"].get(PROGRESS))
        cards.append(
            {
                "crop": row["crop"],
                "market": row["market"],
                "forecast_week": row["forecast_week"],
                "forecast_issue_week": row["forecast_issue_week"],
                "cultivation_intensity": None,
                "cultivation_progress": row["features"].get(PROGRESS),
                "predicted_price": round(prices["predicted_price"], 2),
                "lower_price": round(max(0.01, prices["lower_price"] - qhat), 2),
                "upper_price": round(prices["upper_price"] + qhat, 2),
                "coverage_level": 0.9,
                "commitment_source": "synthetic_calendar_runtime_model",
                "progress_in_model": True,
                "model_version": "runtime_synthetic_progress",
                "note": NOTE,
            }
        )
    (STUDIO_DATA / "live_forecast.json").write_text(json.dumps(cards, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
