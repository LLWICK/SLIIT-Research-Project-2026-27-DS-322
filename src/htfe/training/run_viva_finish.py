"""Attribute the Experiment B gain and save the B boosters the demo calls.

Usage from prototype/member2_ForecastingEngine:
  python -u -m training.run_viva_finish

LightGBM only. Hyperparameters are the frozen Experiment B values, so the
only change across arms is the feature list. This does not retrain A, B, C,
or S, and it does not rewrite their prediction files.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from htfe.training.config import ARTIFACTS, EVAL_MARKETS, HORIZONS, STUDIO_DATA
from htfe.training.evaluate import metrics_tables
from htfe.features.sets import (
    EXTENT_FEATURES,
    MACRO_FEATURES,
    PRICE_HISTORY,
    SEASON_FEATURES,
    WEATHER_FEATURES,
    historical_columns,
)
from htfe.training.run_experiment_a import EVAL_IDS, _design, _eligible, _score_frame
from htfe.training.run_experiment_b import _prepare_table
from htfe.models.trees import _fit_lgbm

B_DIR = ARTIFACTS / "experiment_b"
A_DIR = ARTIFACTS / "experiment_a"
OUT = ARTIFACTS / "experiment_b_attribution"
MODEL_DIR = B_DIR / "models"
from htfe.config import OUTPUTS
REPO = OUTPUTS
DOC = OUTPUTS / "notes" / "Experiment-B-Attribution-IT23415836.md"
NOTE = (
    "Experiment B LightGBM. Observed monthly cultivation progress is not a column in this model, "
    "so a cultivation update is not applied. No hectares were invented."
)


def _params() -> dict:
    manifest = json.loads((B_DIR / "manifest.json").read_text(encoding="utf-8"))
    return dict(manifest["optuna"]["lightgbm"])


def _arms(table: pd.DataFrame) -> dict[str, list[str]]:
    base = historical_columns(table)
    return {
        "weather": base + list(WEATHER_FEATURES),
        "macro": base + list(MACRO_FEATURES),
        "extent": base + list(EXTENT_FEATURES),
        "B": base + list(WEATHER_FEATURES) + list(MACRO_FEATURES) + list(EXTENT_FEATURES),
    }


def _fit_arm(table: pd.DataFrame, columns: list[str], params: dict, feature_set: str) -> tuple[pd.DataFrame, list[dict], Path | None]:
    parts = []
    live: list[dict] = []
    saved_h4: Path | None = None
    for horizon in HORIZONS:
        subset = _eligible(table, horizon)
        design, names = _design(subset, columns)
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
            if feature_set == "B":
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                path = MODEL_DIR / f"lightgbm_B_h{horizon}_{label}.joblib"
                joblib.dump(
                    {"model": model, "columns": names, "horizon": horizon, "alpha": alpha, "feature_set": "B"},
                    path,
                )
                if horizon == 4 and label == "q50":
                    saved_h4 = path
        point = np.maximum(origin * np.exp(np.asarray(fitted["q50"].predict(x_score), dtype=float)), 0.01)
        low = np.maximum(origin * np.exp(np.asarray(fitted["q05"].predict(x_score), dtype=float)), 0.01)
        high = np.maximum(origin * np.exp(np.asarray(fitted["q95"].predict(x_score), dtype=float)), 0.01)
        frame = _score_frame(score, point, low, high, feature_set)
        frame["feature_set"] = feature_set
        parts.append(frame)
        if feature_set == "B" and horizon == 1:
            indexed = score.copy()
            indexed["_pos"] = np.arange(len(indexed))
            last_rows = indexed[indexed["split"].eq("test")].sort_values("forecast_issue_week").groupby(["crop_id", "market_id"], observed=True).tail(1)
            for _, row in last_rows.iterrows():
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
                        "features": vector,
                    }
                )
    return pd.concat(parts, ignore_index=True), live, saved_h4


def _mae(frame: pd.DataFrame, horizon: int) -> float:
    part = frame[frame["split"].eq("test") & frame["horizon"].eq(horizon)]
    return float(np.mean(np.abs(part["y_true"] - part["y_pred"])))


def _saved_mae(path: Path, model: str) -> dict[int, float]:
    metrics = pd.read_csv(path)
    test = metrics[metrics["slice"].eq("test") & metrics["model"].eq(model) & metrics["target_form"].eq("price")]
    return {int(row.horizon): float(row.mae) for row in test.itertuples()}


def _shap(model_path: Path, table: pd.DataFrame, columns: list[str]) -> list[dict]:
    import shap

    bundle = joblib.load(model_path)
    subset = _eligible(table, 4)
    design, names = _design(subset, columns)
    test = subset["split"].eq("test") & subset["market_id"].isin(EVAL_IDS)
    sample = design.loc[test]
    if len(sample) > 400:
        sample = sample.sample(400, random_state=42)
    values = shap.TreeExplainer(bundle["model"]).shap_values(sample)
    mean_abs = np.abs(np.asarray(values)).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    peak = float(mean_abs[order[0]]) if len(order) else 1.0
    rows = []
    for index in order[:12]:
        rows.append(
            {
                "feature": names[int(index)],
                "importance": round(float(mean_abs[int(index)] / peak), 3) if peak else 0.0,
                "std": 0.0,
            }
        )
    return rows


def _write_doc(rows: list[dict]) -> None:
    lines = [
        "# What reduced Experiment B's error",
        "",
        "LightGBM only. The hyperparameters are the frozen Experiment B values, so each arm changes the columns and not the model search.",
        "The test is 2024–2025. Score markets are Dambulla, Colombo, Meegoda, and Nuwara Eliya.",
        "These arms are not Experiment C. Cultivation progress is not a column.",
        "",
        "| Arm | Added to price and season | 4-week MAE | Change vs A | Change vs full B |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['adds']} | {row['mae_h4']:.2f} | {row['vs_a']:+.2f}% | {row['vs_b']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "A negative change versus A means a lower error than price and season alone.",
            "The full B row is a refit with the saved hyperparameters. The official B metrics remain in `artifacts/experiment_b`.",
            "",
        ]
    )
    DOC.write_text("\n".join(lines), encoding="utf-8")


def _studio(live: list[dict], importance: list[dict], official_b: dict[int, float]) -> None:
    cqr = json.loads((B_DIR / "cqr.json").read_text(encoding="utf-8"))
    qhat = float(cqr["qhat_by_model"]["lightgbm"]["1"])
    score_ids = {market.lower().replace(" ", "_") for market in EVAL_MARKETS}
    live = [row for row in live if row["market"] in score_ids]
    live = sorted(live, key=lambda item: (item["crop"], item["market"]))
    cards = []
    for row in live:
        cards.append(
            {
                "crop": row["crop"],
                "market": row["market"],
                "forecast_week": row["forecast_week"],
                "cultivation_intensity": None,
                "cultivation_progress": None,
                "predicted_price": None,
                "lower_price": None,
                "upper_price": None,
                "coverage_level": 0.9,
                "commitment_source": "unavailable_no_district_monthly_series",
                "progress_in_model": False,
                "model_version": "experiment_b",
                "note": NOTE,
                "origin_price": row["origin_price"],
            }
        )
    # Prices come from the saved boosters so the studio and the API match.
    import math

    bundles = {
        label: joblib.load(MODEL_DIR / f"lightgbm_B_h1_{label}.joblib")
        for label in ("q05", "q50", "q95")
    }
    priced = []
    for vector_row, card in zip(live, cards):
        prices = {}
        for label, key in (("q50", "predicted_price"), ("q05", "lower_price"), ("q95", "upper_price")):
            bundle = bundles[label]
            design = pd.DataFrame([{name: vector_row["features"].get(name) for name in bundle["columns"]}])
            design = design.apply(pd.to_numeric, errors="coerce")
            raw = float(bundle["model"].predict(design)[0])
            prices[key] = vector_row["origin_price"] * math.exp(raw)
        card["predicted_price"] = prices["predicted_price"]
        card["lower_price"] = max(0.01, prices["lower_price"] - qhat)
        card["upper_price"] = prices["upper_price"] + qhat
        card.pop("origin_price", None)
        priced.append(card)
    (STUDIO_DATA / "live_forecast.json").write_text(json.dumps(priced, indent=2), encoding="utf-8")
    (B_DIR / "live_vectors.json").write_text(json.dumps(live, indent=2), encoding="utf-8")
    (STUDIO_DATA / "importance.json").write_text(json.dumps(importance, indent=2), encoding="utf-8")
    service = {
        "progress_in_model": False,
        "model_dir": str(MODEL_DIR),
        "feature_set": "B",
        "note": NOTE,
        "model_version": "experiment_b",
    }
    (B_DIR / "service_manifest.json").write_text(json.dumps(service, indent=2), encoding="utf-8")
    a_metrics = pd.read_csv(A_DIR / "metrics_by_model_horizon.csv")
    b_metrics = pd.read_csv(B_DIR / "metrics_by_model_horizon.csv")

    def arm_block(frame: pd.DataFrame, model: str) -> dict:
        test = frame[frame["slice"].eq("test") & frame["model"].eq(model) & frame["target_form"].eq("price")]
        by_horizon = []
        for row in test.sort_values("horizon").itertuples():
            by_horizon.append(
                {
                    "horizon": int(row.horizon),
                    "n_scored": int(row.n_scored),
                    "mae": round(float(row.mae), 2),
                    "rmse": round(float(row.rmse), 2),
                    "mape": round(float(row.mape), 2),
                    "pinball": None if pd.isna(row.pinball) else round(float(row.pinball), 2),
                    "picp": None if pd.isna(row.picp) else round(float(row.picp), 2),
                    "interval_width": None if pd.isna(row.interval_width) else round(float(row.interval_width), 2),
                }
            )
        return by_horizon

    a_rows = arm_block(a_metrics, "lightgbm")
    b_rows = arm_block(b_metrics, "lightgbm")
    ablation = {
        "status": "partial",
        "note": (
            "A and B are the reported experiments. C was not trained because no district-month "
            "target and achieved file was found. The source split of B is in Experiment-B-Attribution."
        ),
        "by_horizon": {"A": a_rows, "B": b_rows},
        "A": {"metrics": a_rows[2], "feature_set": "A"},
        "B": {"metrics": b_rows[2], "feature_set": "B"},
        "C": None,
        "official_b_mae_by_horizon": {str(horizon): round(value, 2) for horizon, value in official_b.items()},
    }
    (STUDIO_DATA / "ablation.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    (STUDIO_DATA / "commitment_weeks.json").write_text("[]", encoding="utf-8")
    meta_path = STUDIO_DATA / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["version"] = "experiment_b"
    meta["feature_set"] = "B"
    meta["progress_in_model"] = False
    meta["commitment_source"] = "unavailable_no_district_monthly_series"
    meta["n_scored_test"] = int(b_rows[0]["n_scored"]) if b_rows else meta.get("n_scored_test")
    meta["interval_note"] = (
        "Intervals are split-conformal on 2023 only. Test coverage is below the nominal 90%."
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    models_path = STUDIO_DATA / "models.json"
    models = json.loads(models_path.read_text(encoding="utf-8"))
    models["lightgbm"]["qhat"] = qhat
    models["lightgbm"]["qhat_by_horizon"] = {
        str(horizon): float(value) for horizon, value in cqr["qhat_by_model"]["lightgbm"].items()
    }
    models["lightgbm"]["note"] = meta["interval_note"]
    models_path.write_text(json.dumps(models, indent=2), encoding="utf-8")


def _live_from_table(table: pd.DataFrame, columns: list[str]) -> list[dict]:
    subset = _eligible(table, 1)
    design, names = _design(subset, columns)
    score_mask = subset["market_id"].isin(EVAL_IDS) & subset["split"].eq("test")
    indexed = subset.loc[score_mask].copy()
    indexed["_pos"] = design.loc[score_mask].index
    last_rows = indexed.sort_values("forecast_issue_week").groupby(["crop_id", "market_id"], observed=True).tail(1)
    live = []
    for _, row in last_rows.iterrows():
        vector = {}
        for name in names:
            value = design.loc[row["_pos"], name]
            vector[name] = None if pd.isna(value) else float(value)
        live.append(
            {
                "crop": str(row["crop_id"]),
                "market": str(row["market_id"]),
                "origin_price": float(row["last_observed_price_lkr_kg"]),
                "forecast_week": pd.Timestamp(row["target_week"]).strftime("%Y-%m-%d"),
                "features": vector,
            }
        )
    return live


def main() -> None:
    table = _prepare_table()
    if (OUT / "summary.json").exists() and (OUT / "shap_h4.json").exists() and (MODEL_DIR / "lightgbm_B_h1_q50.joblib").exists():
        print("attribution already fitted; writing the demo files only", flush=True)
        columns = json.loads((B_DIR / "manifest.json").read_text(encoding="utf-8"))["features"]
        importance = json.loads((OUT / "shap_h4.json").read_text(encoding="utf-8"))
        b_mae = _saved_mae(B_DIR / "metrics_by_model_horizon.csv", "lightgbm")
        _studio(_live_from_table(table, columns), importance, b_mae)
        print(f"wrote {B_DIR / 'service_manifest.json'}", flush=True)
        return
    params = _params()
    arms = _arms(table)
    if arms["B"] != json.loads((B_DIR / "manifest.json").read_text(encoding="utf-8"))["features"]:
        raise RuntimeError("attribution B columns do not match the saved Experiment B feature list")
    a_mae = _saved_mae(A_DIR / "metrics_by_model_horizon.csv", "lightgbm")
    b_mae = _saved_mae(B_DIR / "metrics_by_model_horizon.csv", "lightgbm")
    summaries = []
    predictions = []
    live: list[dict] = []
    h4_path: Path | None = None
    for name, columns in arms.items():
        print(f"fitting {name}", flush=True)
        frame, arm_live, saved = _fit_arm(table, columns, params, name)
        predictions.append(frame)
        if name == "B":
            live = arm_live
            h4_path = saved
        mae4 = _mae(frame, 4)
        summaries.append(
            {
                "arm": name,
                "adds": {
                    "weather": "origin weather",
                    "macro": "diesel, USD/LKR, inflation",
                    "extent": "lagged finished-season extent",
                    "B": "weather + macros + lagged extent",
                }[name],
                "mae_h4": mae4,
                "vs_a": (mae4 - a_mae[4]) / a_mae[4] * 100,
                "vs_b": (mae4 - b_mae[4]) / b_mae[4] * 100,
                "mae_by_horizon": {str(horizon): round(_mae(frame, horizon), 2) for horizon in HORIZONS},
            }
        )
        print(f"{name} 4-week MAE {mae4:.2f}", flush=True)
    if h4_path is None:
        raise RuntimeError("horizon-4 Experiment B model was not saved")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "predictions.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    _write_doc(summaries)
    print("shap", flush=True)
    importance = _shap(h4_path, table, arms["B"])
    (OUT / "shap_h4.json").write_text(json.dumps(importance, indent=2), encoding="utf-8")
    _studio(live, importance, b_mae)
    by_model, _series = metrics_tables(pd.concat(predictions, ignore_index=True))
    by_model.to_csv(OUT / "metrics_by_model_horizon.csv", index=False)
    print(f"wrote {DOC}", flush=True)
    print(f"wrote {B_DIR / 'service_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
