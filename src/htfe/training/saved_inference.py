"""Score frozen LightGBM bundles. This module does not fit models.

The checks reproduce one saved Experiment C median, score twelve horizon-1
rows (3 crops x 4 markets), and swap one frozen vector's cultivation progress
for observed national quantiles. Dynamic same-target reforecasting stays in
reforecast_c.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from htfe.config import OUTPUTS
from htfe.data.build.scope import CROPS, EVAL_MARKETS
from htfe.training.reforecast_c import apply_categories
from htfe.training.run_experiment_a import EVAL_IDS, _design

OUT = OUTPUTS / "experiment_c"
INFERENCE = OUT / "inference"
PROGRESS = "cultivation_progress_ratio_real"
EXCEEDED = "cultivation_target_exceeded_real"
SCORE_MARKETS = tuple(sorted(EVAL_IDS))
REPRODUCTION_TOLERANCE = 1e-4


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_bundles(model_dir: Path, feature_set: str, horizon: int) -> dict:
    import joblib

    bundles = {}
    for label in ("q05", "q50", "q95"):
        path = model_dir / f"lightgbm_{feature_set}_h{int(horizon)}_{label}.joblib"
        if not path.is_file():
            raise FileNotFoundError(path)
        bundles[label] = joblib.load(path)
    return bundles


def design_matrix(frame: pd.DataFrame, bundles: dict, spec: dict) -> pd.DataFrame:
    """Rebuild the training design. Category codes come from the saved manifest."""
    feature_columns = list(bundles["q50"]["feature_columns"])
    saved = list(bundles["q50"]["columns"])
    work = apply_categories(frame, spec)
    design, names = _design(work, feature_columns)
    if list(names) != saved:
        raise RuntimeError(
            "Rebuilt columns do not match the frozen model. "
            f"model has {len(saved)} columns and the rebuild has {len(names)}."
        )
    return design.loc[:, saved].astype("float64")


def predict_prices(design: pd.DataFrame, origin: np.ndarray, bundles: dict) -> dict[str, np.ndarray]:
    """Call already-fitted quantile models. There is no fit step."""
    columns = list(bundles["q50"]["columns"])
    matrix = design.loc[:, columns]
    prices = {}
    for label, bundle in bundles.items():
        log_return = np.asarray(bundle["model"].predict(matrix), dtype=float)
        prices[label] = np.maximum(origin * np.exp(log_return), 0.01)
    return prices


def interval_from_prices(q05: float, q50: float, q95: float, qhat: float) -> tuple[float, float, float]:
    """Ordered 5% and 95% tails, then the 2023 qhat. Same rule as metrics_c.csv."""
    lower = float(min(q05, q95) - qhat)
    upper = float(max(q05, q95) + qhat)
    return lower, float(q50), upper


def raw_quantile_crossed(q05: float, q50: float, q95: float) -> bool:
    return bool(q05 > q50 or q50 > q95)


def exceeded_flag(ratio: float) -> int:
    """1 when the ratio is above 1. Values above 1 are kept."""
    return int(float(ratio) > 1.0)


def _stamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed").dt.strftime("%Y-%m-%d")


def _attach_keys(frame: pd.DataFrame, crop: str, market: str, horizon: str) -> pd.DataFrame:
    out = frame.copy()
    out["crop_key"] = out[crop].astype(str).str.strip().str.lower()
    out["market_key"] = (
        out[market].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False)
    )
    out["issue"] = _stamp(out["forecast_issue_week"])
    out["target"] = _stamp(out["target_week"])
    out["horizon"] = out[horizon].astype(int)
    return out


def _prepare() -> dict:
    manifest = load_json(OUT / "manifest.json")
    cqr = load_json(OUT / "cqr.json")
    predictions = _attach_keys(pd.read_csv(OUT / "predictions_c.csv"), "crop", "market_id", "horizon")
    eligible = _attach_keys(pd.read_parquet(OUT / "c_eligible_table.parquet"), "crop_id", "market_id", "forecast_horizon_weeks")
    return {
        "manifest": manifest,
        "spec": manifest["categories"],
        "qhats": {str(key): float(value) for key, value in cqr["qhat_by_model"]["lightgbm"].items()},
        "predictions": predictions,
        "eligible": eligible,
    }


def _eligible_for(eligible: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    match = eligible[
        eligible["crop_key"].eq(row["crop_key"])
        & eligible["market_key"].eq(row["market_key"])
        & eligible["issue"].eq(row["issue"])
        & eligible["target"].eq(row["target"])
        & eligible["horizon"].eq(int(row["horizon"]))
    ]
    if len(match) != 1:
        raise RuntimeError(
            f"Eligible-row lookup failed for {row['crop_key']}/{row['market_key']}/"
            f"{row['issue']}/h{int(row['horizon'])}: found {len(match)}."
        )
    return match


def _score_row(row: pd.Series, eligible: pd.DataFrame, bundles: dict, spec: dict, qhat: float) -> dict:
    source = _eligible_for(eligible, row)
    design = design_matrix(source, bundles, spec)
    origin = source["last_observed_price_lkr_kg"].to_numpy(dtype=float)
    prices = predict_prices(design, origin, bundles)
    q05 = float(prices["q05"][0])
    q50 = float(prices["q50"][0])
    q95 = float(prices["q95"][0])
    lower, median, upper = interval_from_prices(q05, q50, q95, qhat)
    saved = float(row["y_pred"])
    difference = abs(saved - median)
    return {
        "design": design,
        "source": source,
        "q05": q05,
        "q50": q50,
        "q95": q95,
        "lower": lower,
        "median": median,
        "upper": upper,
        "saved_median": saved,
        "absolute_difference": difference,
        "raw_quantile_crossing": raw_quantile_crossed(q05, q50, q95),
        "qhat": qhat,
    }


def _lightgbm_c(predictions: pd.DataFrame, horizon: int, split: str) -> pd.DataFrame:
    return predictions[
        predictions["model"].astype(str).str.lower().eq("lightgbm")
        & predictions["feature_set"].astype(str).eq("C")
        & predictions["split"].astype(str).str.lower().eq(split)
        & predictions["horizon"].eq(int(horizon))
    ].copy()


def reproduce_one(bundle: dict) -> dict:
    """One horizon-4 test row. The median must match predictions_c.csv."""
    rows = _lightgbm_c(bundle["predictions"], 4, "test").sort_values(["issue", "crop_key", "market_key"])
    if rows.empty:
        raise RuntimeError("No horizon-4 Experiment C test row is available to reproduce.")
    chosen = rows.iloc[-1]
    qhat = float(bundle["qhats"]["4"])
    models = load_bundles(OUT / "models", "C", 4)
    scored = _score_row(chosen, bundle["eligible"], models, bundle["spec"], qhat)
    if scored["absolute_difference"] > REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"Saved C median differs by {scored['absolute_difference']}. No model was refit."
        )
    source = scored["source"].iloc[0]
    payload = {
        "check": "reproduce_one_saved_c_forecast",
        "fit_called": False,
        "crop": str(chosen["crop_key"]),
        "market": str(chosen["market_key"]),
        "forecast_issue_week": str(chosen["issue"]),
        "target_week": str(chosen["target"]),
        "horizon_weeks": 4,
        "cultivation_report_date": _date_or_none(source["cultivation_report_date"]),
        "cultivation_progress_ratio": float(source[PROGRESS]),
        "target_exceeded": int(float(source[EXCEEDED])),
        "last_observed_price": round(float(source["last_observed_price_lkr_kg"]), 2),
        "lower_price": round(scored["lower"], 2),
        "predicted_price": round(scored["median"], 2),
        "upper_price": round(scored["upper"], 2),
        "saved_median": round(scored["saved_median"], 6),
        "actual_price_historical_demo": round(float(chosen["y_true"]), 2),
        "reproduction_absolute_difference": scored["absolute_difference"],
        "raw_quantile_crossing": scored["raw_quantile_crossing"],
        "interval_rule": (
            "Lower and upper use the ordered 5% and 95% prices plus this run's 2023 qhat. "
            "The median is the frozen q50 prediction."
        ),
        "model": "lightgbm",
        "feature_set": "C",
    }
    return {"payload": payload, "scored": scored, "row": chosen, "models": models}


def _date_or_none(value) -> str | None:
    stamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(stamp):
        return None
    return pd.Timestamp(stamp).strftime("%Y-%m-%d")


def observed_national_ratios(crop: str) -> pd.Series:
    audit = pd.read_csv(OUT / "audit" / "doa_cultivation_progress_audited.csv")
    crop_key = audit["crop"].astype(str).str.strip().str.lower()
    scope = audit["scope_level"].fillna("").astype(str).str.strip().str.lower()
    ratio = pd.to_numeric(audit["cultivation_progress_ratio"], errors="coerce")
    observed = ratio[crop_key.eq(crop) & scope.eq("national")].dropna().astype(float)
    return observed


def sensitivity(reproduced: dict) -> pd.DataFrame:
    """Low, median, and high observed national ratios on one frozen horizon-4 vector."""
    scored = reproduced["scored"]
    crop = str(reproduced["payload"]["crop"])
    observed = observed_national_ratios(crop)
    if len(observed) < 3:
        raise RuntimeError(f"Not enough observed national ratios for {crop} to run the sensitivity check.")
    levels = {
        "observed_low_q25": float(observed.quantile(0.25)),
        "observed_median": float(observed.quantile(0.50)),
        "observed_high_q75": float(observed.quantile(0.75)),
    }
    origin = scored["source"]["last_observed_price_lkr_kg"].to_numpy(dtype=float)
    frozen_ratio = float(scored["source"].iloc[0][PROGRESS])
    rows = [
        {
            "state": "frozen_saved_vector",
            "observed_progress_ratio_used": frozen_ratio,
            "target_exceeded": int(float(scored["source"].iloc[0][EXCEEDED])),
            "lower_price": round(scored["lower"], 2),
            "predicted_price": round(scored["median"], 2),
            "upper_price": round(scored["upper"], 2),
            "role": "unchanged saved vector",
        }
    ]
    for label, ratio in levels.items():
        design = scored["design"].copy()
        design[PROGRESS] = float(ratio)
        design[EXCEEDED] = float(exceeded_flag(ratio))
        prices = predict_prices(design, origin, reproduced["models"])
        lower, median, upper = interval_from_prices(
            float(prices["q05"][0]),
            float(prices["q50"][0]),
            float(prices["q95"][0]),
            scored["qhat"],
        )
        rows.append(
            {
                "state": label,
                "observed_progress_ratio_used": float(ratio),
                "target_exceeded": exceeded_flag(ratio),
                "lower_price": round(lower, 2),
                "predicted_price": round(median, 2),
                "upper_price": round(upper, 2),
                "role": "sensitivity only, not a held-out accuracy result",
            }
        )
    return pd.DataFrame(rows)


def twelve_forecasts(bundle: dict) -> tuple[pd.DataFrame, list[dict]]:
    """One horizon-1 test row for each crop and score market. Medians must match the file."""
    pool = _lightgbm_c(bundle["predictions"], 1, "test")
    pool = pool[pool["crop_key"].isin(CROPS) & pool["market_key"].isin(SCORE_MARKETS)]
    if pool.empty:
        raise RuntimeError("No horizon-1 Experiment C test rows were found for the 12 score pairs.")
    chosen = pool.sort_values(["crop_key", "market_key", "issue"]).groupby(["crop_key", "market_key"], sort=False).tail(1)
    expected = {(crop, market) for crop in CROPS for market in SCORE_MARKETS}
    found = set(zip(chosen["crop_key"], chosen["market_key"], strict=False))
    missing = sorted(expected - found)
    if missing or len(chosen) != 12:
        raise RuntimeError(f"Expected 12 crop-market forecasts, found {len(chosen)}. Missing: {missing}.")
    models = load_bundles(OUT / "models", "C", 1)
    qhat = float(bundle["qhats"]["1"])
    forecasts = []
    checks = []
    for _, row in chosen.sort_values(["crop_key", "market_key"]).iterrows():
        scored = _score_row(row, bundle["eligible"], models, bundle["spec"], qhat)
        if scored["absolute_difference"] > REPRODUCTION_TOLERANCE:
            raise RuntimeError(
                f"Saved C horizon-1 median differs for {row['crop_key']}/{row['market_key']} "
                f"by {scored['absolute_difference']}."
            )
        source = scored["source"].iloc[0]
        checks.append(
            {
                "crop": str(row["crop_key"]),
                "market": str(row["market_key"]),
                "absolute_difference": scored["absolute_difference"],
            }
        )
        actual = float(row["y_true"])
        forecasts.append(
            {
                "crop": str(row["crop_key"]),
                "market": str(row["market_key"]),
                "forecast_issue_week": str(row["issue"]),
                "forecast_week": str(row["target"]),
                "horizon_weeks": 1,
                "cultivation_report_date": _date_or_none(source["cultivation_report_date"]),
                "cultivation_progress": float(source[PROGRESS]),
                "target_exceeded": int(float(source[EXCEEDED])),
                "last_observed_price": round(float(source["last_observed_price_lkr_kg"]), 2),
                "lower_price": round(scored["lower"], 2),
                "predicted_price": round(scored["median"], 2),
                "upper_price": round(scored["upper"], 2),
                "actual_price_historical_demo": round(actual, 2),
                "inside_interval_historical_demo": bool(scored["lower"] <= actual <= scored["upper"]),
                "model": "lightgbm",
                "feature_set": "C",
                "model_version": "experiment_c",
                "reproduction_absolute_difference": scored["absolute_difference"],
            }
        )
    return pd.DataFrame(forecasts), checks


def member3_packet(forecasts: list[dict], metrics: pd.DataFrame, status: dict) -> dict:
    light = metrics[metrics["model"].astype(str).str.lower().eq("lightgbm")]
    measured = {
        str(int(row.horizon)): None if pd.isna(row.picp_cqr) else float(row.picp_cqr)
        for row in light.itertuples(index=False)
    }
    below = [horizon for horizon, picp in measured.items() if picp is not None and picp < 90]
    downstream = []
    for item in forecasts:
        downstream.append(
            {
                "crop": item["crop"],
                "market": item["market"],
                "forecast_issue_week": item["forecast_issue_week"],
                "forecast_week": item["forecast_week"],
                "horizon_weeks": item["horizon_weeks"],
                "predicted_price": item["predicted_price"],
                "lower_price": item["lower_price"],
                "upper_price": item["upper_price"],
                "coverage_level_aim": 0.90,
                "cultivation_progress": item["cultivation_progress"],
                "cultivation_report_date": item["cultivation_report_date"],
                "cultivation_progress_spatial_scope": "national",
                "model": "lightgbm",
                "model_version": "experiment_c",
                "feature_set": "C",
            }
        )
    note = (
        "0.90 is the intended interval level. "
        "measured_picp_pct_by_horizon is this repository's matched-row Experiment C run."
    )
    if below:
        note += " Measured calibrated PICP is below 90% at horizon " + ", ".join(below) + "."
    return {
        "project": "J26-DS-322",
        "from_member": 2,
        "to_member": 3,
        "engine": "Hybrid Temporal Forecasting Engine",
        "purpose": "Forward price forecasts and the measured interval from this Experiment C run.",
        "model_version": "experiment_c",
        "feature_set": "C",
        "cultivation_progress_spatial_scope": "national",
        "coverage_level_aim": 0.90,
        "coverage_level_note": note,
        "measured_picp_pct_by_horizon": measured,
        "experiment_c_status": status.get("status"),
        "forecasts": downstream,
        "source": "predictions_c.csv and metrics_c.csv from this run. Notebook printed PICP values were not copied.",
    }


def required_paths() -> list[str]:
    names = [
        "status.json",
        "manifest.json",
        "cqr.json",
        "metrics_c.csv",
        "metrics_b_star.csv",
        "b_star_vs_c.csv",
        "predictions_c.csv",
        "predictions_b_star.csv",
        "c_eligible_table.parquet",
        "dynamic_reforecast.csv",
        "dynamic_reforecast_note.json",
        "member3_packet.json",
        "lightgbm_h4_by_crop.csv",
        "lightgbm_h4_by_market.csv",
        "audit/doa_cultivation_progress_audited.csv",
        "protocol_difference.json",
        "inference/saved_c_example.json",
        "inference/c_observed_ratio_sensitivity.csv",
        "inference/sensitivity_note.txt",
        "inference/forecasts_12_c.json",
        "inference/forecasts_12_c.csv",
        "inference/forecasts_12_c_reproduction_check.csv",
        "inference/member3_forecast_c.json",
        "evaluation/common_key_b_star_vs_c.csv",
        "evaluation/quantile_crossing_c.csv",
        "evaluation/conditional_picp_c.csv",
        "evaluation/residual_summary_c_h4.csv",
        "evaluation/shock_slice_c_h4.csv",
        "evaluation/robustness_c_h4_by_crop.csv",
        "evaluation/robustness_c_h4_by_market.csv",
        "evaluation/robustness_c_h4_by_season.csv",
        "evaluation/bstar_vs_c_block_bootstrap.csv",
        "evaluation/final_probabilistic_evaluation_c.csv",
        "evaluation/quantile_crossing_diagnostic_c.csv",
        "evaluation/conditional_interval_evaluation_c.csv",
        "live_vectors.json",
    ]
    for horizon in (1, 2, 4, 8, 12):
        for label in ("q05", "q50", "q95"):
            names.append(f"models/lightgbm_C_h{horizon}_{label}.joblib")
    return names


def benchmark_paths() -> list[str]:
    return [
        "common_key_benchmark.csv",
        "seasonal_naive_benchmark.csv",
        "mase_scale_52w_by_series.csv",
        "model_vs_naive_block_bootstrap.csv",
        "common_key_counts.csv",
        "best_model_by_horizon.csv",
        "benchmark_note.txt",
        "inference/saved_b_example.json",
        "studio_json/meta.json",
        "studio_json/final_evaluation_package.json",
    ]


def manifest_check(extra_absent: list[str] | None = None) -> dict:
    missing = [name for name in required_paths() if not (OUT / name).is_file()]
    benchmark_root = OUTPUTS / "evaluation"
    missing.extend(f"evaluation/{name}" for name in benchmark_paths() if not (benchmark_root / name).is_file())
    return {
        "missing_required": missing,
        "required_count": len(required_paths()) + len(benchmark_paths()),
        "notebooks_imported": False,
        "fit_called": False,
        "still_only_in_notebook": extra_absent
        or [
            "Colab Drive mount and the notebook cell printouts.",
        ],
    }


def recover_b_issue_fields(vectors: list[dict], table: pd.DataFrame, predictions: pd.DataFrame) -> list[dict]:
    """Add issue week and horizon from the saved vector's origin price and target week.

    A vector is accepted only when one training row has the same crop, market,
    target week, and last observed price. The saved prediction on that row supplies
    y_true. Nothing is filled in when that match is missing.
    """
    work = table.copy()
    work["crop_key"] = work["crop_id"].astype(str).str.strip().str.lower()
    work["market_key"] = work["market_id"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    work["target_week"] = pd.to_datetime(work["target_week"], format="mixed").dt.normalize()
    work["forecast_issue_week"] = pd.to_datetime(work["forecast_issue_week"], format="mixed").dt.normalize()
    saved = predictions.copy()
    saved["crop_key"] = saved["crop"].astype(str).str.strip().str.lower()
    saved["market_key"] = saved["market"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    saved["target_week"] = pd.to_datetime(saved["target_week"], format="mixed").dt.normalize()
    saved["forecast_issue_week"] = pd.to_datetime(saved["forecast_issue_week"], format="mixed").dt.normalize()
    saved["model"] = saved["model"].astype(str).str.lower()
    enriched = []
    for item in vectors:
        crop = str(item["crop"]).strip().lower()
        market = str(item["market"]).strip().lower().replace(" ", "_")
        target = pd.Timestamp(item.get("target_week") or item["forecast_week"]).normalize()
        origin = float(item["origin_price"])
        candidates = work[
            work["crop_key"].eq(crop) & work["market_key"].eq(market) & work["target_week"].eq(target)
        ].copy()
        candidates["origin_gap"] = (candidates["last_observed_price_lkr_kg"] - origin).abs()
        exact = candidates[candidates["origin_gap"].le(1e-4)]
        if len(exact) != 1:
            raise RuntimeError(
                f"Saved B vector {crop}/{market} target {target.date()} matched {len(exact)} origin prices. "
                "The issue week was not guessed."
            )
        chosen = exact.iloc[0]
        horizon = int(chosen["forecast_horizon_weeks"])
        issue = pd.Timestamp(chosen["forecast_issue_week"])
        match = saved[
            saved["model"].eq("lightgbm")
            & saved["feature_set"].astype(str).eq("B")
            & saved["crop_key"].eq(crop)
            & saved["market_key"].eq(market)
            & saved["horizon"].astype(int).eq(horizon)
            & saved["forecast_issue_week"].eq(issue)
            & saved["target_week"].eq(target)
        ]
        if len(match) != 1:
            raise RuntimeError(f"Saved B prediction lookup failed for {crop}/{market} h{horizon}: {len(match)} rows.")
        updated = dict(item)
        updated["forecast_issue_week"] = issue.strftime("%Y-%m-%d")
        updated["target_week"] = target.strftime("%Y-%m-%d")
        updated["horizon"] = horizon
        updated["y_true"] = float(match["y_true"].iloc[0])
        updated["saved_median"] = float(match["y_pred"].iloc[0])
        updated["issue_week_source"] = "matched crop, market, target week, and last observed price to the training table"
        enriched.append(updated)
    return enriched


def reproduce_saved_b(vectors: list[dict]) -> dict:
    """Call the frozen B model for one recovered vector. There is no fit."""
    if not vectors:
        raise RuntimeError("No Experiment B live vectors are available.")
    chosen = vectors[0]
    horizon = int(chosen["horizon"])
    bundles = load_bundles(OUTPUTS / "experiment_b" / "models", "B", horizon)
    columns = list(bundles["q50"]["columns"])
    missing = [name for name in columns if name not in chosen["features"]]
    if missing:
        raise RuntimeError("Saved B vector is missing model columns: " + ", ".join(missing))
    design = pd.DataFrame([{name: chosen["features"][name] for name in columns}])
    origin = np.array([float(chosen["origin_price"])], dtype=float)
    prices = predict_prices(design, origin, bundles)
    qhat = float(load_json(OUTPUTS / "experiment_b" / "cqr.json")["qhat_by_model"]["lightgbm"][str(horizon)])
    lower, median, upper = interval_from_prices(float(prices["q05"][0]), float(prices["q50"][0]), float(prices["q95"][0]), qhat)
    difference = abs(float(chosen["saved_median"]) - median)
    if difference > REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"Saved B median differs by {difference} at horizon {horizon}. No prediction was invented."
        )
    return {
        "check": "reproduce_one_saved_b_forecast",
        "fit_called": False,
        "crop": chosen["crop"],
        "market": chosen["market"],
        "forecast_issue_week": chosen["forecast_issue_week"],
        "target_week": chosen["target_week"],
        "horizon_weeks": horizon,
        "last_observed_price": round(float(chosen["origin_price"]), 2),
        "lower_price": round(lower, 2),
        "predicted_price": round(median, 2),
        "upper_price": round(upper, 2),
        "actual_price_historical_demo": round(float(chosen["y_true"]), 2),
        "saved_median": round(float(chosen["saved_median"]), 6),
        "reproduction_absolute_difference": difference,
        "feature_set": "B",
        "model": "lightgbm",
        "note": (
            "The stored live file had forecast_week and origin_price, and no issue week. "
            "The issue week is the unique training-table row with that target week and origin price. "
            f"That row is horizon {horizon}. The file does not contain a separate horizon-4 vector."
        ),
    }


def write_c_live_vectors(bundle: dict) -> int:
    """Latest real-progress test vector per crop, market, and horizon. No fit and no new prediction."""
    rows = []
    eligible = bundle["eligible"]
    for horizon in (1, 2, 4, 8, 12):
        bundles = load_bundles(OUT / "models", "C", horizon)
        part = eligible[
            eligible["horizon"].eq(horizon)
            & eligible["split"].astype(str).str.lower().eq("test")
            & eligible["market_key"].isin(SCORE_MARKETS)
        ].copy()
        if part.empty:
            continue
        latest = part.sort_values("issue").groupby(["crop_key", "market_key"], sort=False).tail(1).reset_index(drop=True)
        design = design_matrix(latest, bundles, bundle["spec"])
        for position, row in latest.iterrows():
            features = {}
            for name in design.columns:
                value = design.iloc[int(position)][name]
                features[str(name)] = None if pd.isna(value) else float(value)
            rows.append(
                {
                    "crop": str(row["crop_key"]),
                    "market": str(row["market_key"]),
                    "horizon": int(horizon),
                    "origin_price": float(row["last_observed_price_lkr_kg"]),
                    "forecast_issue_week": str(row["issue"]),
                    "target_week": str(row["target"]),
                    "y_true": float(row["target_price_lkr_kg"]),
                    "real_progress_ratio": float(row[PROGRESS]),
                    "real_target_exceeded": int(float(row[EXCEEDED])),
                    "cultivation_report_date": _date_or_none(row["cultivation_report_date"]),
                    "features": features,
                }
            )
    write_json(OUT / "live_vectors.json", {"fit_called": False, "vectors": rows})
    return len(rows)


def build_b_live_vectors(model_dir: Path | None = None) -> list[dict]:
    """Latest horizon-1 test row per scored market, using the saved B columns. No fit."""
    import joblib

    from htfe.config import built_file
    from htfe.training.run_experiment_a import _eligible

    folder = model_dir or (OUTPUTS / "experiment_b" / "models")
    bundle = joblib.load(folder / "lightgbm_B_h1_q50.joblib")
    table = pd.read_parquet(built_file("training_table.parquet"))
    subset = _eligible(table, 1)
    subset = subset[subset["split"].astype(str).str.lower().eq("test")].copy()
    subset = subset[subset["market_id"].astype(str).str.lower().str.replace(" ", "_", regex=False).isin(EVAL_IDS)]
    subset["issue"] = pd.to_datetime(subset["forecast_issue_week"])
    latest = (
        subset.sort_values("issue")
        .groupby(["crop_id", "market_id"], sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    spec = bundle["categories"]
    work = apply_categories(latest, spec)
    design, names = _design(work, list(bundle["feature_columns"]))
    saved = list(bundle["columns"])
    if list(names) != saved:
        raise RuntimeError("Rebuilt Experiment B columns do not match the saved horizon-1 model.")
    rows = []
    for position, row in latest.iterrows():
        features = {}
        for name in saved:
            value = design.iloc[int(position)][name]
            features[str(name)] = None if pd.isna(value) else float(value)
        rows.append(
            {
                "crop": str(row["crop_id"]),
                "market": str(row["market_id"]).lower().replace(" ", "_"),
                "origin_price": float(row["last_observed_price_lkr_kg"]),
                "forecast_week": pd.Timestamp(row["target_week"]).strftime("%Y-%m-%d"),
                "target_week": pd.Timestamp(row["target_week"]).strftime("%Y-%m-%d"),
                "forecast_issue_week": pd.Timestamp(row["forecast_issue_week"]).strftime("%Y-%m-%d"),
                "horizon": 1,
                "features": features,
            }
        )
    return rows


def write_b_example() -> dict:
    from htfe.config import built_file

    path = OUTPUTS / "experiment_b" / "live_vectors.json"
    if not path.is_file():
        built = build_b_live_vectors()
        write_json(path, {"fit_called": False, "vectors": built, "source": "latest horizon-1 test rows from the saved Experiment B columns"})
    stored = load_json(path)
    if isinstance(stored, dict) and "vectors" in stored:
        stored = stored["vectors"]
    table = pd.read_parquet(built_file("training_table.parquet"))
    predictions = pd.read_csv(OUTPUTS / "experiment_b" / "predictions.csv")
    enriched = recover_b_issue_fields(stored, table, predictions)
    payload = reproduce_saved_b(enriched)
    path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    destination = OUTPUTS / "evaluation" / "inference"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "saved_b_example.json", payload)
    print(
        f"reproduced B h{payload['horizon_weeks']} median difference {payload['reproduction_absolute_difference']}",
        flush=True,
    )
    return payload


def write_inference() -> dict:
    b_example = write_b_example()
    bundle = _prepare()
    reproduced = reproduce_one(bundle)
    ratios = sensitivity(reproduced)
    forecast_frame, checks = twelve_forecasts(bundle)
    metrics = pd.read_csv(OUT / "metrics_c.csv")
    status = load_json(OUT / "status.json")
    packet = member3_packet(forecast_frame.to_dict(orient="records"), metrics, status)
    INFERENCE.mkdir(parents=True, exist_ok=True)
    write_json(INFERENCE / "saved_c_example.json", reproduced["payload"])
    ratios.to_csv(INFERENCE / "c_observed_ratio_sensitivity.csv", index=False)
    note = (
        "The frozen_saved_vector row is the unchanged horizon-4 test vector. "
        "The q25, median, and q75 rows replace only cultivation progress on that same vector "
        "with observed national ratios for this crop. Ratios above 1 are kept. "
        "This is not another held-out accuracy result."
    )
    (INFERENCE / "sensitivity_note.txt").write_text(note, encoding="utf-8")
    write_json(
        INFERENCE / "forecasts_12_c.json",
        {
            "horizon_weeks": 1,
            "model_version": "experiment_c",
            "feature_set": "C",
            "model": "lightgbm",
            "coverage_level_aim": 0.90,
            "measured_picp_horizon_1": packet["measured_picp_pct_by_horizon"].get("1"),
            "spatial_scope_of_cultivation_progress": "national",
            "fit_called": False,
            "forecasts": forecast_frame.to_dict(orient="records"),
        },
    )
    forecast_frame.to_csv(INFERENCE / "forecasts_12_c.csv", index=False)
    pd.DataFrame(checks).to_csv(INFERENCE / "forecasts_12_c_reproduction_check.csv", index=False)
    write_json(INFERENCE / "member3_forecast_c.json", packet)
    n_vectors = write_c_live_vectors(bundle)
    worst = max(item["absolute_difference"] for item in checks)
    print(
        f"reproduced h4 median difference {reproduced['payload']['reproduction_absolute_difference']}",
        flush=True,
    )
    print(f"12 horizon-1 forecasts, max median difference {worst}", flush=True)
    print(f"C live vectors with issue week: {n_vectors}", flush=True)
    return {
        "reproduce_difference": reproduced["payload"]["reproduction_absolute_difference"],
        "n_forecasts": len(forecast_frame),
        "max_forecast_difference": worst,
        "measured_picp": packet["measured_picp_pct_by_horizon"],
        "b_reproduction_difference": b_example["reproduction_absolute_difference"],
        "b_horizon": b_example["horizon_weeks"],
        "c_live_vectors": n_vectors,
    }
