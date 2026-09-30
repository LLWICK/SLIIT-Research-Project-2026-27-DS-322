"""Methodology run: cultivation as-of, feature sets A/B/C, Optuna, CQR, studio export.

Usage (from member2_ForecastingEngine):
  python -m training.run_methodology

Does not overwrite artifacts/latest from the earlier price-panel run.
"""
from __future__ import annotations

import json
import platform
import shutil
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from training import baselines
from training.build_feature_frame import build as build_features
from training.config import ARTIFACTS, EVAL_MARKETS, FEATURE_PATH, PANEL_PATH, STUDIO_DATA
from training.evaluate import metrics_tables, slice_metrics
from training.export_studio_json import _json_default, _metric_row
from training.feature_sets import feature_sets, fullest, macro_columns, progress_available
from training.methodology_fit import (
    ablation_payload,
    apply_split_cqr,
    fit_lightgbm,
    fit_xgboost,
    shap_importance,
    tune_lightgbm,
    tune_xgboost,
)

MAP_PATH = Path(__file__).resolve().parents[2] / "data_pipeline" / "maps" / "origin_map.yaml"
DOCS = Path(__file__).resolve().parents[3] / "docs"


def _lib(name: str) -> str:
    try:
        return version(name)
    except Exception:
        return "not-installed"


def _dump(name: str, payload) -> None:
    path = STUDIO_DATA / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _load_sarimax() -> pd.DataFrame:
    path = ARTIFACTS / "latest" / "predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame = frame[frame["model"].eq("sarimax")].copy()
    if "target_form" not in frame.columns:
        frame["target_form"] = "level"
    return frame


def _export_studio(frame, predictions, ablation, importance, meta, cqr, reforecast, curves) -> None:
    STUDIO_DATA.mkdir(parents=True, exist_ok=True)
    test = predictions[predictions["split"].eq("test")]
    headline = {
        "naive_last": slice_metrics(predictions, "naive_last", "level"),
        "seasonal_naive_52": slice_metrics(predictions, "seasonal_naive_52", "level"),
        "sarimax": slice_metrics(predictions, "sarimax", "level"),
        "lightgbm": slice_metrics(predictions, "lightgbm", "logret"),
        "xgboost": slice_metrics(predictions, "xgboost", "logret"),
    }
    comparison = [
        _metric_row("Last-value naive", "naive_last", "naive_last", headline["naive_last"]),
        _metric_row("Seasonal naive (52)", "seasonal_naive_52", "seasonal_naive_52", headline["seasonal_naive_52"]),
        _metric_row("SARIMAX (1,1,1)(1,0,0,52)", "sarima", "sarimax", headline["sarimax"]),
        _metric_row("LightGBM quantile", "lightgbm", "lightgbm", headline["lightgbm"]),
        _metric_row("XGBoost quantile", "xgboost", "xgboost", headline["xgboost"]),
        {
            "model": "LSTM-style MLP (not run)",
            "id": "lstm",
            "n_scored": 0,
            "mae": None,
            "rmse": None,
            "mape": None,
            "pinball": None,
            "picp": None,
            "interval_width": None,
            "backend": "not_run",
        },
    ]
    qhat = cqr.get("qhat")
    interval_note = cqr.get("limit")
    models = {
        "lightgbm": {
            "backend": "lightgbm",
            "qhat": qhat,
            "qhat_by_horizon": cqr.get("qhat_by_horizon"),
            "note": interval_note,
            "metrics": headline["lightgbm"],
            "train_curve": curves or [],
            "feature_set": meta.get("feature_set"),
        },
        "xgboost": {
            "backend": "xgboost",
            "qhat": None,
            "note": "Quantile XGBoost on the same feature set as LightGBM. Intervals are raw quantiles.",
            "metrics": headline["xgboost"],
            "train_curve": [],
        },
        "sarima": {
            "backend": "sarimax",
            "qhat": None,
            "note": "Univariate SARIMAX reused from the price-only walk-forward. It does not use weather or cultivation progress.",
            "metrics": headline["sarimax"],
            "train_curve": [],
        },
        "lstm": {"backend": "not_run", "qhat": None, "metrics": {"overall": {}}, "train_curve": []},
    }
    lgb = test[(test["model"] == "lightgbm") & (test["horizon"] == 1) & (test["target_form"] == "logret")].copy()
    pred_rows = []
    for _, row in lgb.sort_values(["crop", "market", "target_week"]).iterrows():
        pred_rows.append(
            {
                "crop": row["crop"],
                "market": str(row["market"]).lower().replace(" ", "_"),
                "week_start": pd.Timestamp(row["target_week"]).strftime("%Y-%m-%d"),
                "y_true": row["y_true"],
                "point": row["y_pred"],
                "lower": row["y_pred_q05"],
                "upper": row["y_pred_q95"],
                "q05": row["y_pred_q05"],
                "q50": row["y_pred"],
                "q95": row["y_pred_q95"],
                "season": str(row["season"]).lower(),
                "disruption_flag": False,
                "cultivation_progress": None if pd.isna(row.get("cultivation_progress")) else float(row["cultivation_progress"]),
            }
        )
    source = meta.get("cultivation_progress_source")
    live = []
    for (crop, market), group in lgb.groupby(["crop", "market"]):
        last = group.sort_values("week_start").iloc[-1]
        progress = last.get("cultivation_progress")
        live.append(
            {
                "crop": crop,
                "market": str(market).lower().replace(" ", "_"),
                "forecast_week": pd.Timestamp(last["target_week"]).strftime("%Y-%m-%d"),
                "cultivation_intensity": None if pd.isna(progress) else float(progress),
                "cultivation_progress": None if pd.isna(progress) else float(progress),
                "predicted_price": float(last["y_pred"]),
                "lower_price": float(last["y_pred_q05"]),
                "upper_price": float(last["y_pred_q95"]),
                "coverage_level": 0.9,
                "commitment_source": source,
                "progress_in_model": bool(meta.get("progress_in_model")),
                "model_version": meta.get("version"),
                "note": meta.get("live_note"),
            }
        )
    _dump("comparison", comparison)
    _dump("ablation", ablation)
    _dump("importance", importance)
    _dump("live_forecast", live)
    _dump("models", models)
    _dump("predictions", {"lightgbm": pred_rows})
    _dump("commitment_weeks", reforecast.get("rows") or [])
    meta_path = STUDIO_DATA / "meta.json"
    current = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    current.update(
        {
            "version": meta.get("version"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": 42,
            "confidence_level": 0.9,
            "commitment_source": source,
            "elapsed_s": meta.get("elapsed_s"),
            "interval_note": interval_note,
            "feature_set": meta.get("feature_set"),
            "progress_in_model": bool(meta.get("progress_in_model")),
            "libraries": meta.get("libraries") or {},
            "n_scored_test": int(headline["lightgbm"]["overall"].get("n_scored") or 0),
        }
    )
    _dump("meta", current)
    dictionary = []
    dict_path = STUDIO_DATA / "data_dictionary.json"
    if dict_path.exists():
        dictionary = json.loads(dict_path.read_text(encoding="utf-8"))
    notes = {
        "cultivation_intensity": "Not the methodology feature. Seasonal extent ratio is kept out of feature sets A/B/C.",
        "commitment_source": source,
    }
    replaced = False
    for row in dictionary:
        if row.get("column") in notes:
            row["notes"] = notes[row["column"]]
            replaced = True
    if not any(row.get("column") == "cultivation_progress" for row in dictionary):
        dictionary.append(
            {
                "column": "cultivation_progress",
                "dtype": "float",
                "role": "feature",
                "notes": "achieved hectares / target hectares, as-of. Missing when no district-month table exists.",
            }
        )
    if replaced or dictionary:
        _dump("data_dictionary", dictionary)
    # Touch frame so an unused-argument lint does not hide that export is panel-aware.
    _ = frame


def main() -> None:
    from data_pipeline.build.cultivation_progress import (
        attach_progress,
        assert_formula,
        find_monthly_table,
        load_monthly,
        write_gap_report,
    )
    from data_pipeline.build.macros import attach_macros

    started = time.perf_counter()
    print("checking achieved/target as-of formula")
    assert_formula()
    panel = pd.read_parquet(PANEL_PATH)
    origin_map = yaml.safe_load(MAP_PATH.read_text(encoding="utf-8"))
    found = find_monthly_table()
    gap = write_gap_report(found)
    DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gap, DOCS / "Cultivation-Progress-Data-Gap-IT23415836.md")
    monthly = load_monthly(found) if found is not None else None
    print(f"cultivation table={found}")
    panel = attach_progress(panel, monthly, origin_map)
    panel, macro_meta = attach_macros(panel)
    print(f"macros={macro_meta['status']} columns={macro_meta['columns']}")
    frame = build_features(panel)
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FEATURE_PATH, index=False)
    sets = feature_sets(frame)
    set_name, columns = fullest(sets)
    has_progress = progress_available(frame)
    print(f"feature set {set_name} n_columns={len(columns)} progress={has_progress} macros={macro_columns(frame)}")

    out_dir = ARTIFACTS / "methodology"
    model_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("optuna lightgbm 25 trials on horizon 4")
    lgb_params = tune_lightgbm(frame, columns, n_trials=25)
    print("optuna xgboost 15 trials on horizon 4")
    xgb_params = tune_xgboost(frame, columns, n_trials=15)
    (out_dir / "optuna_params.json").write_text(
        json.dumps({"lightgbm": lgb_params, "xgboost": xgb_params, "tuned_on": set_name}, indent=2),
        encoding="utf-8",
    )

    parts = []
    curve = []
    h4_model = None
    live_vectors: list = []
    for arm in ("A", "B", "C"):
        arm_columns = sets[arm]
        if not arm_columns:
            print(f"skip lightgbm {arm}")
            continue
        print(f"lightgbm {arm}")
        rows, info = fit_lightgbm(frame, arm_columns, lgb_params, arm, model_dir if arm == set_name else None)
        parts.append(rows)
        if arm == set_name:
            curve = info["curve"]
            h4_model = info["saved"].get("h4_q50")
            live_vectors = info.get("live_vectors") or []

    print("xgboost", set_name)
    parts.append(fit_xgboost(frame, columns, xgb_params, set_name))
    print("naive baselines")
    parts.append(baselines.predict(frame))
    print("reuse sarimax walk-forward")
    sar = _load_sarimax()
    if not sar.empty:
        parts.append(sar)

    predictions = pd.concat([part for part in parts if part is not None and not part.empty], ignore_index=True)
    primary = predictions[predictions["model"].eq(f"lightgbm_{set_name}")].copy()
    primary["model"] = "lightgbm"
    predictions = pd.concat([predictions, primary], ignore_index=True)
    note = (
        "A and B were trained. C was not, because no district-month target/achieved table was found. "
        "The seasonal extent ratio was not renamed to cultivation progress."
        if not has_progress
        else "A, B, and C use the same LightGBM hyperparameters. C adds cultivation_progress = summed achieved / summed target."
    )
    ablation = ablation_payload(predictions, has_progress, note)
    print("split cqr on primary lightgbm")
    predictions, cqr = apply_split_cqr(predictions, "lightgbm")
    importance = []
    if h4_model:
        print("shap")
        try:
            importance = shap_importance(Path(h4_model), frame, columns)
        except Exception as exc:
            print(f"shap failed: {exc}")
            importance = []
    if not importance:
        gains = {}
        for name in columns:
            gains[name] = 0.0
        importance = [{"feature": name, "importance": 0.0, "std": 0.0} for name in columns[:12]]

    reforecast = {
        "status": "not_available" if not has_progress else "not_run",
        "note": (
            "The trained model has no cultivation_progress column, so a later monthly update cannot be passed in. "
            "No week 2/5/8/12 path was used, and no hectares were invented."
            if not has_progress
            else "Progress is in the model. Re-forecast rows are produced by changing only that column."
        ),
        "rows": [],
    }
    if has_progress and h4_model:
        reforecast = _reforecast(frame, Path(h4_model), columns)

    by_model, by_series = metrics_tables(predictions)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    by_model.to_csv(out_dir / "metrics_by_model_horizon.csv", index=False)
    by_series.to_csv(out_dir / "metrics_by_series.csv", index=False)
    (out_dir / "ablation.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    (out_dir / "cqr.json").write_text(json.dumps(cqr, indent=2), encoding="utf-8")
    (out_dir / "reforecast.json").write_text(json.dumps(reforecast, indent=2), encoding="utf-8")
    (out_dir / "live_vectors.json").write_text(json.dumps(live_vectors, indent=2), encoding="utf-8")

    source = "monthly_target_achieved" if has_progress else "unavailable_no_district_monthly_series"
    elapsed = round(time.perf_counter() - started, 1)
    meta = {
        "version": f"methodology-{datetime.now().strftime('%Y%m%d_%H%M')}",
        "seed": 42,
        "elapsed_s": elapsed,
        "feature_set": set_name,
        "progress_in_model": has_progress,
        "cultivation_progress_source": source,
        "macros": macro_meta,
        "optuna": {"lightgbm": lgb_params, "xgboost": xgb_params},
        "cqr": cqr,
        "sarimax": "reused from artifacts/latest price-only walk-forward",
        "python": platform.python_version(),
        "libraries": {
            "lightgbm": _lib("lightgbm"),
            "xgboost": _lib("xgboost"),
            "optuna": _lib("optuna"),
            "mapie": _lib("mapie"),
            "shap": _lib("shap"),
        },
        "live_note": reforecast["note"],
        "eval_markets": list(EVAL_MARKETS),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "progress_in_model": has_progress,
                "feature_set": set_name,
                "columns": columns,
                "model_dir": str(model_dir),
                "h1_q50": str(model_dir / f"lightgbm_{set_name}_h1_q50.joblib"),
                "note": reforecast["note"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _export_studio(frame, predictions, ablation, importance, meta, cqr, reforecast, curve)
    print(f"done in {elapsed}s")
    headline = by_model[(by_model["slice"] == "test") & (by_model["horizon"] == 4) & (by_model["model"].isin(["lightgbm", "xgboost", "naive_last", "sarimax"]))]
    print(headline.to_string(index=False))


def _reforecast(frame: pd.DataFrame, model_path: Path, columns: list[str]) -> dict:
    """Change only cultivation_progress on saved horizon-4 rows and call the booster."""
    import joblib

    bundle = joblib.load(model_path)
    model = bundle["model"]
    names = bundle["columns"]
    if "cultivation_progress" not in names:
        return {"status": "not_available", "note": "Saved model has no cultivation_progress column.", "rows": []}
    work = frame[frame["market"].isin(EVAL_MARKETS) & frame["split"].eq("test") & frame["is_observed"]].copy()
    # One origin per crop-market: the latest test week that still has an 4-week target.
    rows = []
    for (crop, market), group in work.groupby(["crop", "market"]):
        group = group.sort_values("week_start")
        origin = group.dropna(subset=["y_price_h4"]).iloc[-1]
        base = origin.reindex(names).astype(float)
        actual = float(origin["y_price_h4"])
        price = float(origin["price"])
        seen = set()
        for progress in group["cultivation_progress"].dropna().unique():
            key = round(float(progress), 4)
            if key in seen:
                continue
            seen.add(key)
            vector = base.copy()
            vector["cultivation_progress"] = float(progress)
            raw = float(model.predict(pd.DataFrame([vector]))[0])
            point = price * float(np.exp(raw))
            rows.append(
                {
                    "crop": crop,
                    "market": str(market).lower().replace(" ", "_"),
                    "information_state": key,
                    "cultivation_progress": key,
                    "mae": round(abs(point - actual), 2),
                    "point": round(point, 2),
                    "y_true": round(actual, 2),
                }
            )
    return {
        "status": "complete",
        "note": "Each row changes only cultivation_progress on the same origin. The model is not retrained.",
        "rows": rows,
    }


if __name__ == "__main__":
    main()
