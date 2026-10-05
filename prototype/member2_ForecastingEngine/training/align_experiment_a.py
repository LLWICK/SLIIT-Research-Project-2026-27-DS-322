"""Add Experiment A comparability, SARIMAX gap rows, and split CQR.

Does not overwrite predictions.csv, artifacts/latest, or artifacts/methodology.

Usage from prototype/member2_ForecastingEngine:
  python -u -m training.align_experiment_a
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from training.config import ARTIFACTS, EVAL_MARKETS, FEATURE_PATH, HORIZONS, TRAIN_END
from training.evaluate import _block
from training.methodology_fit import _conformal_quantile
from training.sarima_auto import _fit

A_DIR = ARTIFACTS / "experiment_a"
PREDICTIONS = A_DIR / "predictions.csv"
ORDER_PATH = ARTIFACTS / "latest" / "models" / "sarima_order.json"
KEYS = ["crop", "market", "forecast_issue_week", "horizon"]


def _stamp(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["forecast_issue_week"] = pd.to_datetime(out["forecast_issue_week"]).astype("datetime64[ns]")
    out["horizon"] = out["horizon"].astype(int)
    out["crop"] = out["crop"].astype(str)
    out["market"] = out["market"].astype(str)
    return out


def _key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = _stamp(frame)
    out["forecast_issue_week"] = out["forecast_issue_week"].dt.strftime("%Y-%m-%d")
    return out


def _overlap(predictions: pd.DataFrame) -> dict:
    test = predictions[predictions["split"].eq("test")]
    counts = {}
    keysets = {}
    for model in ("lightgbm", "xgboost", "sarimax"):
        part = _key_frame(test[test["model"].eq(model)])
        keysets[model] = set(map(tuple, part[KEYS].to_numpy()))
        counts[model] = len(keysets[model])
    by_horizon = []
    for horizon in HORIZONS:
        row = {"horizon": int(horizon)}
        sets = {}
        for model in ("lightgbm", "xgboost", "sarimax"):
            part = test[test["model"].eq(model) & test["horizon"].eq(horizon)]
            sets[model] = set(map(tuple, _key_frame(part)[["crop", "market", "forecast_issue_week"]].to_numpy()))
            row[f"{model}_rows"] = len(sets[model])
        row["lightgbm_equals_xgboost"] = sets["lightgbm"] == sets["xgboost"]
        row["sarimax_missing_from_trees"] = len(sets["lightgbm"] - sets["sarimax"])
        row["sarimax_extra"] = len(sets["sarimax"] - sets["lightgbm"])
        row["overlap"] = len(sets["lightgbm"] & sets["sarimax"])
        by_horizon.append(row)
    return {
        "test_period": "2024-2025",
        "key": KEYS,
        "observed_price_mask": "rows already scored in artifacts/experiment_a/predictions.csv",
        "counts": counts,
        "lightgbm_equals_xgboost": keysets["lightgbm"] == keysets["xgboost"],
        "exact_match_before_gap_fill": keysets["lightgbm"] == keysets["xgboost"] == keysets["sarimax"],
        "reused_test_rows": len(keysets["lightgbm"] & keysets["sarimax"]),
        "missing_test_rows": len(keysets["lightgbm"] - keysets["sarimax"]),
        "extra_sarimax_test_rows": len(keysets["sarimax"] - keysets["lightgbm"]),
        "by_horizon": by_horizon,
    }


def _cqr(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    try:
        import mapie

        mapie_version = getattr(mapie, "__version__", "installed")
    except Exception:
        mapie_version = None
    rows = []
    qhats: dict[str, dict[str, float]] = {}
    for model in ("lightgbm", "xgboost"):
        qhats[model] = {}
        block = predictions[predictions["model"].eq(model)]
        for horizon, group in block.groupby("horizon"):
            calibrate = group[group["split"].eq("calibrate")]
            test = group[group["split"].eq("test")]
            scores = np.maximum(
                calibrate["y_pred_q05"].to_numpy(dtype=float) - calibrate["y_true"].to_numpy(dtype=float),
                calibrate["y_true"].to_numpy(dtype=float) - calibrate["y_pred_q95"].to_numpy(dtype=float),
            )
            qhat = _conformal_quantile(scores, 0.1)
            qhats[model][str(int(horizon))] = round(qhat, 4)
            y_true = test["y_true"].to_numpy(dtype=float)
            y_pred = test["y_pred"].to_numpy(dtype=float)
            low = test["y_pred_q05"].to_numpy(dtype=float)
            high = test["y_pred_q95"].to_numpy(dtype=float)
            raw = _block(y_true, y_pred, low, high)
            calibrated = _block(y_true, y_pred, low - qhat, high + qhat)
            rows.append(
                {
                    "model": model,
                    "horizon": int(horizon),
                    "n_calibration": int(len(calibrate)),
                    "n_test": int(len(test)),
                    "qhat": round(qhat, 4),
                    "pinball": raw["pinball"],
                    "picp_raw": raw["picp"],
                    "interval_width_raw": raw["interval_width"],
                    "picp_cqr": calibrated["picp"],
                    "interval_width_cqr": calibrated["interval_width"],
                }
            )
    payload = {
        "method": "split_cqr_romano2019",
        "alpha": 0.1,
        "calibration": "2023 issue weeks only",
        "test": "2024-2025 issue weeks, intervals adjusted, point forecasts unchanged",
        "models": ["lightgbm", "xgboost"],
        "sarimax": "interval columns stay blank; CQR was not applied",
        "predictions_file_unchanged": True,
        "mapie_version": mapie_version,
        "mapie_note": (
            "MAPIE is installed, but ConformalizedQuantileRegressor refits an estimator and does not accept the three already-fit quantile boosters. "
            "The conformity score was applied directly to the saved 0.05 and 0.95 forecasts. qhat uses the 2023 calibration block only."
            if mapie_version
            else "MAPIE was not importable. Split CQR was applied directly."
        ),
        "qhat_by_model": qhats,
        "by_model_horizon": rows,
        "limit": "Time-ordered prices are not exchangeable, so this is an empirical calibration, not a universal 90% guarantee.",
    }
    return pd.DataFrame(rows), payload


def _write_metrics_with_cqr(cqr_rows: pd.DataFrame) -> None:
    metrics = pd.read_csv(A_DIR / "metrics_by_model_horizon.csv")
    extra = cqr_rows.rename(
        columns={
            "picp_raw": "picp_cqr_source_raw",
            "interval_width_raw": "width_cqr_source_raw",
        }
    )
    keep = extra[
        ["model", "horizon", "qhat", "picp_cqr", "interval_width_cqr", "n_calibration"]
    ]
    merged = metrics.merge(keep, on=["model", "horizon"], how="left")
    merged.loc[~merged["slice"].eq("test"), ["qhat", "picp_cqr", "interval_width_cqr", "n_calibration"]] = np.nan
    merged.to_csv(A_DIR / "metrics_with_cqr.csv", index=False)


def _replay(feature: pd.DataFrame, needed: pd.DataFrame, cached: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    order_payload = json.loads(ORDER_PATH.read_text(encoding="utf-8"))
    order = tuple(order_payload["order"])
    seasonal = tuple(order_payload["seasonal_order"])
    scoped = feature[feature["market"].isin(EVAL_MARKETS)].copy()
    scoped["week_start"] = pd.to_datetime(scoped["week_start"]).astype("datetime64[ns]")
    scoped["crop"] = scoped["crop"].astype(str)
    scoped["market"] = scoped["market"].astype(str)
    needed = _stamp(needed)
    cached = _stamp(cached)
    rows = []
    checks = []
    max_h = max(HORIZONS)
    for (crop, market), ask in needed.groupby(["crop", "market"], sort=False):
        group = scoped[(scoped["crop"].str.lower() == crop.lower()) & (scoped["market"] == market)].copy()
        group = group.sort_values("week_start").reset_index(drop=True)
        if group["week_start"].diff().dt.days.dropna().ne(7).any():
            raise AssertionError(f"{crop} {market} is not a regular Monday grid")
        y_state = group["price_feature"].astype(float).ffill().bfill().to_numpy()
        train_idx = np.where(group["week_start"] <= pd.Timestamp(TRAIN_END))[0]
        if len(train_idx) < 80:
            raise RuntimeError(f"{crop} {market} has too little train history for SARIMAX")
        origin0 = int(train_idx[-1])
        result = _fit(y_state[: origin0 + 1], order, seasonal)
        ask_weeks = set(pd.to_datetime(ask["forecast_issue_week"]))
        verify = cached[(cached["crop"].str.lower() == crop.lower()) & (cached["market"] == market) & cached["split"].eq("test")]
        verify = verify.sample(n=min(12, len(verify)), random_state=42) if len(verify) else verify
        verify_weeks = set(pd.to_datetime(verify["forecast_issue_week"])) if len(verify) else set()
        positions = {pd.Timestamp(stamp): index for index, stamp in enumerate(group["week_start"])}
        for origin in range(origin0, len(group) - 1):
            issue = pd.Timestamp(group.at[origin, "week_start"])
            observed = bool(group.at[origin, "is_observed"])
            if origin > origin0 and issue in ask_weeks and not observed:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    forecast = np.asarray(result.get_forecast(max_h + 1).predicted_mean, dtype=float)
                _emit(rows, ask, group, origin, issue, forecast, before_append=True)
            if origin > origin0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = result.append(y_state[origin : origin + 1], refit=False)
            if issue in verify_weeks and observed:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    forecast = np.asarray(result.get_forecast(max_h).predicted_mean, dtype=float)
                _check(checks, verify, crop, market, issue, forecast, positions, origin)
            if issue in ask_weeks and observed:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    forecast = np.asarray(result.get_forecast(max_h).predicted_mean, dtype=float)
                _emit(rows, ask, group, origin, issue, forecast, before_append=False)
        print(f"replayed {crop} {market}", flush=True)
    made = pd.DataFrame(rows)
    report = {
        "order": list(order),
        "seasonal_order": list(seasonal),
        "verification_rows": len(checks),
        "max_abs_error_vs_cached": None if not checks else round(float(np.max(np.abs(checks))), 6),
    }
    if checks and float(np.max(np.abs(checks))) > 0.05:
        raise AssertionError(f"replayed SARIMAX does not match the cache: max abs {report['max_abs_error_vs_cached']}")
    return made, report


def _emit(rows: list, ask: pd.DataFrame, group: pd.DataFrame, origin: int, issue: pd.Timestamp, forecast: np.ndarray, before_append: bool) -> None:
    part = ask[pd.to_datetime(ask["forecast_issue_week"]).eq(issue)]
    for _, item in part.iterrows():
        horizon = int(item["horizon"])
        target = pd.Timestamp(item["target_week"])
        hits = np.flatnonzero(group["week_start"].eq(target))
        if len(hits) != 1:
            raise AssertionError(f"target week {target.date()} is missing for {issue.date()}")
        steps = int(hits[0] - origin)
        index = steps if before_append else steps - 1
        if index < 0 or index >= len(forecast):
            raise AssertionError(f"forecast index {index} is outside the SARIMAX horizon")
        value = float(forecast[index])
        if not np.isfinite(value):
            raise AssertionError(f"SARIMAX forecast is not finite for {issue.date()} h{horizon}")
        rows.append(
            {
                "unique_id": f"{item['crop']}__{item['market']}",
                "crop": item["crop"],
                "market": item["market"],
                "forecast_issue_week": issue,
                "target_week": target,
                "horizon": horizon,
                "y_true": float(item["y_true"]),
                "y_pred": max(value, 0.01),
                "y_pred_q05": np.nan,
                "y_pred_q95": np.nan,
                "model": "sarimax",
                "split": item["split"],
                "is_shock": bool(item["is_shock"]),
                "season": item["season"],
                "target_form": "price",
                "feature_set": "sarimax_gap_refit",
            }
        )


def _check(checks: list, verify: pd.DataFrame, crop: str, market: str, issue: pd.Timestamp, forecast: np.ndarray, positions: dict, origin: int) -> None:
    part = verify[pd.to_datetime(verify["forecast_issue_week"]).eq(issue)]
    for _, item in part.iterrows():
        horizon = int(item["horizon"])
        target = pd.Timestamp(item["target_week"])
        steps = int(positions[target] - origin)
        index = steps - 1
        checks.append(float(forecast[index]) - float(item["y_pred"]))


def _missing_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    trees = _stamp(predictions[predictions["model"].eq("lightgbm") & predictions["split"].isin(["calibrate", "test"])])
    sarimax = _stamp(predictions[predictions["model"].eq("sarimax") & predictions["split"].isin(["calibrate", "test"])])
    trees["issue_key"] = trees["forecast_issue_week"].dt.strftime("%Y-%m-%d")
    sarimax["issue_key"] = sarimax["forecast_issue_week"].dt.strftime("%Y-%m-%d")
    merged = trees.merge(
        sarimax[["crop", "market", "issue_key", "horizon"]],
        on=["crop", "market", "issue_key", "horizon"],
        how="left",
        indicator=True,
    )
    missing = merged[merged["_merge"].eq("left_only")].drop(columns=["_merge", "issue_key"])
    return missing.reset_index(drop=True)


def main() -> None:
    predictions = _stamp(pd.read_csv(PREDICTIONS))
    before = _overlap(predictions)
    print(json.dumps({k: before[k] for k in ("counts", "exact_match_before_gap_fill", "missing_test_rows")}, indent=2), flush=True)
    cqr_rows, cqr_payload = _cqr(predictions)
    (A_DIR / "cqr.json").write_text(json.dumps(cqr_payload, indent=2), encoding="utf-8")
    _write_metrics_with_cqr(cqr_rows)
    feature = pd.read_parquet(FEATURE_PATH)
    missing = _missing_rows(predictions)
    cached = predictions[predictions["model"].eq("sarimax")]
    made, replay = _replay(feature, missing, cached)
    if len(made) != len(missing):
        raise AssertionError(f"filled {len(made)} SARIMAX gaps, expected {len(missing)}")
    made.to_csv(A_DIR / "sarimax_gap_predictions.csv", index=False)
    filled_test = len(made[made["split"].eq("test")])
    exact_after = before["missing_test_rows"] == filled_test and before["extra_sarimax_test_rows"] == 0
    payload = {
        **before,
        "exact_match": before["exact_match_before_gap_fill"],
        "reuse_acceptable_without_refit": before["exact_match_before_gap_fill"],
        "decision": (
            "Cached SARIMAX keys match LightGBM and XGBoost exactly on the 2024-2025 test. Reuse is acceptable."
            if before["exact_match_before_gap_fill"]
            else "Cached SARIMAX keys are a subset of the LightGBM and XGBoost keys. Matching rows were reused from artifacts/latest. Missing issue weeks were refit with the saved order; artifacts/latest was not rewritten."
        ),
        "gap_fill": {
            **replay,
            "rows_written": int(len(made)),
            "test_rows_written": int(filled_test),
            "calibrate_rows_written": int((made["split"] == "calibrate").sum()),
            "path": str(A_DIR / "sarimax_gap_predictions.csv"),
            "reason": "SARIMAX skipped issue weeks whose feature-frame price was not observed. The trees still score those Mondays when an earlier price and the target price were observed.",
        },
        "exact_match_after_gap_fill": bool(exact_after and before["lightgbm_equals_xgboost"]),
        "predictions_csv_unchanged": True,
        "latest_unchanged": True,
    }
    (A_DIR / "sarimax_comparability.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["gap_fill"], indent=2), flush=True)
    print(payload["decision"], flush=True)
    print(f"exact_match_after_gap_fill={payload['exact_match_after_gap_fill']}", flush=True)


if __name__ == "__main__":
    main()
