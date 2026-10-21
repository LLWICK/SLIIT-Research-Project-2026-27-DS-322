"""Split conformal intervals.

The reported tails are ordered around the unchanged q50 forecast. qhat is the
conformal quantile of the 2023 calibration scores only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from htfe.training.evaluate import _block, project_interval_around_median


def conformal_qhat(scores: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * 0.90) / n)
    return float(np.quantile(scores, level, method="higher"))


def cqr_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    qhats: dict[str, dict[str, float]] = {}
    for model in ("lightgbm", "xgboost"):
        qhats[model] = {}
        block = predictions[predictions["model"].eq(model)]
        if block.empty:
            continue
        for horizon, group in block.groupby("horizon"):
            calibrate = group[group["split"].eq("calibrate")]
            test = group[group["split"].eq("test")]
            cal_lo, _, cal_hi = project_interval_around_median(
                calibrate["y_pred_q05"].to_numpy(float),
                calibrate["y_pred"].to_numpy(float),
                calibrate["y_pred_q95"].to_numpy(float),
            )
            test_lo, test_med, test_hi = project_interval_around_median(
                test["y_pred_q05"].to_numpy(float),
                test["y_pred"].to_numpy(float),
                test["y_pred_q95"].to_numpy(float),
            )
            y_cal = calibrate["y_true"].to_numpy(float)
            scores = np.maximum(cal_lo - y_cal, y_cal - cal_hi)
            qhat = conformal_qhat(scores)
            qhats[model][str(int(horizon))] = round(qhat, 4)
            raw = _block(test["y_true"].to_numpy(float), test_med, test_lo, test_hi)
            calibrated = _block(
                test["y_true"].to_numpy(float),
                test_med,
                test_lo - qhat,
                test_hi + qhat,
            )
            raw_q05 = test["y_pred_q05"].to_numpy(float)
            raw_q95 = test["y_pred_q95"].to_numpy(float)
            raw_cross = (raw_q05 > test_med) | (test_med > raw_q95)
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
                    "raw_quantile_crossing_count": int(raw_cross.sum()),
                    "raw_quantile_crossing_rate_pct": float(raw_cross.mean() * 100) if len(test) else None,
                    "reported_interval_ordered": True,
                }
            )
    payload = {
        "method": "split_cqr_with_ordered_interval_projection",
        "alpha": 0.1,
        "quantile_ordering": {
            "raw_predictions_retained_for_diagnostic": True,
            "reported_interval_rule": "lower=min(q05,q50,q95); median=q50 unchanged; upper=max(q05,q50,q95)",
            "point_forecast_changed": False,
        },
        "model_selection_validation": "2022 target weeks inside the training period; 2023 is not used for early stopping",
        "final_tree_fit": "all eligible pre-2023 training rows after tree-count selection",
        "calibration": "2023 issue weeks only; reserved exclusively for CQR",
        "test": "2024-2025 issue weeks; ordered interval projected before CQR; q50 point forecasts unchanged",
        "qhat_by_model": qhats,
        "by_model_horizon": rows,
        "limit": "Time-ordered prices are not exchangeable, so this is empirical calibration rather than a universal guarantee.",
        "intended_coverage": 0.90,
    }
    return pd.DataFrame(rows), payload
