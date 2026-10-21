"""Conformal helper for the runtime model.

Experiments A, B, and C are trained by their run_experiment modules. Running this
file does not fit a second SARIMAX cache.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from htfe.training.evaluate import _block
from htfe.training.methodology_fit import _conformal_quantile


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


def main() -> None:
    raise RuntimeError(
        "Experiment A is trained with python -u -m htfe.training.run_experiment_a. "
        "This module only keeps the conformal helper used by the runtime model."
    )


if __name__ == "__main__":
    main()
