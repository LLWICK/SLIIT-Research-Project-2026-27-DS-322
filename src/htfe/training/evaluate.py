"""Score originally observed prices only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _block(y_true: np.ndarray, y_pred: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    error = y_pred - y_true
    ape = np.abs(error) / np.clip(y_true, 1e-6, None)
    out = {
        "n_scored": int(len(y_true)),
        "mae": round(float(np.mean(np.abs(error))), 2) if len(y_true) else None,
        "rmse": round(float(np.sqrt(np.mean(error**2))), 2) if len(y_true) else None,
        "mape": round(float(np.mean(ape) * 100), 2) if len(y_true) else None,
    }
    if np.isfinite(low).any() and np.isfinite(high).any():
        covered = (y_true >= low) & (y_true <= high)
        width = high - low
        residual = y_true - y_pred
        pinball = np.mean(np.where(residual >= 0, 0.5 * residual, -0.5 * residual))
        out["pinball"] = round(float(pinball), 2)
        out["picp"] = round(float(np.nanmean(covered) * 100), 2)
        out["interval_width"] = round(float(np.nanmean(width)), 2)
    else:
        out["pinball"] = None
        out["picp"] = None
        out["interval_width"] = None
    return out


def _arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        frame["y_true"].to_numpy(dtype=float),
        frame["y_pred"].to_numpy(dtype=float),
        frame["y_pred_q05"].to_numpy(dtype=float),
        frame["y_pred_q95"].to_numpy(dtype=float),
    )


def metrics_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = predictions[predictions["split"].eq("test")].copy()
    by_model = []
    by_series = []
    for (model, horizon, target_form), group in test.groupby(["model", "horizon", "target_form"], sort=True):
        block = _block(*_arrays(group))
        by_model.append({"model": model, "horizon": int(horizon), "target_form": target_form, "slice": "test", **block})
        shock = group[group["is_shock"].astype(bool)]
        if not shock.empty:
            by_model.append(
                {"model": model, "horizon": int(horizon), "target_form": target_form, "slice": "shock_2024q1", **_block(*_arrays(shock))}
            )
        for (unique_id, crop, market), series in group.groupby(["unique_id", "crop", "market"]):
            by_series.append(
                {
                    "model": model,
                    "horizon": int(horizon),
                    "target_form": target_form,
                    "unique_id": unique_id,
                    "crop": crop,
                    "market": market,
                    **_block(*_arrays(series)),
                }
            )
    return pd.DataFrame(by_model), pd.DataFrame(by_series)


def slice_metrics(predictions: pd.DataFrame, model: str, target_form: str = "logret") -> dict:
    """Headline metrics pooled over horizons, for the studio cards."""
    group = predictions[
        predictions["model"].eq(model) & predictions["split"].eq("test") & predictions["target_form"].eq(target_form)
    ]
    overall = _block(*_arrays(group)) if not group.empty else _block(np.array([]), np.array([]), np.array([]), np.array([]))
    by_crop = {}
    by_market = {}
    by_season = {}
    for crop, part in group.groupby("crop"):
        by_crop[str(crop)] = _block(*_arrays(part))
    for market, part in group.groupby("market"):
        by_market[str(market).lower().replace(" ", "_")] = _block(*_arrays(part))
    for season, part in group.groupby(group["season"].str.lower()):
        by_season[str(season)] = _block(*_arrays(part))
    return {"overall": overall, "by_crop": by_crop, "by_market": by_market, "by_season": by_season}
