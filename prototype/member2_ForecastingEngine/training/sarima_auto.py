"""Seasonal ARIMA rolling forecasts.

Order is chosen with AutoARIMA on carrot__Dambulla through 2022 (the smoke test).
That order is then walked forward on every evaluation series with statsmodels,
appending each new week without a full refit. A fresh AutoARIMA search on every
origin would not finish in a progress-presentation run.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from training.config import EVAL_MARKETS, HORIZONS, TRAIN_END

DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL = (1, 0, 0, 52)


def select_order(y: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int, int], str]:
    """Return (order, seasonal_order, note) from one AutoARIMA fit, or the default."""
    try:
        from statsforecast.models import AutoARIMA
    except ImportError:
        return DEFAULT_ORDER, DEFAULT_SEASONAL, "statsforecast not installed; fixed SARIMAX order"

    model = AutoARIMA(
        season_length=52,
        approximation=True,
        stepwise=True,
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
        max_d=1,
        max_D=1,
    )
    try:
        model.fit(np.asarray(y, dtype=float))
        fitted = model.model_
        order = tuple(int(v) for v in fitted.order)
        seasonal = tuple(int(v) for v in fitted.seasonal_order)
        return order, seasonal, "AutoARIMA on carrot__Dambulla train"
    except Exception as exc:  # noqa: BLE001
        return DEFAULT_ORDER, DEFAULT_SEASONAL, f"AutoARIMA failed ({exc}); fixed SARIMAX order"


def _fit(y: np.ndarray, order, seasonal):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=50)


def predict(frame: pd.DataFrame, order=None, seasonal=None) -> tuple[pd.DataFrame, dict]:
    scoped = frame[frame["market"].isin(EVAL_MARKETS)].copy()
    smoke = scoped[(scoped["unique_id"] == "carrot__Dambulla") & (scoped["week_start"] <= TRAIN_END)]
    smoke_y = smoke["price_feature"].astype(float).ffill().bfill().to_numpy()
    if order is None or seasonal is None:
        order, seasonal, note = select_order(smoke_y)
    else:
        note = "order supplied by caller"
    meta = {"order": list(order), "seasonal_order": list(seasonal), "note": note, "smoke_series": "carrot__Dambulla"}

    rows: list[dict] = []
    for unique_id, group in scoped.groupby("unique_id", sort=False):
        group = group.sort_values("week_start").reset_index(drop=True)
        y_state = group["price_feature"].astype(float).ffill().bfill().to_numpy()
        train_idx = np.where(group["week_start"] <= TRAIN_END)[0]
        if len(train_idx) < 80:
            continue
        origin0 = int(train_idx[-1])
        try:
            result = _fit(y_state[: origin0 + 1], order, seasonal)
        except Exception as exc:  # noqa: BLE001
            meta.setdefault("fit_failures", []).append(f"{unique_id}: {exc}")
            continue
        max_h = max(HORIZONS)
        for origin in range(origin0, len(group) - 1):
            if origin > origin0:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = result.append(y_state[origin : origin + 1], refit=False)
                except Exception:
                    break
            split = group.at[origin, "split"]
            if split not in {"calibrate", "test"}:
                continue
            if not bool(group.at[origin, "is_observed"]):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    forecast = np.asarray(result.get_forecast(max_h).predicted_mean, dtype=float)
            except Exception:
                continue
            for horizon in HORIZONS:
                target = origin + horizon
                if target >= len(group) or not bool(group.at[target, "is_observed"]):
                    continue
                y_true = float(group.at[target, "price"])
                if not np.isfinite(y_true) or y_true <= 0 or horizon - 1 >= len(forecast):
                    continue
                y_pred = float(forecast[horizon - 1])
                if not np.isfinite(y_pred):
                    continue
                rows.append(
                    {
                        "unique_id": unique_id,
                        "crop": group.at[origin, "crop"],
                        "market": group.at[origin, "market"],
                        "week_start": pd.Timestamp(group.at[origin, "week_start"]),
                        "target_week": pd.Timestamp(group.at[target, "week_start"]),
                        "horizon": horizon,
                        "y_true": y_true,
                        "y_pred": max(y_pred, 0.01),
                        "y_pred_q05": np.nan,
                        "y_pred_q95": np.nan,
                        "model": "sarimax",
                        "split": split,
                        "is_shock": bool(group.at[origin, "is_shock"]),
                        "season": group.at[origin, "season"],
                        "target_form": "level",
                    }
                )
    return pd.DataFrame(rows), meta
