"""Seasonal ARIMA rolling forecasts.

The order is fixed at (1, 1, 1) x (1, 0, 0, 52). It is not selected by AutoARIMA.
That order is walked forward on every evaluation series with statsmodels,
appending each new week without a full refit.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from htfe.training.config import EVAL_MARKETS, HORIZONS, TRAIN_END

DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL = (1, 0, 0, 52)
FIXED_NOTE = "fixed order (1, 1, 1) x (1, 0, 0, 52)"


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
    order = DEFAULT_ORDER if order is None else order
    seasonal = DEFAULT_SEASONAL if seasonal is None else seasonal
    meta = {
        "order": list(order),
        "seasonal_order": list(seasonal),
        "note": FIXED_NOTE,
    }

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


def fit_notebook_from_table(table: pd.DataFrame, partial_path=None) -> tuple[pd.DataFrame, dict]:
    """Notebook 1 SARIMAX: fixed (1,1,1)x(1,0,0,52), append without refit.

    The state is the wholesale week, with 1–2 week holes filled the same way as
    the analysis panel, then forward-filled after the first usable week. 2023
    and 2024–2025 issue weeks are scored. The target must be an observed price.
    """
    from htfe.data.build.build_analysis_ready_panel import _calendar_grid, filter_harti
    from htfe.data.build.missingness import apply_missingness
    from htfe.config import processed_dir
    from htfe.data.build.scope import CROPS
    from htfe.training.config import SHOCK_END, SHOCK_START
    from htfe.training.run_experiment_a import EVAL_IDS, MARKET_NAME

    weekly = pd.read_csv(processed_dir() / "harti_prices_weekly_long.csv", low_memory=False)
    panel = apply_missingness(_calendar_grid(filter_harti(weekly)))
    panel = panel[panel["market"].isin(EVAL_MARKETS) & panel["crop"].isin(CROPS)].copy()
    panel["week_start"] = pd.to_datetime(panel["week_start"])
    h1 = table[table["forecast_horizon_weeks"].eq(1) & table["market_id"].astype(str).isin(EVAL_IDS)].copy()
    h1["forecast_issue_week"] = pd.to_datetime(h1["forecast_issue_week"])
    h1["crop"] = h1["crop_id"].astype(str)
    h1["market"] = h1["market_id"].astype(str).map(MARKET_NAME)
    season = h1[["crop", "market", "forecast_issue_week", "season"]].drop_duplicates(
        ["crop", "market", "forecast_issue_week"]
    )
    panel = panel.merge(season, left_on=["crop", "market", "week_start"], right_on=["crop", "market", "forecast_issue_week"], how="left")
    panel["unique_id"] = panel["crop"].astype(str) + "__" + panel["market"].astype(str)
    order, seasonal_order = DEFAULT_ORDER, DEFAULT_SEASONAL
    meta = {
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "note": "fixed order because statsforecast is not installed",
        "protocol": "notebook_append_refit_false",
    }
    rows: list[dict] = []
    for unique_id, group in panel.groupby("unique_id", sort=False):
        group = group.sort_values("week_start").reset_index(drop=True)
        first_valid = group["price_feature"].first_valid_index()
        if first_valid is None:
            print("skip", unique_id, "no usable price state", flush=True)
            continue
        if int(first_valid) > 0:
            group = group.iloc[int(first_valid) :].reset_index(drop=True)
        y_state = group["price_feature"].astype(float).ffill().to_numpy()
        if not np.isfinite(y_state).all():
            raise RuntimeError(f"SARIMAX state still contains missing values for {unique_id}")
        train_idx = np.where(group["week_start"] <= TRAIN_END)[0]
        if len(train_idx) < 80:
            print("skip", unique_id, "train weeks", len(train_idx), flush=True)
            continue
        origin0 = int(train_idx[-1])
        print("SARIMAX", unique_id, "train weeks", len(train_idx), flush=True)
        try:
            result = _fit(y_state[: origin0 + 1], order, seasonal_order)
        except Exception as exc:  # noqa: BLE001
            meta.setdefault("fit_failures", []).append(f"{unique_id}: {exc}")
            continue
        for origin in range(origin0, len(group) - 1):
            if origin > origin0:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = result.append(y_state[origin : origin + 1], refit=False)
                except Exception:
                    break
            issue = pd.Timestamp(group.at[origin, "week_start"])
            if issue <= pd.Timestamp(TRAIN_END):
                split = "train"
            elif issue.year == 2023:
                split = "calibrate"
            elif issue.year in (2024, 2025):
                split = "test"
            else:
                split = "other"
            if split not in {"calibrate", "test"}:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    forecast = np.asarray(result.get_forecast(max(HORIZONS)).predicted_mean, dtype=float)
            except Exception:
                continue
            for horizon in HORIZONS:
                target = origin + horizon
                if target >= len(group) or not bool(group.at[target, "is_observed"]):
                    continue
                y_true = float(group.at[target, "price"])
                if not np.isfinite(y_true) or y_true <= 0:
                    continue
                y_pred = float(forecast[horizon - 1])
                if not np.isfinite(y_pred):
                    continue
                season_value = group.at[origin, "season"]
                rows.append(
                    {
                        "unique_id": unique_id,
                        "crop": group.at[origin, "crop"],
                        "market": group.at[origin, "market"],
                        "forecast_issue_week": issue,
                        "target_week": pd.Timestamp(group.at[target, "week_start"]),
                        "horizon": horizon,
                        "y_true": y_true,
                        "y_pred": max(y_pred, 0.01),
                        "y_pred_q05": np.nan,
                        "y_pred_q95": np.nan,
                        "model": "sarimax",
                        "split": split,
                        "is_shock": bool(issue >= pd.Timestamp(SHOCK_START) and issue <= pd.Timestamp(SHOCK_END)),
                        "season": "" if pd.isna(season_value) else str(season_value),
                        "target_form": "price",
                        "feature_set": "A",
                    }
                )
        if partial_path is not None:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            print("  saved partial SARIMAX rows", len(rows), flush=True)
    return pd.DataFrame(rows), meta
