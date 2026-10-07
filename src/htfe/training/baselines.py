"""Last-value and lag-52 seasonal naive forecasts at each horizon."""
from __future__ import annotations

import numpy as np
import pandas as pd

from htfe.training.config import EVAL_MARKETS, HORIZONS, TRAIN_END


def predict(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    scoped = frame[frame["market"].isin(EVAL_MARKETS)].copy()
    for unique_id, group in scoped.groupby("unique_id", sort=False):
        group = group.sort_values("week_start")
        price = group["price"].to_numpy(dtype=float)
        observed = group["is_observed"].to_numpy(dtype=bool)
        weeks = group["week_start"].to_numpy()
        crops = group["crop"].to_numpy()
        markets = group["market"].to_numpy()
        splits = group["split"].to_numpy()
        shocks = group["is_shock"].to_numpy()
        seasons = group["season"].to_numpy()
        for index in range(len(group)):
            if splits[index] not in {"calibrate", "test"}:
                continue
            if not observed[index] or not np.isfinite(price[index]) or price[index] <= 0:
                continue
            # Do not score an origin whose horizon target falls inside the training window.
            for horizon in HORIZONS:
                target = index + horizon
                if target >= len(group) or not observed[target] or not np.isfinite(price[target]):
                    continue
                target_week = pd.Timestamp(weeks[target])
                if splits[index] == "train" and target_week <= pd.Timestamp(TRAIN_END):
                    continue
                common = {
                    "unique_id": unique_id,
                    "crop": crops[index],
                    "market": markets[index],
                    "week_start": pd.Timestamp(weeks[index]),
                    "target_week": target_week,
                    "horizon": horizon,
                    "y_true": float(price[target]),
                    "split": splits[index],
                    "is_shock": bool(shocks[index]),
                    "season": seasons[index],
                    "y_pred_q05": np.nan,
                    "y_pred_q95": np.nan,
                    "target_form": "level",
                }
                rows.append({**common, "model": "naive_last", "y_pred": float(price[index])})
                if target >= 52 and observed[target - 52] and np.isfinite(price[target - 52]):
                    rows.append({**common, "model": "seasonal_naive_52", "y_pred": float(price[target - 52])})
    return pd.DataFrame(rows)


def _block(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ok = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    if not ok.any():
        return {"n_scored": 0, "mae": None, "rmse": None, "mape": None}
    error = y_pred[ok] - y_true[ok]
    return {
        "n_scored": int(ok.sum()),
        "mae": round(float(np.mean(np.abs(error))), 2),
        "rmse": round(float(np.sqrt(np.mean(error**2))), 2),
        "mape": round(float(np.mean(np.abs(error) / y_true[ok]) * 100), 2),
    }


def score_same_rows(reference: pd.DataFrame, table: pd.DataFrame) -> dict[str, dict]:
    """Score naive forecasts on the same issue weeks as an already scored model.

    Last-value is the latest published price strictly before the issue Monday.
    Seasonal naive is the published price 52 weeks before the target Monday.
    A target with no published price 52 weeks earlier is left out of that model only.
    """
    scored = reference.copy()
    scored["crop_id"] = scored["crop"].astype(str).str.strip().str.lower()
    scored["market_id"] = scored["market"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    scored["issue"] = pd.to_datetime(scored["forecast_issue_week"])
    scored["target"] = pd.to_datetime(scored["target_week"])

    origin = table.copy()
    origin["issue"] = pd.to_datetime(origin["forecast_issue_week"])
    origin["target"] = pd.to_datetime(origin["target_week"])
    origin = origin.drop_duplicates(["crop_id", "market_id", "issue", "target"])
    matched = scored.merge(
        origin[["crop_id", "market_id", "issue", "target", "last_observed_price_lkr_kg"]],
        on=["crop_id", "market_id", "issue", "target"],
        how="left",
    )

    observed = table.loc[table["target_is_observed"].astype(bool), ["crop_id", "market_id", "target_week", "target_price_lkr_kg"]]
    observed = observed.copy()
    observed["target"] = pd.to_datetime(observed["target_week"])
    observed = observed.drop_duplicates(["crop_id", "market_id", "target"])
    prices = {
        (row.crop_id, row.market_id, row.target): float(row.target_price_lkr_kg)
        for row in observed.itertuples(index=False)
    }
    seasonal = [
        prices.get((crop, market, target - pd.Timedelta(weeks=52)), np.nan)
        for crop, market, target in zip(matched["crop_id"], matched["market_id"], matched["target"], strict=True)
    ]
    y_true = matched["y_true"].to_numpy(dtype=float)
    return {
        "naive_last": _block(y_true, matched["last_observed_price_lkr_kg"].to_numpy(dtype=float)),
        "seasonal_naive_52": _block(y_true, np.asarray(seasonal, dtype=float)),
    }
