"""Last-value and lag-52 seasonal naive forecasts at each horizon."""
from __future__ import annotations

import numpy as np
import pandas as pd

from training.config import EVAL_MARKETS, HORIZONS, TRAIN_END


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
