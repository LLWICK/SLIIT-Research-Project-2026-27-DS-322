"""Missing-week flags. Short holes may be filled for features; targets stay observed-only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_missingness(panel: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for (_, _), group in panel.groupby(["crop", "market"], sort=False):
        pieces.append(_one_series(group.sort_values("week_start").copy()))
    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values(["crop", "market", "week_start"]).reset_index(drop=True)


def _one_series(group: pd.DataFrame) -> pd.DataFrame:
    observed = group["price"].notna().to_numpy()
    n = len(group)
    run_length = np.zeros(n, dtype=int)
    disruption = np.array(["none"] * n, dtype=object)
    feature = group["price"].to_numpy(dtype=float).copy()

    index = 0
    while index < n:
        if observed[index]:
            index += 1
            continue
        end = index
        while end < n and not observed[end]:
            end += 1
        length = end - index
        for pos in range(index, end):
            run_length[pos] = pos - index + 1
        flag = _gap_flag(group.iloc[index:end], length)
        if flag:
            disruption[index:end] = flag
        if flag is None and length <= 2:
            left = feature[index - 1] if index > 0 and np.isfinite(feature[index - 1]) else np.nan
            right = feature[end] if end < n and np.isfinite(feature[end]) else np.nan
            for step, pos in enumerate(range(index, end), start=1):
                if np.isfinite(left) and np.isfinite(right):
                    feature[pos] = left + (right - left) * step / (length + 1)
                elif np.isfinite(left):
                    feature[pos] = left
                elif np.isfinite(right):
                    feature[pos] = right
        index = end

    group["missing_run_length"] = run_length
    group["disruption_flag"] = disruption
    group["price_feature"] = feature
    group["is_imputed"] = (~group["price"].notna()) & np.isfinite(feature)
    group["is_observed"] = group["price"].notna()
    # A filled feature must not leak into the published price column.
    return group


def _gap_flag(gap: pd.DataFrame, length: int) -> str | None:
    if length < 3:
        return None
    starts = pd.to_datetime(gap["week_start"])
    if ((starts.dt.year == 2017) & (gap["week"].to_numpy() <= 12)).any():
        return "collection_gap_2017"
    if starts.dt.year.isin([2020, 2021]).any():
        return "covid"
    return None
