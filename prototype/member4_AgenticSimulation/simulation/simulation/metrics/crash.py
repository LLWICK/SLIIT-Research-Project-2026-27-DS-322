"""Season-level price crashes. See CrashRule in simulation.config.settings."""

from __future__ import annotations

import pandas as pd


def season_is_crash(price: float, previous_price: float | None, drop_fraction: float) -> bool:
    """True when price is at least drop_fraction below a positive previous price."""

    if not 0 < drop_fraction <= 1:
        raise ValueError("drop_fraction must be between 0 and 1")
    if previous_price is None or previous_price <= 0:
        return False
    return price <= previous_price * (1.0 - drop_fraction)


def crash_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Number of crash seasons per crop. The frame is one baseline run."""

    counts: dict[str, int] = {}
    for crop_code, rows in frame.groupby("crop_code", sort=True):
        counts[str(crop_code)] = int(rows["is_crash"].sum())
    return counts
