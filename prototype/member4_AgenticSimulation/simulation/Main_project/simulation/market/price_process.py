"""Season price that keeps part of last season and then takes a shock.

The clearing price is the part that moves when planted area moves.
Persistence is estimated from HARTI. It is clipped to the range 0 to 1,
so a negative estimate cannot be used to invent a cobweb.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from simulation.calibration.compare import consecutive_pairs


@dataclass(frozen=True)
class PriceProcess:
    crop_code: str
    center: float
    persistence: float
    shock_std: float


def combine_price(last_price: float, cleared_price: float, persistence: float, shock: float) -> float:
    """Rs/kg. The result is never negative."""

    if not 0 <= persistence <= 1:
        raise ValueError("persistence must be between 0 and 1")
    return max(0.0, persistence * last_price + (1.0 - persistence) * cleared_price + shock)


def fit_price_process(crop_rows: pd.DataFrame) -> PriceProcess:
    """AR(1) around the mean of the observed season prices."""

    if crop_rows.empty:
        raise ValueError("no season prices to fit")
    crop_code = str(crop_rows["crop_code"].iloc[0])
    pairs = consecutive_pairs(crop_rows)
    if len(pairs) < 4:
        raise ValueError(f"{crop_code} needs at least 4 consecutive season pairs")
    center = float(crop_rows["price"].mean())
    previous = pairs["prev_price"].astype(float) - center
    current = pairs["price"].astype(float) - center
    variance = float((previous * previous).mean())
    persistence = 0.0 if variance == 0 else float((previous * current).mean()) / variance
    persistence = min(1.0, max(0.0, persistence))
    residual = current - persistence * previous
    shock_std = float(residual.std(ddof=1))
    if shock_std < 0 or shock_std != shock_std:
        raise ValueError(f"{crop_code} shock scale is not usable")
    return PriceProcess(crop_code, center, persistence, shock_std)
