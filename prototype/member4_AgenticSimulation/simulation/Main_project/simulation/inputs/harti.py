"""Colombo wholesale season prices from the local warehouse. Read only."""

from __future__ import annotations

import duckdb
import pandas as pd

from data_layer.paths import DUCKDB_PATH
from simulation.config.settings import season_ordinal

MIN_WEEKS = 15
MARKET_CODE = "colombo"
PRICE_TYPE = "wholesale"


def colombo_wholesale_seasons(min_weeks: int = MIN_WEEKS) -> pd.DataFrame:
    """One row per crop per season. The price is the mean of the weeks in that season.

    Weeks with no HARTI price are left out of the mean. A season with fewer than
    min_weeks priced weeks is left out. Weeks that fall in two seasons have no
    season label and are left out.
    """

    if min_weeks < 1:
        raise ValueError("min_weeks must be at least 1")
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(f"local warehouse not found: {DUCKDB_PATH}")
    connection = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        frame = connection.execute(
            """
            SELECT
                crop_code,
                season_code,
                season_label,
                AVG(harti_price) AS price,
                COUNT(*) AS n_weeks
            FROM curated.crop_market_week
            WHERE market_code = ?
              AND price_type = ?
              AND harti_price IS NOT NULL
              AND season_label IS NOT NULL
            GROUP BY crop_code, season_code, season_label
            HAVING COUNT(*) >= ?
            ORDER BY crop_code, season_label
            """,
            [MARKET_CODE, PRICE_TYPE, min_weeks],
        ).fetchdf()
    finally:
        connection.close()
    if frame.empty:
        raise ValueError("the warehouse returned no Colombo wholesale seasons")
    frame["price"] = frame["price"].astype(float)
    frame["n_weeks"] = frame["n_weeks"].astype(int)
    frame["ordinal"] = [
        season_ordinal(code, label)
        for code, label in zip(frame["season_code"], frame["season_label"], strict=True)
    ]
    return frame.sort_values(["crop_code", "ordinal"]).reset_index(drop=True)
