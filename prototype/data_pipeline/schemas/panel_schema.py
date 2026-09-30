"""Pandera checks for the wholesale weekly analysis panel."""
from __future__ import annotations

import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:  # pragma: no cover
    import pandera as pa


def schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "crop": pa.Column(str, nullable=False),
            "market": pa.Column(str, nullable=False),
            "price_type": pa.Column(str, nullable=False),
            "week_start": pa.Column(nullable=False),
            "year": pa.Column(int, nullable=False),
            "week": pa.Column(int, nullable=False),
            "price": pa.Column(float, nullable=True),
            "is_observed": pa.Column(bool, nullable=False),
            "is_imputed": pa.Column(bool, nullable=False),
            "is_outlier_flag": pa.Column(bool, nullable=False),
            "missing_run_length": pa.Column(int, nullable=False),
            "disruption_flag": pa.Column(str, nullable=False),
            "season": pa.Column(str, nullable=False),
            "commitment_source": pa.Column(str, nullable=False),
        },
        unique=["crop", "market", "week_start"],
        strict=False,
        coerce=True,
    )


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    checked = frame.copy()
    checked["week_start"] = pd.to_datetime(checked["week_start"]).astype("datetime64[ns]")
    if checked["week_start"].dt.dayofweek.ne(0).any():
        raise ValueError("week_start must be Monday for every row")
    if not (checked["price_type"] == "wholesale").all():
        raise ValueError("panel is wholesale only")
    return schema().validate(checked)
