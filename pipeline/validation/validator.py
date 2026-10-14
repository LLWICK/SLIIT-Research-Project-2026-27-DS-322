"""Data quality validator for cleaned weather observations.

Validates date continuity, domain physical bounds, absence of duplicates,
and reports completeness statistics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("validation.weather")

# Reasonable physical meteorological boundaries for Sri Lanka
PHYSICAL_BOUNDS = {
    "precipitation_mm": (0.0, 700.0),       # mm/day (Sri Lanka daily record ~500-600mm)
    "temp_mean_c": (0.0, 48.0),             # °C (Nuwara Eliya near-frost to dry zone heatwave)
    "temp_min_c": (-2.0, 40.0),            # °C
    "temp_max_c": (5.0, 50.0),              # °C
    "relative_humidity_pct": (0.0, 100.0),  # %
    "wind_speed_ms": (0.0, 60.0),           # m/s
}


class WeatherValidator:
    """Validates cleaned weather data against schema and physical bounds."""

    def __init__(self, bounds: dict[str, tuple[float, float]] = PHYSICAL_BOUNDS) -> None:
        self.bounds = bounds

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run comprehensive validation suite on cleaned weather DataFrame.

        Returns:
            Validation report dictionary with status, metrics, and any warnings.
        """
        if df.empty:
            return {
                "is_valid": False,
                "total_rows": 0,
                "errors": ["DataFrame is empty."],
                "warnings": [],
                "completeness": {},
            }

        errors: list[str] = []
        warnings: list[str] = []

        # 1. Check required columns
        required_cols = [
            "obs_date",
            "district",
            "year",
            "month",
            "season_type",
            "season_year",
        ]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # 2. Check for duplicate district + obs_date combinations
        dups = df.duplicated(subset=["district", "obs_date"]).sum()
        if dups > 0:
            errors.append(f"Found {dups} duplicate district-date observations.")

        # 3. Check date continuity
        date_series = pd.to_datetime(df["obs_date"], errors="coerce")
        if date_series.isna().any():
            errors.append("Found unparseable dates in 'obs_date' column.")
        else:
            sorted_dates = date_series.sort_values()
            expected_range = pd.date_range(sorted_dates.min(), sorted_dates.max(), freq="D")
            missing_dates = expected_range.difference(sorted_dates)
            if len(missing_dates) > 0:
                warnings.append(
                    f"Date continuity gap: {len(missing_dates)} calendar days missing "
                    f"between {sorted_dates.min().date()} and {sorted_dates.max().date()}."
                )

        # 4. Check temp_max >= temp_min
        if "temp_max_c" in df.columns and "temp_min_c" in df.columns:
            inconsistent_temps = (df["temp_max_c"] < df["temp_min_c"]).sum()
            if inconsistent_temps > 0:
                errors.append(
                    f"Found {inconsistent_temps} rows where temp_max_c < temp_min_c."
                )

        # 5. Domain physical bounds and completeness per measure column
        completeness: dict[str, dict[str, Any]] = {}
        for col, (b_min, b_max) in self.bounds.items():
            if col not in df.columns:
                continue

            vals = df[col]
            n_total = len(vals)
            n_missing = int(vals.isna().sum())
            n_valid = n_total - n_missing
            comp_ratio = round((n_valid / n_total) * 100.0, 2) if n_total > 0 else 0.0

            valid_vals = vals.dropna()
            out_of_bounds = int(((valid_vals < b_min) | (valid_vals > b_max)).sum())

            if out_of_bounds > 0:
                warnings.append(
                    f"Column '{col}' has {out_of_bounds} values outside physical bounds [{b_min}, {b_max}]."
                )

            completeness[col] = {
                "total_observations": n_total,
                "valid_observations": n_valid,
                "missing_observations": n_missing,
                "completeness_pct": comp_ratio,
                "out_of_bounds_count": out_of_bounds,
                "min_observed": float(valid_vals.min()) if len(valid_vals) else None,
                "max_observed": float(valid_vals.max()) if len(valid_vals) else None,
                "mean_observed": round(float(valid_vals.mean()), 2) if len(valid_vals) else None,
            }

        is_valid = len(errors) == 0

        report = {
            "is_valid": is_valid,
            "district": str(df["district"].iloc[0]) if "district" in df.columns else "unknown",
            "total_rows": len(df),
            "date_range": {
                "start": str(df["obs_date"].min()) if "obs_date" in df.columns else None,
                "end": str(df["obs_date"].max()) if "obs_date" in df.columns else None,
            },
            "errors": errors,
            "warnings": warnings,
            "completeness": completeness,
        }

        if is_valid:
            logger.info(f"Validation PASSED for {report['district']} ({len(df)} rows).")
        else:
            logger.error(f"Validation FAILED for {report['district']}: {errors}")

        return report
