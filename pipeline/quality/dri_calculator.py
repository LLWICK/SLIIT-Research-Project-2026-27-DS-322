"""Data Reliability Index (DRI) Calculator for Member 1 Data Engineering.

Evaluates and audits factual observation datasets (market price, weather, cultivation)
across three fundamental dimensions of empirical quality:
1. Completeness: Ratio of valid, non-null, non-zero observations.
2. Timeliness / Recency: Temporal freshness of latest observation relative to evaluation date.
3. Consistency: Adherence to domain economic bounds and sanity constraints without anomalies.

Methodological Notice:
The baseline composite formula, component normalizations, and default weights below
represent engineering implementation choices designed for Sri Lankan agricultural markets.
These parameters are fully configurable via `DRIConfig` and should be reviewed and
approved by the research team for individual study phases.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pipeline.config import PROCESSED_PRICE_DIR

logger = logging.getLogger("quality.dri")


@dataclass
class DRIConfig:
    """Configurable parameter registry for the Data Reliability Index."""

    # Component weights (must sum to 1.0)
    weight_completeness: float = 0.40
    weight_timeliness: float = 0.30
    weight_consistency: float = 0.30

    # Timeliness parameters
    # Maximum acceptable observation latency in calendar days before recency score decays to zero
    max_acceptable_delay_days: int = 30

    # Quality classification thresholds
    high_reliability_threshold: float = 80.0
    moderate_reliability_threshold: float = 60.0

    # Minimum observations required to avoid insufficient data penalty
    min_observations_required: int = 5

    def __post_init__(self) -> None:
        """Validate parameter ranges and weight integrity."""
        total_weight = round(
            self.weight_completeness + self.weight_timeliness + self.weight_consistency, 4
        )
        if total_weight != 1.0:
            raise ValueError(
                f"DRI weights must sum to 1.0 (got {total_weight}: "
                f"comp={self.weight_completeness}, time={self.weight_timeliness}, cons={self.weight_consistency})"
            )
        if self.max_acceptable_delay_days <= 0:
            raise ValueError("max_acceptable_delay_days must be strictly positive.")


class DRICalculator:
    """Computes transparent, reproducible Data Reliability Index scores for tabular datasets."""

    def __init__(self, config: Optional[DRIConfig] = None) -> None:
        self.config = config or DRIConfig()

    def compute_completeness(self, df: pd.DataFrame, target_col: str) -> tuple[float, dict[str, Any]]:
        """Calculate completeness score as proportion of valid non-null observations.

        Returns:
            Tuple of (score [0.0, 1.0], metadata_dict)
        """
        if df.empty or target_col not in df.columns:
            return 0.0, {"valid": 0, "total": 0, "missing": 0, "ratio": 0.0}

        total = len(df)
        valid_series = df[target_col].dropna()
        # Non-positive numbers for prices or physical measures are considered invalid
        strictly_valid = valid_series[valid_series > 0]
        valid_count = len(strictly_valid)
        missing_count = total - valid_count

        ratio = valid_count / total if total > 0 else 0.0
        # Normalization: direct linear ratio in [0.0, 1.0]
        score = min(max(ratio, 0.0), 1.0)

        meta = {
            "valid_observations": valid_count,
            "total_observations": total,
            "missing_observations": missing_count,
            "completeness_ratio": round(ratio, 4),
        }
        return score, meta

    def compute_timeliness(
        self,
        df: pd.DataFrame,
        date_col: str = "obs_date",
        evaluation_date: Optional[datetime | str] = None,
    ) -> tuple[float, dict[str, Any]]:
        """Calculate recency score using linear decay across acceptable delay window.

        Returns:
            Tuple of (score [0.0, 1.0], metadata_dict)
        """
        if df.empty or date_col not in df.columns:
            return 0.0, {"latest_date": None, "delay_days": None, "score": 0.0}

        parsed_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if parsed_dates.empty:
            return 0.0, {"latest_date": None, "delay_days": None, "score": 0.0, "warning": "No parseable dates"}

        latest_obs = parsed_dates.max()

        if evaluation_date is None:
            eval_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        elif isinstance(evaluation_date, str):
            eval_dt = pd.to_datetime(evaluation_date).to_pydatetime()
        elif isinstance(evaluation_date, datetime):
            eval_dt = evaluation_date
        else:
            eval_dt = datetime.now(timezone.utc).replace(tzinfo=None)

        delay_days = max(0, (eval_dt.date() - latest_obs.date()).days)

        # Linear decay: score = 1.0 at delay 0; score = 0.0 at max_acceptable_delay_days
        max_delay = self.config.max_acceptable_delay_days
        if delay_days <= 1:
            score = 1.0
        elif delay_days >= max_delay:
            score = 0.0
        else:
            score = round(1.0 - (delay_days / max_delay), 4)

        meta = {
            "latest_observation_date": str(latest_obs.date()),
            "evaluation_date": str(eval_dt.date()),
            "latency_days": delay_days,
            "max_window_days": max_delay,
        }
        return score, meta

    def compute_consistency(
        self,
        df: pd.DataFrame,
        validation_report: Optional[dict[str, Any]] = None,
    ) -> tuple[float, dict[str, Any]]:
        """Calculate consistency score penalizing schema errors, boundary breaches, and anomalies.

        Returns:
            Tuple of (score [0.0, 1.0], metadata_dict)
        """
        if df.empty:
            return 0.0, {"violation_count": 0, "score": 0.0}

        total_rows = len(df)
        violations = 0

        # If a pre-computed validation report is provided, leverage its findings
        if validation_report:
            errors_count = len(validation_report.get("errors", []))
            warnings_count = len(validation_report.get("warnings", []))
            # Heavy penalty for hard errors, minor penalty for warnings
            violations += errors_count * 5 + warnings_count * 1
        else:
            # Inline sanity checks
            if "price_lkr_per_kg" in df.columns:
                p = df["price_lkr_per_kg"].dropna()
                violations += int(((p < 5.0) | (p > 5000.0)).sum())

            if "min_price_lkr_per_kg" in df.columns and "max_price_lkr_per_kg" in df.columns:
                both = df.dropna(subset=["min_price_lkr_per_kg", "max_price_lkr_per_kg"])
                violations += int((both["min_price_lkr_per_kg"] > both["max_price_lkr_per_kg"]).sum())

        penalty_ratio = violations / total_rows if total_rows > 0 else 1.0
        score = max(0.0, round(1.0 - min(penalty_ratio, 1.0), 4))

        meta = {
            "violation_units": violations,
            "total_rows": total_rows,
            "penalty_ratio": round(penalty_ratio, 4),
        }
        return score, meta

    def evaluate_price_dataset(
        self,
        df: pd.DataFrame,
        validation_report: Optional[dict[str, Any]] = None,
        evaluation_date: Optional[datetime | str] = None,
        district: str = "National",
        crop: str = "All Crops",
        season_year: Optional[int] = None,
        season_type: Optional[str] = None,
        source_agency: Optional[str] = None,
        source_metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Compute the full Data Reliability Index composite score and audit report.

        Aligns with schema fields in `fact_data_reliability_audit`.
        """
        # Resolve source agency if not explicitly provided
        resolved_source_agency = source_agency
        if not resolved_source_agency and not df.empty and "source_agency" in df.columns:
            non_null_sources = df["source_agency"].dropna()
            if not non_null_sources.empty:
                resolved_source_agency = str(non_null_sources.iloc[0])
        resolved_source_agency = resolved_source_agency or "Unknown"

        validation_is_valid = validation_report.get("is_valid", True) if validation_report else None
        validation_status = "PASSED" if validation_is_valid is True else ("FAILED" if validation_is_valid is False else "NOT_EVALUATED")

        if df.empty:
            return {
                "calculated_dri": 0.0,
                "reliability_tier": "INSUFFICIENT_DATA",
                "validation_status": validation_status,
                "source_agency": resolved_source_agency,
                "district": district,
                "crop": crop,
                "season_year": season_year,
                "season_type": season_type,
                "total_rows": 0,
                "completeness_score": 0.0,
                "timeliness_score": 0.0,
                "consistency_score": 0.0,
                "weights_used": {
                    "completeness": self.config.weight_completeness,
                    "timeliness": self.config.weight_timeliness,
                    "consistency": self.config.weight_consistency,
                },
                "component_metadata": {},
                "source_metadata": source_metadata or {},
                "audit_notes": "Dataset is empty. Zero observations available for evaluation.",
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            }

        # 1. Completeness
        s_comp, meta_comp = self.compute_completeness(df, target_col="price_lkr_per_kg")

        # 2. Timeliness
        s_time, meta_time = self.compute_timeliness(df, date_col="obs_date", evaluation_date=evaluation_date)

        # 3. Consistency
        s_cons, meta_cons = self.compute_consistency(df, validation_report=validation_report)

        # 4. Composite weighted DRI score (0 to 100 scale)
        composite_ratio = (
            self.config.weight_completeness * s_comp
            + self.config.weight_timeliness * s_time
            + self.config.weight_consistency * s_cons
        )
        calculated_dri = round(composite_ratio * 100.0, 2)

        # Sample size safety discount
        notes: list[str] = []
        if len(df) < self.config.min_observations_required:
            calculated_dri = round(calculated_dri * 0.7, 2)
            notes.append(
                f"Small sample penalty applied: {len(df)} rows < {self.config.min_observations_required} required."
            )

        # Reliability Tier assignment
        if calculated_dri >= self.config.high_reliability_threshold:
            tier = "HIGH"
            notes.append("Dataset meets rigorous quality standards for production DSS model feeding.")
        elif calculated_dri >= self.config.moderate_reliability_threshold:
            tier = "MODERATE"
            notes.append("Dataset has moderate confidence; usable with monitoring or fallback checks.")
        else:
            tier = "LOW"
            notes.append("Low reliability; potential data gap, staleness, or anomalies detected.")

        # Guard: Do not silently treat invalid data as reliable
        if validation_is_valid is False:
            notes.append(
                f"Validation status is FAILED with {len(validation_report.get('errors', []))} error(s)."
            )
            if tier == "HIGH":
                tier = "MODERATE"
                notes.append("Reliability tier downgraded from HIGH to MODERATE due to validation failure.")

        audit_record = {
            "calculated_dri": calculated_dri,
            "reliability_tier": tier,
            "validation_status": validation_status,
            "source_agency": resolved_source_agency,
            "district": district,
            "crop": crop,
            "season_year": season_year or (int(df["season_year"].iloc[0]) if "season_year" in df.columns and pd.notna(df["season_year"].iloc[0]) else None),
            "season_type": season_type or (str(df["season_type"].iloc[0]) if "season_type" in df.columns and pd.notna(df["season_type"].iloc[0]) else None),
            "total_rows": len(df),
            "completeness_score": round(s_comp, 4),
            "timeliness_score": round(s_time, 4),
            "consistency_score": round(s_cons, 4),
            "weights_used": {
                "completeness": self.config.weight_completeness,
                "timeliness": self.config.weight_timeliness,
                "consistency": self.config.weight_consistency,
            },
            "component_metadata": {
                "completeness": meta_comp,
                "timeliness": meta_time,
                "consistency": meta_cons,
            },
            "source_metadata": source_metadata or {},
            "audit_notes": " | ".join(notes),
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"DRI evaluation complete for {crop} ({district}): DRI={calculated_dri} [{tier}] (Validation={validation_status})"
        )
        return audit_record

    def save_audit_ledger_record(
        self,
        audit_record: dict[str, Any],
        output_path: Optional[Path | str] = None,
    ) -> Path:
        """Safely append one complete audit record line to the JSON Lines audit ledger.

        Preserves existing entries, creates parent directories when needed,
        and provides transparent error logging. Note that this provides local-file
        JSONL persistence rather than a live relational database connection.

        Args:
            audit_record: Dictionary containing DRI evaluation and audit fields.
            output_path: Optional destination file path. If None, defaults to
                `data/processed/prices/dri_audit_ledger.jsonl`.

        Returns:
            Path to the written ledger file.

        Raises:
            IsADirectoryError: If output_path is an existing directory.
            OSError: If file creation or append operation fails.
            TypeError: If audit_record cannot be serialized to JSON.
        """
        if output_path is None:
            target_path = PROCESSED_PRICE_DIR / "dri_audit_ledger.jsonl"
        else:
            target_path = Path(output_path)

        try:
            if target_path.is_dir():
                raise IsADirectoryError(f"Target audit ledger path is a directory: {target_path}")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(audit_record, ensure_ascii=False, default=str)
            with open(target_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            logger.info(f"Appended DRI audit record to ledger at {target_path}")
            return target_path
        except (OSError, TypeError, ValueError) as exc:
            logger.error(f"Failed to append DRI audit record to ledger at {target_path}: {exc}")
            raise


def save_audit_ledger_record(
    audit_record: dict[str, Any],
    output_path: Optional[Path | str] = None,
) -> Path:
    """Module-level convenience wrapper for DRICalculator.save_audit_ledger_record."""
    return DRICalculator().save_audit_ledger_record(audit_record, output_path=output_path)