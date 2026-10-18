"""
Data Validation and Quality Assurance for DCS and DOA Statistics.
SLIIT Research Project DS-322 (2026/27) - Member 1

Validates completeness, bounds, district naming, and provenance integrity.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from pipeline.transformers.dcs_doa_cleaner import OFFICIAL_DISTRICTS


VALID_METRICS: Set[str] = {
    "production",
    "extent",
    "yield",
    "farmers",
    "imports",
    "cultivation_forecast",
}

VALID_UNITS: Set[str] = {
    "metric_ton",
    "hectare",
    "kg_per_hectare",
    "count",
    "unknown",
}


class DCSDOAValidator:
    """Validates cleaned DCS and DOA agricultural statistics records."""

    def __init__(self, strict_districts: bool = True):
        self.strict_districts = strict_districts
        self.known_districts: Set[str] = set(OFFICIAL_DISTRICTS.values())

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates a single cleaned record against schema and business rules.

        Returns:
            Tuple of (is_valid: bool, issues: List[str])
        """
        issues: List[str] = []

        # 1. Mandatory Schema Keys
        source = record.get("source_name")
        if not source:
            issues.append("Missing 'source_name'")

        crop = record.get("crop")
        if not crop or crop == "Unknown":
            issues.append("Missing or unknown 'crop'")

        district = record.get("district")
        if not district or district == "Unknown":
            issues.append("Missing or unknown 'district'")
        elif self.strict_districts and district not in self.known_districts:
            issues.append(f"Unrecognized Sri Lankan district: '{district}'")

        # 2. Source-specific field validations
        if source == "DCS":
            metric = record.get("metric")
            if not metric or metric not in VALID_METRICS:
                issues.append(f"Invalid or missing DCS metric: '{metric}'")

            unit = record.get("unit")
            if not unit or unit not in VALID_UNITS:
                issues.append(f"Invalid or missing DCS unit: '{unit}'")

            val = record.get("value")
            raw_status = record.get("raw_status")
            if raw_status == "valid":
                if val is None:
                    issues.append("Status is 'valid' but value is None")
                elif val < 0:
                    issues.append(f"Negative value not allowed: {val}")
            elif raw_status not in ("nil", "missing", "unparseable"):
                issues.append(f"Unknown raw_status: '{raw_status}'")

        elif source == "DOA_SEPC":
            season = record.get("season")
            if not season or season not in ("Yala", "Maha", "Unknown"):
                issues.append(f"Invalid agricultural season: '{season}'")

            year = record.get("year")
            if not year or year < 1990 or year > 2050:
                issues.append(f"Year out of expected range: {year}")

            for field in ("target_extent_ha", "cultivated_extent_ha", "expected_production_mt"):
                f_val = record.get(field)
                if f_val is not None and f_val < 0:
                    issues.append(f"Negative value not allowed for {field}: {f_val}")

        # 3. Time bounds validation
        p_start = record.get("period_start")
        p_end = record.get("period_end")
        if p_start and p_end and p_start > p_end:
            issues.append(f"Inverted period: period_start '{p_start}' is after period_end '{p_end}'")

        # 4. Provenance checks
        prov_hash = record.get("provenance_hash")
        if not prov_hash:
            issues.append("Missing 'provenance_hash' for audit trail")

        is_valid = len(issues) == 0
        return is_valid, issues

    def validate_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates a collection of records and compiles an audit summary.
        """
        total = len(records)
        valid_count = 0
        invalid_count = 0
        all_issues: List[Dict[str, Any]] = []

        for idx, rec in enumerate(records):
            is_valid, issues = self.validate_record(rec)
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                all_issues.append({
                    "record_index": idx,
                    "crop": rec.get("crop"),
                    "district": rec.get("district"),
                    "period_label": rec.get("period_label"),
                    "issues": issues,
                })

        return {
            "total_records": total,
            "valid_records": valid_count,
            "invalid_records": invalid_count,
            "pass_rate_pct": round((valid_count / total * 100.0), 2) if total > 0 else 100.0,
            "issues": all_issues,
        }
