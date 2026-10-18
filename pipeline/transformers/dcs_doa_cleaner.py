"""
Data Cleaner and Normalizer for DCS and DOA Agricultural Statistics.
SLIIT Research Project DS-322 (2026/27) - Member 1

Normalizes crop names, districts, units, dates, and handles missing/estimated values explicitly.
"""

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from pipeline.extractors.dcs_extractor import DCSExtractionRecord
from pipeline.extractors.doa_extractor import DOACultivationRecord


# Standard 25 administrative districts of Sri Lanka plus aggregates
OFFICIAL_DISTRICTS = {
    "ampara": "Ampara",
    "anuradhapura": "Anuradhapura",
    "badulla": "Badulla",
    "batticaloa": "Batticaloa",
    "colombo": "Colombo",
    "galle": "Galle",
    "gampaha": "Gampaha",
    "hambantota": "Hambantota",
    "jaffna": "Jaffna",
    "kalutara": "Kalutara",
    "kandy": "Kandy",
    "kegalle": "Kegalle",
    "kilinochchi": "Kilinochchi",
    "kurunegala": "Kurunegala",
    "mannar": "Mannar",
    "matale": "Matale",
    "matara": "Matara",
    "monaragala": "Monaragala",
    "mullaitivu": "Mullaitivu",
    "nuwara eliya": "Nuwara Eliya",
    "nuwaraeliya": "Nuwara Eliya",
    "polonnaruwa": "Polonnaruwa",
    "puttalam": "Puttalam",
    "ratnapura": "Ratnapura",
    "trincomalee": "Trincomalee",
    "vavuniya": "Vavuniya",
    "national total": "National Total",
    "other districts": "Other Districts",
    "other district": "Other Districts",
}

# Standard crop vocabulary mapping
CROP_VOCABULARY = {
    "potato": "Potato",
    "solanum tuberosum": "Potato",
    "big onion": "Big Onion",
    "red onion": "Red Onion",
    "chilli": "Green Chilli",
    "green chilli": "Green Chilli",
    "green chili": "Green Chilli",
    "tomato": "Tomato",
    "carrot": "Carrot",
    "cabbage": "Cabbage",
    "beans": "Beans",
    "bush beans": "Beans",
    "pole beans": "Beans",
    "brinjal": "Brinjal",
    "eggplant": "Brinjal",
    "pumpkin": "Pumpkin",
    "cassava": "Cassava",
    "manioc": "Cassava",
    "sugarcane": "Sugarcane",
}


class DCSDOACleaner:
    """Cleans, normalizes, and enriches DCS and DOA records."""

    def clean_dcs_record(self, raw_record: Union[DCSExtractionRecord, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Transforms a raw DCSExtractionRecord into a cleaned, validated canonical dictionary.
        """
        if isinstance(raw_record, DCSExtractionRecord):
            rec = raw_record.to_dict()
        else:
            rec = dict(raw_record)

        # 1. Normalize Crop
        cleaned_crop = self.normalize_crop(rec.get("crop", ""))

        # 2. Normalize District
        cleaned_district = self.normalize_district(rec.get("district", ""))

        # 3. Normalize Metric & Unit
        cleaned_metric = self.normalize_metric(rec.get("metric", ""))
        cleaned_unit = self.normalize_unit(rec.get("raw_unit", ""))

        # 4. Parse Value & Handle Missing / Estimated
        raw_val = str(rec.get("raw_value", "")).strip()
        parsed_val, is_estimated, raw_status = self.parse_value(raw_val)

        # 5. Parse Date / Period
        period_label = str(rec.get("period_label", "")).strip()
        period_info = self.parse_time_period(period_label)

        # 6. Generate Provenance Signature
        prov_hash = self._generate_provenance_hash(
            source="DCS",
            crop=cleaned_crop,
            district=cleaned_district,
            period_label=period_label,
            metric=cleaned_metric,
            raw_val=raw_val,
            source_url=rec.get("source_url", ""),
        )

        return {
            "source_name": "DCS",
            "crop": cleaned_crop,
            "district": cleaned_district,
            "metric": cleaned_metric,
            "unit": cleaned_unit,
            "value": parsed_val,
            "is_estimated": is_estimated,
            "raw_status": raw_status,
            "raw_value": raw_val,
            "period_type": period_info["period_type"],
            "period_label": period_label,
            "period_start": period_info["period_start"],
            "period_end": period_info["period_end"],
            "year": period_info["year"],
            "season": period_info["season"],
            "table_id": rec.get("table_id", ""),
            "table_title": rec.get("table_title", ""),
            "source_url": rec.get("source_url", ""),
            "extraction_timestamp": rec.get("extraction_timestamp", ""),
            "provenance_hash": prov_hash,
            "footnote": rec.get("footnote"),
        }

    def clean_doa_cultivation_record(self, raw_record: Union[DOACultivationRecord, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Transforms a raw DOACultivationRecord into a cleaned, canonical dictionary.
        """
        if isinstance(raw_record, DOACultivationRecord):
            rec = raw_record.to_dict()
        else:
            rec = dict(raw_record)

        cleaned_crop = self.normalize_crop(rec.get("crop", ""))
        cleaned_district = self.normalize_district(rec.get("district", ""))
        season = self.normalize_season(rec.get("season", ""))
        year = int(rec.get("year", 0))

        # Time period based on season and year
        period_info = self._get_season_period_info(season, year)

        target_ha = rec.get("target_extent_ha")
        cultivated_ha = rec.get("cultivated_extent_ha")
        production_mt = rec.get("expected_production_mt")

        achievement_pct = None
        if target_ha is not None and target_ha > 0 and cultivated_ha is not None:
            achievement_pct = round((cultivated_ha / target_ha) * 100.0, 2)

        prov_hash = self._generate_provenance_hash(
            source="DOA_SEPC",
            crop=cleaned_crop,
            district=cleaned_district,
            period_label=f"{season}_{year}",
            metric="cultivation_forecast",
            raw_val=str(cultivated_ha),
            source_url=rec.get("raw_source", ""),
        )

        return {
            "source_name": "DOA_SEPC",
            "crop": cleaned_crop,
            "district": cleaned_district,
            "season": season,
            "year": year,
            "target_extent_ha": target_ha,
            "cultivated_extent_ha": cultivated_ha,
            "expected_production_mt": production_mt,
            "achievement_pct": achievement_pct,
            "period_type": "season",
            "period_label": f"{season} {year}" if season else str(year),
            "period_start": period_info["period_start"],
            "period_end": period_info["period_end"],
            "raw_source": rec.get("raw_source", ""),
            "extraction_timestamp": rec.get("extraction_timestamp", ""),
            "provenance_hash": prov_hash,
        }

    def normalize_crop(self, crop_raw: str) -> str:
        """Normalizes crop names to canonical English vegetable names."""
        if not crop_raw:
            return "Unknown"
        text = crop_raw.strip().lower()
        # Direct match or alias lookup
        for key, canonical in CROP_VOCABULARY.items():
            if key in text:
                return canonical
        # Fallback to title-cased trimmed string
        return crop_raw.strip().title()

    def normalize_district(self, district_raw: str) -> str:
        """Normalizes Sri Lankan district names, handling typos and casing."""
        if not district_raw:
            return "Unknown"
        cleaned = re.sub(r"[*\d]", "", district_raw).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).lower()
        return OFFICIAL_DISTRICTS.get(cleaned, district_raw.strip().title())

    def normalize_metric(self, metric_raw: str) -> str:
        """Standardizes metric classification."""
        if not metric_raw:
            return "unknown"
        m_lower = metric_raw.lower()
        if "production" in m_lower:
            return "production"
        if "extent" in m_lower:
            return "extent"
        if "yield" in m_lower:
            return "yield"
        if "farmer" in m_lower:
            return "farmers"
        if "import" in m_lower:
            return "imports"
        return metric_raw.strip().lower()

    def normalize_unit(self, unit_raw: str) -> str:
        """Normalizes unit to standard unit identifiers."""
        if not unit_raw:
            return "unknown"
        u_lower = unit_raw.lower()
        if "kg/ha" in u_lower or "kg per ha" in u_lower:
            return "kg_per_hectare"
        if any(term in u_lower for term in ("metric ton", "mt")):
            return "metric_ton"
        if any(term in u_lower for term in ("hectare", "ha")):
            return "hectare"
        if any(term in u_lower for term in ("count", "farmer", "number")):
            return "count"
        return "unknown"

    def normalize_season(self, season_raw: str) -> str:
        """Normalizes agricultural season."""
        if not season_raw:
            return "Unknown"
        s = season_raw.strip().lower()
        if "yala" in s:
            return "Yala"
        if "maha" in s:
            return "Maha"
        return season_raw.strip().title()

    def parse_value(self, val_str: str) -> Tuple[Optional[float], bool, str]:
        """
        Parses numeric value string while tracking estimated flags and missing statuses.

        Returns:
            Tuple of (parsed_float_or_none, is_estimated_bool, raw_status_str)
            raw_status can be: 'valid', 'nil', 'missing', 'unparseable'
        """
        if not val_str:
            return None, False, "missing"

        # Explicit nil / zero markers in government statistics
        cleaned = val_str.strip()
        if cleaned in ("-", "nil", "Nil", "NIL", "—", "–"):
            return 0.0, False, "nil"

        # Explicit missing markers
        if cleaned.upper() in ("N/A", "NA", "NOT AVAILABLE", "NULL", "NONE", ""):
            return None, False, "missing"

        # Detect estimated footnote asterisk
        is_estimated = "*" in cleaned
        cleaned = cleaned.replace("*", "").replace(",", "").strip()

        try:
            parsed = float(cleaned)
            return parsed, is_estimated, "valid"
        except ValueError:
            return None, is_estimated, "unparseable"

    def parse_time_period(self, label: str) -> Dict[str, Any]:
        """
        Parses time period label into standardized start/end dates and period types.

        Examples:
            '2023' -> annual
            '2018 upto May' -> partial_year
            '2006/2007' -> crop_year
            'Nov/Dec' -> bimonthly
        """
        cleaned = label.strip()

        # Format 1: 4-digit Year (e.g. '2023')
        year_match = re.match(r"^(\d{4})$", cleaned)
        if year_match:
            y = int(year_match.group(1))
            return {
                "period_type": "annual",
                "year": y,
                "season": None,
                "period_start": f"{y}-01-01",
                "period_end": f"{y}-12-31",
            }

        # Format 2: Partial year (e.g. '2018 upto May')
        partial_match = re.search(r"^(\d{4})\s+upto\s+([A-Za-z]+)", cleaned, re.IGNORECASE)
        if partial_match:
            y = int(partial_match.group(1))
            month_name = partial_match.group(2).lower()
            month_end = {
                "jan": "01-31", "feb": "02-28", "mar": "03-31", "apr": "04-30",
                "may": "05-31", "jun": "06-30", "jul": "07-31", "aug": "08-31",
                "sep": "09-30", "oct": "10-31", "nov": "11-30", "dec": "12-31"
            }
            end_md = month_end.get(month_name[:3], "05-31")
            return {
                "period_type": "partial_year",
                "year": y,
                "season": None,
                "period_start": f"{y}-01-01",
                "period_end": f"{y}-{end_md}",
            }

        # Format 3: Crop Split Year (e.g. '2006/2007')
        split_match = re.match(r"^(\d{4})/(\d{4})$", cleaned)
        if split_match:
            y_start = int(split_match.group(1))
            y_end = int(split_match.group(2))
            return {
                "period_type": "crop_year",
                "year": y_end,
                "season": None,
                "period_start": f"{y_start}-10-01",
                "period_end": f"{y_end}-09-30",
            }

        # Format 4: Season with Year (e.g. 'Yala 2024' or '2024 Yala')
        season_match = re.search(r"(Yala|Maha)\s*(\d{4})|(\d{4})\s*(Yala|Maha)", cleaned, re.IGNORECASE)
        if season_match:
            s_name = (season_match.group(1) or season_match.group(4)).title()
            y_str = season_match.group(2) or season_match.group(3)
            y = int(y_str)
            return self._get_season_period_info(s_name, y)

        # Default fallback
        year_any = re.search(r"\b(20\d{2}|19\d{2})\b", cleaned)
        y = int(year_any.group(1)) if year_any else None
        start_dt = f"{y}-01-01" if y else None
        end_dt = f"{y}-12-31" if y else None

        return {
            "period_type": "other",
            "year": y,
            "season": None,
            "period_start": start_dt,
            "period_end": end_dt,
        }

    def _get_season_period_info(self, season: str, year: int) -> Dict[str, Any]:
        """Provides calendar boundaries for Sri Lankan agricultural seasons."""
        if season == "Yala":
            # Yala season typically runs May to August/September
            start_dt = f"{year}-05-01" if year else None
            end_dt = f"{year}-08-31" if year else None
        elif season == "Maha":
            # Maha season typically runs September of prior year to March of current year
            prev_year = (year - 1) if year else None
            start_dt = f"{prev_year}-09-01" if prev_year else None
            end_dt = f"{year}-03-31" if year else None
        else:
            start_dt = f"{year}-01-01" if year else None
            end_dt = f"{year}-12-31" if year else None

        return {
            "period_type": "season",
            "year": year,
            "season": season,
            "period_start": start_dt,
            "period_end": end_dt,
        }

    def _generate_provenance_hash(self, **kwargs) -> str:
        """Generates a reproducible SHA-256 fingerprint for data provenance."""
        payload = "|".join(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
