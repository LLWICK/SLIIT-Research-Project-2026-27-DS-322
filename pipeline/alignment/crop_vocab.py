"""
Crop vocabulary enforcement, canonical normalization, Big Onion origin isolation,
and ambiguous commodity quarantine logic for the five research target crops.
"""

from typing import Optional, Tuple, Set, Dict, List
import json
import os
import pandas as pd


# The five official research target crops
CANONICAL_TARGET_CROPS: Set[str] = {
    "Carrot",
    "Cabbage",
    "Leek",
    "Big Onion",
    "Tomato",
}

# Alias dictionary mapping normalized lower-case strings to canonical names
CROP_ALIAS_MAP: Dict[str, str] = {
    # 1. Carrot
    "carrot": "Carrot",
    "carrots": "Carrot",
    
    # 2. Cabbage (including regional varieties)
    "cabbage": "Cabbage",
    "cabbage (n'eliya)": "Cabbage",
    "cabbage (kandy)": "Cabbage",
    "cabbage n'eliya": "Cabbage",
    "cabbage kandy": "Cabbage",
    "cabbage-n'eliya": "Cabbage",
    "cabbage-kandy": "Cabbage",
    
    # 3. Leek (enforce singular form matching prototype standard)
    "leek": "Leek",
    "leeks": "Leek",
    "leek (local)": "Leek",
    "leeks (local)": "Leek",
    
    # 4. Big Onion (Local)
    "big onion": "Big Onion",
    "big-onion": "Big Onion",
    "big onion (local)": "Big Onion",
    "big-onion local": "Big Onion",
    "big onion local": "Big Onion",
    "b'onion local": "Big Onion",
    "b-onion local": "Big Onion",
    "b'onion": "Big Onion",
    "b-onion": "Big Onion",
    
    # Big Onion (Imported)
    "big onion (imported)": "Big Onion",
    "big-onion (imported)": "Big Onion",
    "big onion (imp)": "Big Onion",
    "big-onion (imp)": "Big Onion",
    "big onion imported": "Big Onion",
    "b'onion imported": "Big Onion",
    "b-onion imported": "Big Onion",
    
    # 5. Tomato
    "tomato": "Tomato",
    "tomatoes": "Tomato",
}

# Explicitly excluded non-target crops (recognized commodities that are not in the 5 research crops)
EXPLICIT_NON_TARGET_CROPS: Set[str] = {
    "red onion", "red-onion", "red onion (vedalan)", "red onion (sindo)",
    "red-onion vedalan", "red-onion sindo", "shallot", "shallots",
    "beans", "green beans", "butter beans", "pole beans",
    "raddish", "radish", "white raddish",
    "beetroot", "beet", "beets",
    "brinjal", "eggplant", "aubergine",
    "pumpkin",
    "ladies fingers", "ladies finger", "okra", "bandakka",
    "bitter gourd", "snake gourd", "ridge gourd", "ash plantain",
    "knol khol", "knolkhol", "capsicum", "green chilli", "green chili",
    "potato", "potatoes", "sweet potato", "cassava", "manioc"
}

# Ambiguous commodity names that MUST NEVER be silently mapped to Big Onion
AMBIGUOUS_ONION_TERMS: Set[str] = {
    "onion",
    "onions",
    "local onion",
    "local onions",
    "dry onion",
    "dry onions",
    "common onion",
}


def standardize_crop(
    crop_name: str,
    source_commodity_name: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], bool, Optional[str]]:
    """
    Standardize a crop string to one of the 5 canonical research crops,
    preserve Big Onion origin variety (Local vs. Imported), and detect
    ambiguous onion records that require quarantine.

    Args:
        crop_name: The primary crop or commodity string.
        source_commodity_name: Optional secondary source commodity string.

    Returns:
        Tuple of (canonical_crop_name, origin_variety, is_quarantined, quarantine_reason)
        - canonical_crop_name: Canonical crop name if recognized in the 5 crops, else None.
        - origin_variety: "Local", "Imported", or None.
        - is_quarantined: True if record is ambiguous and must be quarantined.
        - quarantine_reason: Human-readable explanation if quarantined, else None.
    """
    if not crop_name or not isinstance(crop_name, str):
        return None, None, False, None

    normalized_primary = crop_name.strip().lower()
    normalized_source = source_commodity_name.strip().lower() if (source_commodity_name and isinstance(source_commodity_name, str)) else ""

    # Check for ambiguous onion terms
    if normalized_primary in AMBIGUOUS_ONION_TERMS:
        # Check if the secondary source commodity clarifies it
        if "big" in normalized_source:
            # Clarified by secondary name
            pass
        elif "red" in normalized_source or "shallot" in normalized_source:
            # Clearly red onion, cleanly exclude
            return None, None, False, None
        else:
            reason = (
                f"Ambiguous commodity '{crop_name}' (source: '{source_commodity_name}'). "
                f"Cannot determine whether Big Onion or Red Onion. Quarantined to prevent bias."
            )
            return None, None, True, reason

    # Check if primary or source is explicitly an excluded non-target crop
    if normalized_primary in EXPLICIT_NON_TARGET_CROPS:
        return None, None, False, None
    if normalized_source in EXPLICIT_NON_TARGET_CROPS:
        return None, None, False, None

    # Check canonical alias mapping
    canonical = CROP_ALIAS_MAP.get(normalized_primary)
    if not canonical and normalized_source:
        canonical = CROP_ALIAS_MAP.get(normalized_source)

    if not canonical:
        # Check if it contains an ambiguous onion mention
        if "onion" in normalized_primary and "big" not in normalized_primary and "red" not in normalized_primary:
            reason = f"Unresolved onion variation '{crop_name}'. Quarantined without mapping."
            return None, None, True, reason
        # Unrecognized / non-target crop
        return None, None, False, None

    # Determine origin_variety for Big Onion
    origin_variety = None
    if canonical == "Big Onion":
        combined_text = f"{normalized_primary} {normalized_source}"
        if any(term in combined_text for term in ["imported", "imp", "b'onion imported"]):
            origin_variety = "Imported"
        elif any(term in combined_text for term in ["local", "big-onion local", "sri lanka"]):
            origin_variety = "Local"
        else:
            # If not explicitly marked, default to Local for domestic price/production series
            origin_variety = "Local"

    return canonical, origin_variety, False, None


class CropVocabularyEnforcer:
    """
    Filters and normalizes tabular datasets to the five research crops,
    tracks observation counts, isolates Big Onion varieties, and routes
    ambiguous commodities to an audit quarantine ledger.
    """

    def __init__(self, quarantine_sink_path: Optional[str] = None):
        """
        Initialize the enforcer.

        Args:
            quarantine_sink_path: Optional path to a JSONL file where quarantined records are appended.
        """
        self.quarantine_sink_path = quarantine_sink_path
        self.quarantined_records: List[Dict] = []

    def filter_and_standardize(
        self,
        df: pd.DataFrame,
        crop_col: str = "crop_name",
        commodity_col: Optional[str] = "source_commodity_name",
    ) -> pd.DataFrame:
        """
        Filter a DataFrame to the 5 target crops, normalize crop names,
        populate 'origin_variety', and record ambiguous items to the quarantine ledger.

        Args:
            df: Input DataFrame.
            crop_col: Name of the crop column.
            commodity_col: Optional name of the raw commodity column.

        Returns:
            Filtered and standardized pandas DataFrame containing only the 5 target crops.
        """
        if df.empty or crop_col not in df.columns:
            return df.copy()

        has_commodity = commodity_col is not None and commodity_col in df.columns

        standardized_crops = []
        origin_varieties = []
        keep_mask = []

        for idx, row in df.iterrows():
            c_val = str(row[crop_col]) if pd.notna(row[crop_col]) else ""
            cmd_val = str(row[commodity_col]) if has_commodity and pd.notna(row[commodity_col]) else None

            canonical, origin, is_quarantined, reason = standardize_crop(c_val, cmd_val)

            if is_quarantined:
                quarantine_entry = {
                    "row_index": idx,
                    "crop_value": c_val,
                    "commodity_value": cmd_val,
                    "reason": reason,
                    "row_data": {k: (v if pd.notna(v) else None) for k, v in row.to_dict().items()},
                }
                self.quarantined_records.append(quarantine_entry)
                self._persist_quarantine_record(quarantine_entry)
                keep_mask.append(False)
            elif canonical is not None:
                standardized_crops.append(canonical)
                origin_varieties.append(origin)
                keep_mask.append(True)
            else:
                keep_mask.append(False)

        filtered_df = df[keep_mask].copy()
        filtered_df[crop_col] = standardized_crops
        filtered_df["origin_variety"] = origin_varieties

        return filtered_df

    def _persist_quarantine_record(self, record: Dict) -> None:
        """Append a single quarantine record to the configured JSONL file if path is set."""
        if not self.quarantine_sink_path:
            return

        os.makedirs(os.path.dirname(os.path.abspath(self.quarantine_sink_path)), exist_ok=True)
        with open(self.quarantine_sink_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
