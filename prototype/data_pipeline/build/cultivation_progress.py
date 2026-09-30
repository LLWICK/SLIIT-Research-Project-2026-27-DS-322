"""Monthly cultivation progress: achieved hectares / target hectares.

The methodology requires a real district-month table. This module never invents
hectares. If that table is absent, the panel column is left missing and the
gap is written to the audit folder.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.config import DATA_ROOT, processed_dir

REQUIRED = ("year", "month", "crop", "district", "target_ha", "achieved_ha")
ALIASES = {
    "target hectares": "target_ha",
    "target_hectares": "target_ha",
    "target ha": "target_ha",
    "achieved hectares": "achieved_ha",
    "achieved_hectares": "achieved_ha",
    "achieved ha": "achieved_ha",
    "progress ha": "achieved_ha",
    "cultivated ha": "achieved_ha",
    "publication_date": "available_date",
    "as_of": "available_date",
    "asof_date": "available_date",
}
SOURCE_MISSING = "unavailable_no_district_monthly_series"
SOURCE_FOUND = "monthly_target_achieved"


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace("-", " ")
        renamed[column] = ALIASES.get(key, key.replace(" ", "_"))
    out = frame.rename(columns=renamed)
    if "available_date" not in out.columns and {"year", "month"}.issubset(out.columns):
        # A month is knowable only after it has finished, unless a publication date is given.
        period = pd.to_datetime(
            out["year"].astype(int).astype(str) + "-" + out["month"].astype(int).astype(str) + "-01",
            errors="coerce",
        )
        out["available_date"] = period + pd.offsets.MonthEnd(0)
    return out


def _has_required(frame: pd.DataFrame) -> bool:
    columns = set(_normalise_columns(frame).columns)
    return set(REQUIRED).issubset(columns)


def _peek(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, nrows=5)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path, nrows=5)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path).head(5)
    except Exception:
        return None
    return None


def find_monthly_table() -> Path | None:
    """Return a file under the data root whose header has the required columns."""
    if not DATA_ROOT.exists():
        return None
    for path in DATA_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".xlsx", ".xls", ".parquet"}:
            continue
        if path.stat().st_size > 80_000_000:
            continue
        peeked = _peek(path)
        if peeked is not None and _has_required(peeked):
            return path
    return None


def load_monthly(path: Path | None = None) -> pd.DataFrame | None:
    found = path or find_monthly_table()
    if found is None:
        return None
    if found.suffix.lower() == ".csv":
        frame = pd.read_csv(found)
    elif found.suffix.lower() == ".parquet":
        frame = pd.read_parquet(found)
    else:
        frame = pd.read_excel(found)
    frame = _normalise_columns(frame)
    frame = frame.dropna(subset=["crop", "district", "year", "month"])
    frame["year"] = frame["year"].astype(int)
    frame["month"] = frame["month"].astype(int)
    frame["target_ha"] = pd.to_numeric(frame["target_ha"], errors="coerce")
    frame["achieved_ha"] = pd.to_numeric(frame["achieved_ha"], errors="coerce")
    frame["available_date"] = pd.to_datetime(frame["available_date"])
    frame["crop_key"] = frame["crop"].astype(str).str.strip().str.lower()
    frame["district_key"] = frame["district"].astype(str).str.strip().str.casefold()
    frame = frame.dropna(subset=["available_date", "target_ha", "achieved_ha"])
    frame = frame.sort_values("available_date").drop_duplicates(
        ["crop_key", "district_key", "year", "month"], keep="last"
    )
    return frame.reset_index(drop=True)


def write_gap_report(found: Path | None) -> Path:
    audit = processed_dir() / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    path = audit / "cultivation_progress_gap.md"
    if found is not None:
        path.write_text(
            "\n".join(
                [
                    "# Cultivation progress source",
                    "",
                    f"A table with year, month, crop, district, target_ha, and achieved_ha was found at `{found}`.",
                    "Progress is summed achieved hectares divided by summed target hectares over the market's supply districts.",
                    "A week uses only rows whose available date is on or before that week. When the file has no publication date, the available date is the month end.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return path
    path.write_text(
        "\n".join(
            [
                "# Cultivation progress source — not found",
                "",
                "Methodology.docx section 3.9 requires district-month cultivation progress = achieved hectares / target hectares.",
                "Searched `D:\\SLIIT\\4th Year\\Reaserch\\data sets` for a csv, xlsx, or parquet whose columns include year, month, crop, district, target hectares, and achieved hectares.",
                "No such table is on disk.",
                "",
                "What is on disk instead:",
                "- `processed/dcs_highland_seasonal.csv` is Yala/Maha extent and production, not monthly target and achieved.",
                "- The Department of Agriculture SEPC downloads page lists Crop Forecast monthly reports from February 2022 onward as Google Drive files, not as a district panel.",
                "- The public November 2019 crop-forecast PDF is a national crop total (carrot, leeks, and tomato each have one target and one progress figure). It is not district-level, so it is not used as the feature.",
                "",
                "No hectares were invented. `cultivation_progress` is missing on every panel row.",
                "Configuration C and the monthly re-forecast are therefore not claimed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _supply_links(origin_map: dict) -> pd.DataFrame:
    rows = []
    for crop, markets in origin_map.items():
        for market, spec in markets.items():
            for district in spec.get("supply_districts") or []:
                rows.append(
                    {
                        "crop": crop,
                        "market": market,
                        "district_key": str(district).strip().casefold(),
                    }
                )
    return pd.DataFrame(rows)


def attach_progress(panel: pd.DataFrame, monthly: pd.DataFrame | None, origin_map: dict) -> pd.DataFrame:
    """As-of join. Sum achieved and sum target across supply districts, then divide."""
    out = panel.copy()
    out["week_start"] = pd.to_datetime(out["week_start"])
    if monthly is None or monthly.empty:
        out["cultivation_progress"] = pd.NA
        out["cultivation_target_ha"] = pd.NA
        out["cultivation_achieved_ha"] = pd.NA
        out["cultivation_asof_date"] = pd.NaT
        out["cultivation_progress_source"] = SOURCE_MISSING
        return out

    links = _supply_links(origin_map)
    base = out[["crop", "market", "week_start"]].copy()
    base["crop_key"] = base["crop"].astype(str).str.strip().str.lower()
    expanded = base.merge(links, on=["crop", "market"], how="inner")
    right = monthly.sort_values(["crop_key", "available_date"])
    expanded = expanded.sort_values(["crop_key", "district_key", "week_start"])
    pieces = []
    for district, left in expanded.groupby("district_key", sort=False):
        right_d = right[right["district_key"] == district].sort_values("available_date")
        if right_d.empty or left.empty:
            continue
        merged = pd.merge_asof(
            left.sort_values("week_start"),
            right_d[["crop_key", "available_date", "target_ha", "achieved_ha"]],
            left_on="week_start",
            right_on="available_date",
            by="crop_key",
            direction="backward",
        )
        pieces.append(merged)
    if not pieces:
        out["cultivation_progress"] = pd.NA
        out["cultivation_target_ha"] = pd.NA
        out["cultivation_achieved_ha"] = pd.NA
        out["cultivation_asof_date"] = pd.NaT
        out["cultivation_progress_source"] = SOURCE_MISSING
        return out
    long = pd.concat(pieces, ignore_index=True)
    grouped = long.groupby(["crop", "market", "week_start"], as_index=False).agg(
        cultivation_target_ha=("target_ha", lambda series: series.sum(min_count=1)),
        cultivation_achieved_ha=("achieved_ha", lambda series: series.sum(min_count=1)),
        cultivation_asof_date=("available_date", "max"),
    )
    grouped["cultivation_progress"] = grouped["cultivation_achieved_ha"] / grouped["cultivation_target_ha"].replace(0, pd.NA)
    merged_panel = out.merge(grouped, on=["crop", "market", "week_start"], how="left")
    merged_panel["cultivation_progress_source"] = SOURCE_FOUND
    merged_panel.loc[merged_panel["cultivation_progress"].isna(), "cultivation_progress_source"] = SOURCE_MISSING
    return merged_panel


def assert_formula() -> None:
    """Check sum-then-divide and the as-of rule on a tiny in-memory table. Nothing is written to the panel."""
    monthly = pd.DataFrame(
        [
            {"year": 2024, "month": 4, "crop": "carrot", "district": "Nuwara Eliya", "target_ha": 100, "achieved_ha": 40, "available_date": "2024-04-30"},
            {"year": 2024, "month": 4, "crop": "carrot", "district": "Badulla", "target_ha": 50, "achieved_ha": 25, "available_date": "2024-04-30"},
            {"year": 2024, "month": 5, "crop": "carrot", "district": "Nuwara Eliya", "target_ha": 100, "achieved_ha": 80, "available_date": "2024-05-31"},
            {"year": 2024, "month": 5, "crop": "carrot", "district": "Badulla", "target_ha": 50, "achieved_ha": 40, "available_date": "2024-05-31"},
        ]
    )
    monthly = _normalise_columns(monthly)
    monthly["crop_key"] = monthly["crop"].str.lower()
    monthly["district_key"] = monthly["district"].str.casefold()
    monthly["available_date"] = pd.to_datetime(monthly["available_date"])
    panel = pd.DataFrame(
        [
            {"crop": "carrot", "market": "Dambulla", "week_start": "2024-05-06"},
            {"crop": "carrot", "market": "Dambulla", "week_start": "2024-06-03"},
        ]
    )
    origin = {"carrot": {"Dambulla": {"supply_districts": ["Nuwara Eliya", "Badulla"]}}}
    checked = attach_progress(panel, monthly, origin)
    april = 65 / 150
    may = 120 / 150
    if abs(float(checked.iloc[0]["cultivation_progress"]) - april) > 1e-9:
        raise AssertionError(checked.iloc[0]["cultivation_progress"])
    if abs(float(checked.iloc[1]["cultivation_progress"]) - may) > 1e-9:
        raise AssertionError(checked.iloc[1]["cultivation_progress"])
