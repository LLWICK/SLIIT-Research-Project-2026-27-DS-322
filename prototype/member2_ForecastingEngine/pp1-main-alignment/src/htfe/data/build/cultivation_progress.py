"""Monthly cultivation progress: achieved hectares / target hectares.

The methodology requires a real district-month table. This module never invents
hectares. If that table is absent, the panel column is left missing and the
gap is written to the audit folder.

The ratio is raw achieved/target. It is not clipped at 1. A later month is
never interpolated into an earlier week: merge_asof keeps only a report whose
available date is on or before the forecast Monday.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from htfe.config import DATA_ROOT, processed_dir

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
DOA_FILENAME = "doa_cultivation_progress.csv"


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
    """Read cultivation_monthly.csv from one known path. Do not scan the data disk."""
    from htfe.config import cultivation_monthly_path

    return cultivation_monthly_path()


def doa_progress_path() -> Path | None:
    """Known Department of Agriculture extract. This is not a disk scan.

    Member 1's folder is checked first, then the processed data folder.
    The file keeps the name ``doa_cultivation_progress.csv``. It is not
    renamed to ``cultivation_monthly.csv`` and it is not a loader source.
    """
    from htfe.config import MEMBER1_DIR, PROCESSED_DIR

    for path in (MEMBER1_DIR / DOA_FILENAME, PROCESSED_DIR / DOA_FILENAME):
        if path.is_file():
            return path
    return None


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_excel(path)


def load_monthly(path: Path | None = None) -> pd.DataFrame | None:
    found = path or find_monthly_table()
    if found is None:
        return None
    frame = _normalise_columns(_read_table(found))
    missing = [name for name in REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError(
            "This cultivation file does not match the district-month contract "
            f"(year, month, crop, district, target_ha, achieved_ha). Missing {missing}. "
            "The district-month loader will not use it. "
            "Experiment C reads national rows from doa_cultivation_progress.csv."
        )
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


def _as_int(value) -> int:
    return int(value)


def _scope_mask(frame: pd.DataFrame, scope: str) -> pd.Series:
    if "scope_level" not in frame.columns:
        if scope == "national":
            return frame["district"].astype(str).str.strip().str.casefold().eq("national")
        return ~frame["district"].astype(str).str.strip().str.casefold().eq("national")
    return frame["scope_level"].astype(str).str.strip().str.casefold().eq(scope)


def inspect_cultivation_sources() -> dict:
    """Describe the known cultivation files without scanning the disk.

    ``doa_cultivation_progress.csv`` keeps its own name. It is not passed to
    the district-month loader. Whether the national Experiment C run has been
    fit is recorded in ``outputs/experiment_c/status.json``.
    """
    monthly = find_monthly_table()
    doa = doa_progress_path()
    status: dict = {
        "monthly_contract_path": None if monthly is None else str(monthly),
        "monthly_contract_present": monthly is not None,
        "doa_path": None if doa is None else str(doa),
        "doa_present": doa is not None,
        "wired_as_loader_source": False,
        "experiment_c_trained": False,
        "matches_loader_contract": False,
        "columns": [],
        "n_rows": 0,
        "counts": [],
        "sentences": [],
        "headline": "National cultivation file not confirmed.",
    }
    if monthly is not None:
        monthly_frame = _normalise_columns(_read_table(monthly))
        missing = [name for name in REQUIRED if name not in monthly_frame.columns]
        status["matches_loader_contract"] = not missing
        status["wired_as_loader_source"] = not missing
        status["monthly_rows"] = _as_int(len(monthly_frame))
        status["monthly_columns"] = [str(column) for column in monthly_frame.columns]
        if missing:
            status["sentences"].append(
                f"cultivation_monthly.csv is at {monthly}, but it is missing {missing}, so the district-month loader will not use it."
            )
        else:
            status["sentences"].append(
                f"cultivation_monthly.csv is at {monthly} and its columns match the loader. "
                "Experiment C is fit from national rows in doa_cultivation_progress.csv, not from this district-month file."
            )
    if doa is None:
        if not status["sentences"]:
            status["sentences"] = [
                "processed/doa_cultivation_progress.csv was not found at the known path.",
                "data/member1/cultivation_monthly.csv and processed/cultivation_monthly.csv are absent.",
                "Experiment C reads the national rows of doa_cultivation_progress.csv when that file is present.",
            ]
        status["headline"] = "processed/doa_cultivation_progress.csv was not found at the known path."
        return status

    frame = pd.read_csv(doa)
    normalised = _normalise_columns(frame)
    columns = [str(column) for column in frame.columns]
    missing = [name for name in REQUIRED if name not in normalised.columns]
    national = frame.loc[_scope_mask(frame, "national")]
    district = frame.loc[_scope_mask(frame, "district")]
    district_dates = sorted(district["observation_date"].dropna().astype(str).unique()) if "observation_date" in district.columns else []
    district_available = sorted(district["available_date"].dropna().astype(str).unique()) if "available_date" in district.columns else []
    district_crops = sorted(district["crop"].dropna().astype(str).str.strip().str.lower().unique()) if len(district) else []
    district_seasons = (
        sorted(district["season"].dropna().astype(str).str.strip().unique())
        if len(district) and "season" in district.columns
        else []
    )
    tomato_district = 0
    if len(district) and "crop" in district.columns:
        tomato_district = _as_int(district["crop"].astype(str).str.strip().str.lower().eq("tomato").sum())
    zero_target = 0
    zero_target_district = 0
    positive_district = 0
    if "target_ha" in frame.columns:
        target = pd.to_numeric(frame["target_ha"], errors="coerce")
        zero_target = _as_int(target.fillna(0).eq(0).sum())
        if len(district):
            district_target = pd.to_numeric(district["target_ha"], errors="coerce")
            zero_target_district = _as_int(district_target.fillna(0).eq(0).sum())
            positive_district = _as_int(district_target.gt(0).sum())
    contract_match = not missing
    # A single district snapshot is not a monthly series the as-of join can train on.
    district_month_series = len(district_dates) > 1
    status.update(
        {
            "columns": columns,
            "n_rows": _as_int(len(frame)),
            "n_national": _as_int(len(national)),
            "n_district": _as_int(len(district)),
            "district_observation_dates": district_dates,
            "district_available_dates": district_available,
            "district_crops": district_crops,
            "district_seasons": district_seasons,
            "tomato_district_rows": tomato_district,
            "zero_target_rows": zero_target,
            "zero_target_district_rows": zero_target_district,
            "positive_target_district_rows": positive_district,
            "missing_contract_columns": missing,
            "doa_matches_loader_contract": contract_match,
            "doa_is_district_month_series": bool(contract_match and district_month_series and len(district) > 0),
            "experiment_c_trained": None,
        }
    )
    # The DOA extract keeps its own filename. It is not renamed to cultivation_monthly.csv.
    if len(district_dates) == 0:
        date_sentence = "No district observation dates were found."
    elif len(district_dates) == 1:
        published = f" and published on {district_available[0]}" if len(district_available) == 1 else ""
        crops = ", ".join(district_crops) if district_crops else "no listed crop"
        date_sentence = f"Every district row is one snapshot, observed on {district_dates[0]}{published}, for {crops}."
    else:
        date_sentence = "District observation dates: " + ", ".join(district_dates) + "."
    sentences = [
        f"processed/{DOA_FILENAME} has {_as_int(len(frame))} rows.",
        "Columns: " + ", ".join(columns) + ".",
        "target_ha and achieved_ha are in the file, so these are target and achieved hectares rather than a renamed unrelated table.",
        (
            "year and month are not columns. The dates in the file are available_date and observation_date."
            if missing
            else "year and month are present."
        ),
        f"{_as_int(len(national))} rows are national totals. {_as_int(len(district))} rows are district rows.",
        date_sentence,
        (
            "Tomato has no district rows."
            if tomato_district == 0
            else f"Tomato district rows: {tomato_district}."
        ),
        (
            f"{zero_target_district} district rows have a target of 0 hectares. "
            f"{positive_district} district rows have a target above 0."
        ),
        (
            "The loader contract is year, month, crop, district, target_ha, and achieved_ha, "
            "joined to each market's supply districts. This file was not copied to cultivation_monthly.csv."
        ),
        (
            "District rows stay in the audit. The scored cultivation signal is national, "
            "matches the official source season, and uses a report only when available_date "
            "is on or before the forecast Monday. Ratios above 1 are kept. A non-positive target is not turned into 1."
        ),
    ]
    status["sentences"] = sentences
    status["headline"] = (
        "The Department of Agriculture file has national totals and district rows. "
        "Scored cultivation progress uses the national rows only. "
        "This file is not renamed to cultivation_monthly.csv and is not joined as district supply. "
        "The dashboard forecast is the runtime model. It is not Experiment C."
    )
    status["counts"] = [
        {"Item": "Rows", "Value": str(status["n_rows"])},
        {"Item": "National rows", "Value": str(status["n_national"])},
        {"Item": "District rows", "Value": str(status["n_district"])},
        {"Item": "District observation dates", "Value": ", ".join(district_dates) if district_dates else "none"},
        {"Item": "District crops", "Value": ", ".join(district_crops) if district_crops else "none"},
        {"Item": "Tomato district rows", "Value": str(tomato_district)},
        {"Item": "District rows with target 0", "Value": str(zero_target_district)},
        {"Item": "District rows with target above 0", "Value": str(positive_district)},
        {"Item": "Missing loader columns", "Value": ", ".join(missing) if missing else "none"},
        {"Item": "District-month loader", "Value": "not used for this file"},
    ]
    return status


def write_gap_report(found: Path | None) -> Path:
    from htfe.config import OUTPUTS

    audit = OUTPUTS / "notes"
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
                "Looked for data/member1/cultivation_monthly.csv, then processed/cultivation_monthly.csv. The data disk is not scanned.",
                "That district-month contract file is not on disk.",
                "",
                "What else was checked:",
                "- `processed/dcs_highland_seasonal.csv` is Yala/Maha extent and production, not monthly target and achieved.",
                "- `processed/doa_cultivation_progress.csv` is read only when it is at that known path. It is not renamed into cultivation_monthly.csv.",
                "",
                *inspect_cultivation_sources()["sentences"],
                "",
                "No hectares were invented. The district-month column on the main training panel stays missing until a file matches that contract.",
                "National exact-season scoring is a separate path and is written under outputs/experiment_c when that run has been executed.",
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


def _blank_progress(out: pd.DataFrame) -> pd.DataFrame:
    out["cultivation_progress"] = pd.NA
    out["cultivation_target_ha"] = pd.NA
    out["cultivation_achieved_ha"] = pd.NA
    out["cultivation_asof_date"] = pd.NaT
    out["cultivation_target_exceeded"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["cultivation_progress_source"] = SOURCE_MISSING
    return out


def attach_progress(panel: pd.DataFrame, monthly: pd.DataFrame | None, origin_map: dict) -> pd.DataFrame:
    """As-of join. Sum achieved and sum target across supply districts, then divide.

    Progress is left above 1 when achieved hectares exceed the target. Future
    months are not interpolated.
    """
    out = panel.copy()
    out["week_start"] = pd.to_datetime(out["week_start"])
    if monthly is None or monthly.empty:
        return _blank_progress(out)

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
        return _blank_progress(out)
    long = pd.concat(pieces, ignore_index=True)
    grouped = long.groupby(["crop", "market", "week_start"], as_index=False).agg(
        cultivation_target_ha=("target_ha", lambda series: series.sum(min_count=1)),
        cultivation_achieved_ha=("achieved_ha", lambda series: series.sum(min_count=1)),
        cultivation_asof_date=("available_date", "max"),
    )
    grouped["cultivation_progress"] = grouped["cultivation_achieved_ha"] / grouped["cultivation_target_ha"].replace(0, pd.NA)
    known = grouped["cultivation_progress"].notna()
    grouped["cultivation_target_exceeded"] = pd.Series(pd.NA, index=grouped.index, dtype="boolean")
    grouped.loc[known, "cultivation_target_exceeded"] = grouped.loc[known, "cultivation_progress"].astype(float).gt(1)
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
            {"year": 2024, "month": 6, "crop": "carrot", "district": "Nuwara Eliya", "target_ha": 100, "achieved_ha": 140, "available_date": "2024-06-30"},
            {"year": 2024, "month": 6, "crop": "carrot", "district": "Badulla", "target_ha": 50, "achieved_ha": 70, "available_date": "2024-06-30"},
        ]
    )
    monthly = _normalise_columns(monthly)
    monthly["crop_key"] = monthly["crop"].str.lower()
    monthly["district_key"] = monthly["district"].str.casefold()
    monthly["available_date"] = pd.to_datetime(monthly["available_date"])
    panel = pd.DataFrame(
        [
            {"crop": "carrot", "market": "Dambulla", "week_start": "2024-04-01"},
            {"crop": "carrot", "market": "Dambulla", "week_start": "2024-05-06"},
            {"crop": "carrot", "market": "Dambulla", "week_start": "2024-06-03"},
            {"crop": "carrot", "market": "Dambulla", "week_start": "2024-07-08"},
        ]
    )
    origin = {"carrot": {"Dambulla": {"supply_districts": ["Nuwara Eliya", "Badulla"]}}}
    checked = attach_progress(panel, monthly, origin).set_index("week_start")
    april = 65 / 150
    may = 120 / 150
    june = 210 / 150
    early = checked.loc[pd.Timestamp("2024-04-01"), "cultivation_progress"]
    if pd.notna(early):
        raise AssertionError(early)
    if abs(float(checked.loc[pd.Timestamp("2024-05-06"), "cultivation_progress"]) - april) > 1e-9:
        raise AssertionError(checked.loc[pd.Timestamp("2024-05-06"), "cultivation_progress"])
    if abs(float(checked.loc[pd.Timestamp("2024-06-03"), "cultivation_progress"]) - may) > 1e-9:
        raise AssertionError(checked.loc[pd.Timestamp("2024-06-03"), "cultivation_progress"])
    raw = float(checked.loc[pd.Timestamp("2024-07-08"), "cultivation_progress"])
    if abs(raw - june) > 1e-9 or raw <= 1:
        raise AssertionError(raw)
    if not bool(checked.loc[pd.Timestamp("2024-07-08"), "cultivation_target_exceeded"]):
        raise AssertionError("ratio above 1 was clipped or not flagged")
    if bool(checked.loc[pd.Timestamp("2024-05-06"), "cultivation_target_exceeded"]):
        raise AssertionError("April ratio was flagged as exceeded")
