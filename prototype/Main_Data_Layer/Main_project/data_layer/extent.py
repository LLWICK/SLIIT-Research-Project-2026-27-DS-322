"""Load checked DOA cultivation-extent snapshots.

A dash in a report is stored as missing, not zero. A crop is included
only when its area rows add up to the Sri Lanka row and to each printed
subtotal.

July 2026, page 11: Carrot and Leeks. Tomato, on page 12, is left out.
Its achievement area rows total 1,748 ha and the Sri Lanka row is 1,744 ha.

October 2025 Yala Final, page 9: Leeks. Carrot is 1 ha short of the Sri
Lanka row on both target and achievement. Tomato, on page 10, is left out
because its area rows do not equal the Sri Lanka row.
"""

import csv

import duckdb
import pandas as pd

from data_layer.paths import DUCKDB_PATH, EXPORT_DIR, PACKAGE_DIR, WAREHOUSE_DIR

SNAPSHOT_CSV = PACKAGE_DIR / "extent_snapshots.csv"
CHECKED_CROPS = {
    ("2026-07-15", "carrot"),
    ("2026-07-15", "leeks"),
    ("2025-10-15", "leeks"),
}

JULY_2026 = {
    "report_date": "2026-07-15",
    "as_of_date": "2026-06-30",
    "season_label": "2026 Yala",
    "source_file": "yala-july-2026.pdf",
    "page_number": 11,
}
OCTOBER_2025_FINAL = {
    "report_date": "2025-10-15",
    "as_of_date": "2025-09-30",
    "season_label": "2025 Yala",
    "source_file": "yala-october-2025-yala-final.pdf",
    "page_number": 9,
}

# Names are the spellings printed in the table. Subtotal rows are kept for
# checking and are not part of the area total.
REGIONS = (
    ("Provincial DDA Areas", "subtotal", "provincial"),
    ("Colombo", "area", "provincial"),
    ("Gampaha", "area", "provincial"),
    ("Kalutara", "area", "provincial"),
    ("Kandy", "area", "provincial"),
    ("Matale", "area", "provincial"),
    ("Nuwaraeliya", "area", "provincial"),
    ("Galle", "area", "provincial"),
    ("Matara", "area", "provincial"),
    ("Hambantota", "area", "provincial"),
    ("Jaffna", "area", "provincial"),
    ("Kilinochchi", "area", "provincial"),
    ("Mannar", "area", "provincial"),
    ("Vavuniya", "area", "provincial"),
    ("Mullathivu", "area", "provincial"),
    ("Batticaloa", "area", "provincial"),
    ("Ampara", "area", "provincial"),
    ("Trincomalee", "area", "provincial"),
    ("Kurunegala", "area", "provincial"),
    ("Puttalam", "area", "provincial"),
    ("Anuradhapura", "area", "provincial"),
    ("Badulla", "area", "provincial"),
    ("Moneragala", "area", "provincial"),
    ("Ratnapura", "area", "provincial"),
    ("Kegalle", "area", "provincial"),
    ("Inter Provincial DDA Area", "subtotal", "inter_provincial"),
    ("Polonnaruwa IP", "area", "inter_provincial"),
    ("Ampara IP", "area", "inter_provincial"),
    ("Monaragala IP", "area", "inter_provincial"),
    ("Hambantota IP", "area", "inter_provincial"),
    ("Hasalaka IP", "area", "inter_provincial"),
    ("Anuradhapura IP", "area", "inter_provincial"),
    ("Mahaweli Areas", "subtotal", "mahaweli"),
    ("Mahaweli H", "area", "mahaweli"),
    ("Mahaweli H 1", "area", "mahaweli"),
    ("Mahaweli L", "area", "mahaweli"),
    ("Mahaweli B", "area", "mahaweli"),
    ("Mahaweli C", "area", "mahaweli"),
    ("Mahaweli G", "area", "mahaweli"),
    ("Udawalawa", "area", "mahaweli"),
    ("Mahaweli D", "area", "mahaweli"),
    ("Rambakanoya", "area", "mahaweli"),
    ("Sri Lanka", "national", "national"),
)

# None is a printed dash. Each tuple is (target ha, achievement ha).
CARROT = (
    (2285, 1195),
    (None, None),
    (None, None),
    (None, None),
    (64, 26),
    (None, None),
    (1565, 738),
    (None, None),
    (None, None),
    (None, None),
    (25, 6),
    (6, None),
    (1, None),
    (6, 1),
    (3, 2),
    (None, None),
    (None, None),
    (1, None),
    (None, None),
    (None, None),
    (None, None),
    (608, 421),
    (1, 0),
    (5, 1),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (2285, 1195),
)

LEEKS = (
    (1341, 761),
    (None, None),
    (None, None),
    (None, None),
    (49, 14),
    (None, None),
    (1040, 550),
    (None, None),
    (None, None),
    (None, None),
    (8, 0),
    (None, None),
    (None, None),
    (None, 0),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (241, 196),
    (1, 1),
    (2, 0),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (1341, 761),
)

# October 2025 Yala Final, page 9. Anuradhapura achievement is a printed zero.
OCTOBER_LEEKS = (
    (1333, 1557),
    (None, None),
    (None, None),
    (None, None),
    (53, 50),
    (None, None),
    (1080, 1107),
    (None, None),
    (None, None),
    (None, None),
    (8, 4),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, 0),
    (190, 394),
    (None, None),
    (2, 2),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (1333, 1557),
)


def _rows(crop_code: str, values: tuple, report: dict) -> list[dict]:
    if len(values) != len(REGIONS):
        raise ValueError(f"{crop_code} has {len(values)} rows, expected {len(REGIONS)}")
    rows = []
    for (name, level, group), (target, achievement) in zip(REGIONS, values):
        rows.append(
            {
                "source": "doa",
                "report_date": report["report_date"],
                "season_label": report["season_label"],
                "as_of_date": report["as_of_date"],
                "crop_code": crop_code,
                "region_name": name,
                "region_level": level,
                "region_group": group,
                "target_ha": target,
                "achievement_ha": achievement,
                "target_missing": target is None,
                "achievement_missing": achievement is None,
                "page_number": report["page_number"],
                "source_file": report["source_file"],
            }
        )
    return rows


def _sum_check(frame: pd.DataFrame) -> None:
    grouped = frame.groupby(["report_date", "crop_code"], sort=False)
    for (report_date, crop_code), crop in grouped:
        label = f"{report_date} {crop_code}"
        for column in ("target_ha", "achievement_ha"):
            national = crop.loc[crop["region_level"] == "national", column].iloc[0]
            if pd.isna(national):
                raise ValueError(f"{label} {column} has no Sri Lanka figure")
            areas = crop.loc[crop["region_level"] == "area", column].sum()
            if abs(float(areas) - float(national)) > 0.05:
                raise ValueError(
                    f"{label} {column}: area rows sum to {areas}, Sri Lanka is {national}"
                )
            for group_name, group in crop.groupby("region_group"):
                if group_name == "national":
                    continue
                subtotal = group.loc[group["region_level"] == "subtotal", column].iloc[0]
                leaves = group.loc[group["region_level"] == "area", column]
                # Some reports print the group name as a heading with no total.
                if pd.isna(subtotal):
                    continue
                leaf_sum = leaves.sum()
                if abs(float(leaf_sum) - float(subtotal)) > 0.05:
                    raise ValueError(
                        f"{label} {column}: {group_name} rows sum to {leaf_sum}, subtotal is {subtotal}"
                    )


def _snapshot_rows() -> list[dict]:
    if not SNAPSHOT_CSV.exists():
        return []
    rows = []
    with SNAPSHOT_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row["report_date"], row["crop_code"]) in CHECKED_CROPS:
                continue
            row["target_ha"] = None if row["target_ha"] == "" else float(row["target_ha"])
            row["achievement_ha"] = None if row["achievement_ha"] == "" else float(row["achievement_ha"])
            row["target_missing"] = row["target_missing"] == "True"
            row["achievement_missing"] = row["achievement_missing"] == "True"
            row["page_number"] = int(row["page_number"])
            rows.append(row)
    return rows


def extent_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        _rows("carrot", CARROT, JULY_2026)
        + _rows("leeks", LEEKS, JULY_2026)
        + _rows("leeks", OCTOBER_LEEKS, OCTOBER_2025_FINAL)
        + _snapshot_rows()
    )
    _sum_check(frame)
    duplicate = frame.duplicated(["report_date", "crop_code", "region_name"])
    if duplicate.any():
        raise ValueError(f"{int(duplicate.sum())} duplicate extent keys")
    return frame


def load_extent() -> dict[str, int]:
    frame = extent_frame()
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS curated")
        con.register("cultivation_extent", frame)
        con.execute(
            "CREATE OR REPLACE TABLE curated.cultivation_extent AS SELECT * FROM cultivation_extent"
        )
        export_path = str(EXPORT_DIR / "cultivation_extent.parquet").replace("\\", "/")
        con.execute(f"COPY curated.cultivation_extent TO '{export_path}' (FORMAT PARQUET)")
        missing = con.execute(
            """
            SELECT count(*) FROM curated.cultivation_extent
            WHERE target_missing OR achievement_missing
            """
        ).fetchone()[0]
    finally:
        con.close()
    return {"rows": len(frame), "missing_cells": int(missing)}


def main() -> None:
    counts = load_extent()
    print(f"warehouse: {DUCKDB_PATH}")
    print(
        f"cultivation_extent rows: {counts['rows']:,}  "
        f"rows with a missing cell: {counts['missing_cells']:,}"
    )


if __name__ == "__main__":
    main()
