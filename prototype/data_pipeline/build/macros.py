"""As-of macroeconomic columns. Real published series only — nothing is synthesised."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.config import DATA_ROOT, processed_dir

MACRO_COLUMNS = ("diesel_lkr_per_litre", "usd_lkr", "inflation_yoy")
ALIASES = {
    "diesel": "diesel_lkr_per_litre",
    "diesel price": "diesel_lkr_per_litre",
    "diesel_lkr": "diesel_lkr_per_litre",
    "usd/lkr": "usd_lkr",
    "usd lkr": "usd_lkr",
    "exchange_rate": "usd_lkr",
    "ccpi_yoy": "inflation_yoy",
    "inflation": "inflation_yoy",
    "inflation yoy": "inflation_yoy",
    "date": "available_date",
    "month_end": "available_date",
}


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        renamed[column] = ALIASES.get(key, key.replace(" ", "_"))
    return frame.rename(columns=renamed)


def find_macro_table() -> Path | None:
    if not DATA_ROOT.exists():
        return None
    for path in DATA_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".xlsx", ".parquet"}:
            continue
        if path.stat().st_size > 40_000_000:
            continue
        try:
            if path.suffix.lower() == ".csv":
                peeked = pd.read_csv(path, nrows=3)
            elif path.suffix.lower() == ".parquet":
                peeked = pd.read_parquet(path).head(3)
            else:
                peeked = pd.read_excel(path, nrows=3)
        except Exception:
            continue
        columns = set(_normalise(peeked).columns)
        if "available_date" in columns and any(name in columns for name in MACRO_COLUMNS):
            return path
    return None


def attach_macros(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join already-published macro values. Leave the columns out when no file exists."""
    found = find_macro_table()
    note_path = processed_dir() / "audit" / "macros_gap.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    if found is None:
        note_path.write_text(
            "\n".join(
                [
                    "# Macroeconomic series — not found",
                    "",
                    "Methodology.docx section 3.8.4 lists diesel price, USD/LKR, and inflation as candidate predictors.",
                    "Searched the data root for a dated table containing at least one of diesel_lkr_per_litre, usd_lkr, inflation_yoy.",
                    "No such file is on disk, and no series was synthesised.",
                    "Feature set B therefore contains origin weather only, until a real macro file is added.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return panel, {"status": "not_found", "columns": [], "path": None}
    if found.suffix.lower() == ".csv":
        series = pd.read_csv(found)
    elif found.suffix.lower() == ".parquet":
        series = pd.read_parquet(found)
    else:
        series = pd.read_excel(found)
    series = _normalise(series)
    series["available_date"] = pd.to_datetime(series["available_date"])
    keep = ["available_date", *[name for name in MACRO_COLUMNS if name in series.columns]]
    series = series[keep].dropna(subset=["available_date"]).sort_values("available_date")
    left = panel.copy()
    left["week_start"] = pd.to_datetime(left["week_start"])
    left = left.sort_values("week_start")
    merged = pd.merge_asof(left, series, left_on="week_start", right_on="available_date", direction="backward")
    merged = merged.drop(columns=["available_date"])
    used = [name for name in MACRO_COLUMNS if name in merged.columns]
    note_path.write_text(
        f"Joined `{found}` as-of week_start. Columns: {', '.join(used)}.\n",
        encoding="utf-8",
    )
    return merged, {"status": "joined", "columns": used, "path": str(found)}
