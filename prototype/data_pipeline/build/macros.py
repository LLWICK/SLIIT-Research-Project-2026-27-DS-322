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


def _peek_columns(path: Path) -> set[str]:
    if path.suffix.lower() == ".csv":
        peeked = pd.read_csv(path, nrows=3)
    elif path.suffix.lower() == ".parquet":
        peeked = pd.read_parquet(path).head(3)
    else:
        peeked = pd.read_excel(path, nrows=3)
    return set(_normalise(peeked).columns)


def find_macro_table() -> Path | None:
    """Prefer the published as-of file, then any dated file with all three series."""
    preferred = processed_dir() / "macros_published.csv"
    if preferred.exists():
        try:
            columns = _peek_columns(preferred)
        except Exception:
            columns = set()
        if "available_date" in columns and all(name in columns for name in MACRO_COLUMNS):
            return preferred
    if not DATA_ROOT.exists():
        return None
    partial: Path | None = None
    for path in DATA_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".xlsx", ".parquet"}:
            continue
        if path.stat().st_size > 40_000_000:
            continue
        try:
            columns = _peek_columns(path)
        except Exception:
            continue
        if "available_date" not in columns:
            continue
        present = [name for name in MACRO_COLUMNS if name in columns]
        if len(present) == len(MACRO_COLUMNS):
            return path
        if present and partial is None:
            partial = path
    return partial


def load_macro_series(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        series = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        series = pd.read_parquet(path)
    else:
        series = pd.read_excel(path)
    series = _normalise(series)
    if "available_date" not in series.columns:
        raise ValueError(f"{path} has no available_date")
    series["available_date"] = pd.to_datetime(series["available_date"])
    keep = ["available_date", *[name for name in MACRO_COLUMNS if name in series.columns]]
    return series[keep].dropna(subset=["available_date"]).sort_values("available_date")


def asof_macros(frame: pd.DataFrame, date_column: str) -> tuple[pd.DataFrame, dict]:
    """Join each macro on its own available date. A later print of one series does not blank the others."""
    found = find_macro_table()
    out = frame.copy()
    out[date_column] = pd.to_datetime(out[date_column]).astype("datetime64[ns]")
    for name in MACRO_COLUMNS:
        out[name] = pd.NA
    if found is None:
        return out, {"status": "not_found", "columns": [], "path": None, "non_null_share": {}}
    series = load_macro_series(found)
    series["available_date"] = pd.to_datetime(series["available_date"]).astype("datetime64[ns]")
    out = out.sort_values(date_column)
    used = []
    shares = {}
    for name in MACRO_COLUMNS:
        if name not in series.columns:
            shares[name] = 0.0
            continue
        right = series[["available_date", name]].dropna().sort_values("available_date")
        right["available_date"] = pd.to_datetime(right["available_date"]).astype("datetime64[ns]")
        if right.empty:
            shares[name] = 0.0
            continue
        right = right.rename(columns={name: "_macro_value"})
        merged = pd.merge_asof(
            out[[date_column]],
            right,
            left_on=date_column,
            right_on="available_date",
            direction="backward",
        )
        leaked = merged["available_date"].notna() & merged["available_date"].gt(merged[date_column])
        if bool(leaked.any()):
            raise AssertionError(f"{name} joined a value published after {date_column}")
        out[name] = pd.to_numeric(merged["_macro_value"], errors="coerce").to_numpy()
        used.append(name)
        shares[name] = round(float(pd.Series(out[name]).notna().mean()), 4)
    return out, {"status": "joined", "columns": used, "path": str(found), "non_null_share": shares}


def write_macro_note(meta: dict) -> None:
    note_path = processed_dir() / "audit" / "macros_gap.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    if meta.get("status") != "joined":
        note_path.write_text(
            "\n".join(
                [
                    "# Macroeconomic series — not found",
                    "",
                    "Methodology.docx section 3.8.4 lists diesel price, USD/LKR, and inflation as candidate predictors.",
                    "Searched the data root for a dated table containing diesel_lkr_per_litre, usd_lkr, and inflation_yoy.",
                    "No leak-safe series was joined, and no series was synthesised.",
                    "Experiment B is unavailable until all three are non-null from a real as-of series.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return
    note_path.write_text(
        "\n".join(
            [
                "# Macroeconomic series — joined as-of",
                "",
                f"File: `{meta['path']}`.",
                "Each column uses only a row whose available_date is on or before the forecast Monday.",
                "Diesel is the CPC Lanka Auto Diesel revision (available on the revision date).",
                "USD/LKR is the CBSL indicative rate (available on the rate date).",
                "Inflation is published CCPI headline year-on-year. The available date is the reference month end plus 21 days, because the reference month is not the publication date.",
                "Base 2013=100 is used through January 2023. Base 2021=100 is used from February 2023. The index was not rescaled.",
                "Annual World Bank pump-price and inflation dumps are not used.",
                f"Non-null shares: {meta['non_null_share']}.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def attach_macros(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join already-published macro values onto week_start. Leave the columns out when no file exists."""
    merged, meta = asof_macros(panel, "week_start")
    write_macro_note(meta)
    if meta["status"] != "joined":
        return merged.drop(columns=list(MACRO_COLUMNS), errors="ignore"), meta
    return merged, meta
