"""Read the HARTI vegetable-price workbook into a long weekly table."""

from pathlib import Path

import pandas as pd

from data_layer.scope import CROP_ALIASES, MARKET_ALIASES, WEEKLY_SHEETS


def read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read one HARTI sheet. The header is the row whose first cell is Year."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_at = raw.index[raw[0].astype(str).str.strip().str.lower() == "year"]
    if len(header_at) == 0:
        raise ValueError(f"{sheet_name}: no header row starting with Year")
    h = int(header_at[0])
    header = raw.iloc[h]
    keep = header.notna() & header.astype(str).str.strip().ne("")
    raw = raw.loc[:, keep]
    header = raw.iloc[h]
    df = raw.iloc[h + 1 :].copy()
    df.columns = header.astype(str).str.strip()
    df = df.rename(
        columns={df.columns[0]: "year", df.columns[1]: "market", df.columns[2]: "crop"}
    )
    df = df.dropna(how="all")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    has_record = df["market"].notna() | df["crop"].notna()
    df.loc[has_record, "year"] = df.loc[has_record, "year"].ffill()
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["market"] = df["market"].astype(str).str.strip()
    df["crop"] = df["crop"].astype(str).str.strip().str.upper()
    return df


def weekly_panel(path: Path) -> pd.DataFrame:
    """Retail and wholesale weekly prices for the scoped crops and markets.

    Blank week cells are kept and marked is_missing. Nothing is imputed.
    """
    frames = [_one_sheet(path, sheet, price_type) for sheet, price_type in WEEKLY_SHEETS.items()]
    panel = pd.concat(frames, ignore_index=True)
    panel["source"] = "harti"
    panel["frequency"] = "weekly"
    return panel[
        [
            "source",
            "price_type",
            "market_code",
            "crop_code",
            "year",
            "week",
            "price",
            "is_missing",
            "sheet_name",
            "source_market_name",
            "source_crop_name",
            "frequency",
        ]
    ]


def _one_sheet(path: Path, sheet_name: str, price_type: str) -> pd.DataFrame:
    wide = read_sheet(path, sheet_name)
    wide["market_code"] = wide["market"].str.upper().map(MARKET_ALIASES)
    wide["crop_code"] = wide["crop"].map(CROP_ALIASES)
    wide = wide[wide["market_code"].notna() & wide["crop_code"].notna()].copy()

    week_cols = [c for c in wide.columns if _week_number(c) is not None]
    long = wide.melt(
        id_vars=["year", "market", "crop", "market_code", "crop_code"],
        value_vars=week_cols,
        var_name="week_label",
        value_name="price",
    )
    long["week"] = long["week_label"].map(_week_number).astype(int)
    long["price"] = pd.to_numeric(long["price"], errors="coerce")
    long["is_missing"] = long["price"].isna()
    long["price_type"] = price_type
    long["sheet_name"] = sheet_name
    long = long.rename(columns={"market": "source_market_name", "crop": "source_crop_name"})
    return long.drop(columns=["week_label"])


def _week_number(column: object) -> int | None:
    text = str(column).strip().upper()
    if not text.startswith("W"):
        return None
    number = text[1:]
    if not number.isdigit():
        return None
    week = int(number)
    if 1 <= week <= 53:
        return week
    return None
