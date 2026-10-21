"""Convert the HARTI "Vegetable Prices.xlsx" workbook into tidy long CSVs.

The workbook (source footer: "Data Mgt.Division/HARTI") has four wide sheets:
Retail - W, Wholesale - W (Year, Location, Varaity, W1..W52[, W53]) and
Retail - M, Wholesale - M (Year, Location, Varaity/ITEM, JAN..DEC).

Outputs (in PROCESSED_DIR):
  harti_prices_weekly_long.csv
  harti_prices_monthly_long.csv

Every value keeps its raw form in ``price_raw``; ``price_lkr_per_kg`` is the
cleaned value and ``quality_flag`` says what was done to it.

Usage (from the ``prototype`` folder):
  python -m htfe.data.ingest.harti_excel_prices [--xlsx PATH]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from htfe.config import LOCAL_EXCEL, processed_dir

SHEETS = {
    "Retail - W": ("retail", "weekly"),
    "Wholesale - W": ("wholesale", "weekly"),
    "Retail - M": ("retail", "monthly"),
    "Wholesale - M": ("wholesale", "monthly"),
}
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# lower-cased raw spelling -> (canonical location, administrative district)
LOCATIONS = {
    "ampara": ("Ampara", "Ampara"),
    "anuradhapura": ("Anuradhapura", "Anuradhapura"),
    "anuradapura": ("Anuradhapura", "Anuradhapura"),
    "badulla": ("Badulla", "Badulla"),
    "bandarawela": ("Bandarawela", "Badulla"),
    "batticaloa": ("Batticaloa", "Batticaloa"),
    "colombo": ("Colombo", "Colombo"),
    "dambulla": ("Dambulla", "Matale"),
    "dehiattakandiya": ("Dehiattakandiya", "Ampara"),
    "embilipitiya": ("Embilipitiya", "Ratnapura"),
    "galenbindunuwewa": ("Galenbindunuwewa", "Anuradhapura"),
    "galenbidunuwewa": ("Galenbindunuwewa", "Anuradhapura"),
    "galle": ("Galle", "Galle"),
    "gampaha": ("Gampaha", "Gampaha"),
    "hambanthota": ("Hambantota", "Hambantota"),
    "hambantota": ("Hambantota", "Hambantota"),
    "hanguranketha": ("Hanguranketha", "Nuwara Eliya"),
    "jaffna": ("Jaffna", "Jaffna"),
    "jafna": ("Jaffna", "Jaffna"),
    "kaluthara": ("Kalutara", "Kalutara"),
    "kandy": ("Kandy", "Kandy"),
    "kegalle": ("Kegalle", "Kegalle"),
    "keppetipola": ("Keppetipola", "Badulla"),
    "kilinochchi": ("Kilinochchi", "Kilinochchi"),
    "kurunegala": ("Kurunegala", "Kurunegala"),
    "mannar": ("Mannar", "Mannar"),
    "matale": ("Matale", "Matale"),
    "matara": ("Matara", "Matara"),
    "meegoda": ("Meegoda", "Colombo"),
    "meegoda(dec)": ("Meegoda", "Colombo"),
    "monaragala": ("Monaragala", "Monaragala"),
    "mullativu": ("Mullaitivu", "Mullaitivu"),
    "mullathivu": ("Mullaitivu", "Mullaitivu"),
    "n'eliya": ("Nuwara Eliya", "Nuwara Eliya"),
    "nuwara eliya": ("Nuwara Eliya", "Nuwara Eliya"),
    "nikaweratiya": ("Nikaweratiya", "Kurunegala"),
    "polonnaruwa": ("Polonnaruwa", "Polonnaruwa"),
    "puttalam": ("Puttalam", "Puttalam"),
    "rathnapura": ("Ratnapura", "Ratnapura"),
    "tabuththegama": ("Thambuttegama", "Anuradhapura"),
    "tabuttegama": ("Thambuttegama", "Anuradhapura"),
    "thabuththegama": ("Thambuttegama", "Anuradhapura"),
    "thissamaharama": ("Tissamaharama", "Hambantota"),
    "trincomalee": ("Trincomalee", "Trincomalee"),
    "trinco": ("Trincomalee", "Trincomalee"),
    "vavuniya": ("Vavuniya", "Vavuniya"),
    "veyangoda": ("Veyangoda", "Gampaha"),
}

# upper-cased raw spelling -> canonical commodity id
COMMODITIES = {
    "ASH PLANTAINS": "ash_plantain",
    "BEETROOT": "beetroot",
    "BIG ONIONS": "big_onion",
    "BIG ONIONS IMPORTED": "big_onion_imported",
    "BIG ONIONS LOCAL": "big_onion_local",
    "BITTER GOURD": "bitter_gourd",
    "BRINJALS": "brinjal",
    "CABBAGE": "cabbage",
    "CAPSICUM": "capsicum",
    "CARROT": "carrot",
    "CUCUMBER": "cucumber",
    "DRUMSTIC": "drumstick",
    "GREEN BEANS": "beans",
    "GREEN CHILLIES": "green_chilli",
    "KNOL-KHOL": "knolkhol",
    "KORALI": "korali",
    "LADIES FINGERS": "ladies_fingers",
    "LEEKS": "leeks",
    "LIME": "lime",
    "LONG BEANS": "long_beans",
    "LUFFA": "luffa",
    "MANIOC": "manioc",
    "PUMPKIN": "pumpkin",
    "RADDISH": "radish",
    "SNAKE GOURD": "snake_gourd",
    "SWEET POTATOES": "sweet_potato",
    "TOMATOES": "tomato",
}

# Wholesale/retail ratio bands used to detect unit problems.
BAG50_RATIO = (25.0, 80.0)          # value is per 50 kg bag -> divide by 50
DIV50_RATIO = (1 / 80.0, 1 / 25.0)  # value was divided by 50 twice -> multiply by 50
# Applied after the onion unit fixes: no genuine per-kg price is 25x off the national retail median.
EXTREME_HIGH_RATIO = 25.0           # keying error -> drop
EXTREME_LOW_RATIO = 1 / 25.0        # keying error -> drop
SPIKE_LOG_THRESHOLD = np.log(3.0)
# Only onions are traded per 50 kg bag; a bag-sized ratio on any other crop is a keying error.
BAG_UNIT_COMMODITIES = {"big_onion", "big_onion_local", "big_onion_imported"}


def _clean_text(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _read_sheet(xlsx: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=sheet, header=3)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "year", df.columns[1]: "location_raw", df.columns[2]: "commodity_raw"})
    if "Unnamed: 55" in df.columns:
        df = df.rename(columns={"Unnamed: 55": "W53"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["location_raw"].notna() & df["commodity_raw"].notna()].copy()
    # One Retail - M row has a blank year inside a year block.
    df["year"] = df["year"].ffill().astype(int)
    df["location_raw"] = df["location_raw"].map(_clean_text)
    df["commodity_raw"] = df["commodity_raw"].map(_clean_text)
    return df


def _melt(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "weekly":
        period_cols = [c for c in df.columns if re.fullmatch(r"W\d+", c)]
    else:
        period_cols = [c for c in df.columns if c in MONTHS]
    long = df.melt(
        id_vars=["year", "location_raw", "commodity_raw"],
        value_vars=period_cols,
        var_name="period_label",
        value_name="price_raw",
    )
    long = long[long["price_raw"].notna()].copy()
    if frequency == "weekly":
        long["period"] = long["period_label"].str[1:].astype(int)
    else:
        long["period"] = long["period_label"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    return long


def harti_week_start(year: pd.Series, week: pd.Series) -> pd.Series:
    """Monday that starts HARTI week ``week`` of ``year``.

    W1 starts on the first Monday on or after 1 January (W-MON). Verified by
    rebuilding the published monthly sheet from the weekly sheet: assigning each
    week to the month of its Monday reproduces 97% of monthly values exactly,
    in every year 2016-2025 (see htfe.data.audit.profile_harti_excel).
    Only years with 53 Mondays (2024 in this workbook) carry a W53.
    """
    jan1 = pd.to_datetime(year.astype(int).astype(str) + "-01-01")
    first_monday = jan1 + pd.to_timedelta((7 - jan1.dt.weekday) % 7, unit="D")
    return first_monday + pd.to_timedelta((week.astype(int) - 1) * 7, unit="D")


def _period_bounds(long: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Assign calendar bounds (weekly: Monday..Sunday; monthly: calendar month)."""
    if frequency == "weekly":
        start = harti_week_start(long["year"], long["period"])
        end = start + pd.Timedelta(days=6)
    else:
        start = pd.to_datetime(dict(year=long["year"], month=long["period"], day=1))
        end = start + pd.offsets.MonthEnd(0)
    long["period_start"] = start.dt.date
    long["period_end"] = end.dt.date
    return long


def _harmonise(long: pd.DataFrame) -> pd.DataFrame:
    key = long["location_raw"].str.lower()
    unknown_loc = sorted(set(key) - set(LOCATIONS))
    if unknown_loc:
        raise ValueError(f"Unmapped locations: {unknown_loc}")
    long["location"] = key.map(lambda k: LOCATIONS[k][0])
    long["district"] = key.map(lambda k: LOCATIONS[k][1])
    ckey = long["commodity_raw"].str.upper()
    unknown_com = sorted(set(ckey) - set(COMMODITIES))
    if unknown_com:
        raise ValueError(f"Unmapped commodities: {unknown_com}")
    long["commodity"] = ckey.map(COMMODITIES)
    return long


def _clean_values(long: pd.DataFrame) -> pd.DataFrame:
    num = pd.to_numeric(long["price_raw"], errors="coerce")
    flag = pd.Series("ok", index=long.index, dtype="object")
    flag[num.isna()] = "non_numeric_dropped"
    flag[num.notna() & (num <= 0)] = "zero_as_missing"
    num = num.where(num > 0)
    long["price_lkr_per_kg"] = num
    long["quality_flag"] = flag
    return long


def _unit_and_outlier_checks(long: pd.DataFrame) -> pd.DataFrame:
    """Compare each value with the national median retail price for the same
    commodity/frequency/period. Retail is the reference because it is never
    reported per bag in this workbook."""
    base = ["frequency", "year", "period"]
    retail = long[(long["price_type"] == "retail") & long["price_lkr_per_kg"].notna()]
    ref = (
        retail.groupby(base + ["commodity"])["price_lkr_per_kg"].median().rename("ref_retail_median")
    )
    # Retail has only the split local/imported onion series from 2018; map the
    # generic 2016-17 "big_onion" to the same reference family.
    long = long.join(ref, on=base + ["commodity"])
    fallback = retail.assign(family=retail["commodity"].str.replace(r"_(local|imported)$", "", regex=True))
    fam_ref = fallback.groupby(base + ["family"])["price_lkr_per_kg"].median().rename("ref_family")
    long["family"] = long["commodity"].str.replace(r"_(local|imported)$", "", regex=True)
    long = long.join(fam_ref, on=base + ["family"])
    long["ref_retail_median"] = long["ref_retail_median"].fillna(long["ref_family"])
    long = long.drop(columns=["ref_family", "family"])

    ratio = long["price_lkr_per_kg"] / long["ref_retail_median"]
    is_ws = (long["price_type"] == "wholesale") & long["commodity"].isin(BAG_UNIT_COMMODITIES)
    bag = is_ws & ratio.between(*BAG50_RATIO)
    long.loc[bag, "price_lkr_per_kg"] = long.loc[bag, "price_lkr_per_kg"] / 50.0
    long.loc[bag, "quality_flag"] = "per_50kg_bag_converted"
    div50 = is_ws & (ratio > DIV50_RATIO[0]) & (ratio < DIV50_RATIO[1])
    long.loc[div50, "price_lkr_per_kg"] = long.loc[div50, "price_lkr_per_kg"] * 50.0
    long.loc[div50, "quality_flag"] = "divided_by_50_corrected"

    ratio = long["price_lkr_per_kg"] / long["ref_retail_median"]
    extreme = long["price_lkr_per_kg"].notna() & ((ratio > EXTREME_HIGH_RATIO) | (ratio < EXTREME_LOW_RATIO))
    long.loc[extreme, "quality_flag"] = "extreme_outlier_dropped"
    long.loc[extreme, "price_lkr_per_kg"] = np.nan
    long["ratio_to_national_retail"] = (long["price_lkr_per_kg"] / long["ref_retail_median"]).round(3)
    return long


def _flag_spikes(long: pd.DataFrame) -> pd.DataFrame:
    """Flag (but keep) values more than 3x away from the centred 5-period
    rolling median of their own series."""
    long = long.sort_values(["price_type", "frequency", "location", "commodity", "year", "period"])
    keys = ["price_type", "frequency", "location", "commodity"]
    logp = np.log(long["price_lkr_per_kg"])
    med = logp.groupby([long[k] for k in keys]).transform(
        lambda s: s.rolling(5, center=True, min_periods=3).median()
    )
    spike = (logp - med).abs() > SPIKE_LOG_THRESHOLD
    long.loc[spike & (long["quality_flag"] == "ok"), "quality_flag"] = "spike_suspect_kept"
    long["is_outlier_flag"] = long["quality_flag"].isin(["spike_suspect_kept", "extreme_outlier_dropped"])
    return long


def build(xlsx: Path) -> dict[str, pd.DataFrame]:
    frames = []
    for sheet, (price_type, frequency) in SHEETS.items():
        df = _read_sheet(xlsx, sheet)
        long = _melt(df, frequency)
        long["sheet"] = sheet
        long["price_type"] = price_type
        long["frequency"] = frequency
        long = _period_bounds(long, frequency)
        frames.append(long)
    data = pd.concat(frames, ignore_index=True)
    data = _harmonise(data)
    data = _clean_values(data)
    out = {}
    for frequency in ("weekly", "monthly"):
        part = data[data["frequency"] == frequency].copy()
        part = _unit_and_outlier_checks(part)
        part = _flag_spikes(part)
        part["unit"] = "LKR/kg"
        part["source"] = "HARTI Data Management Division (Vegetable Prices.xlsx)"
        part = part[
            [
                "source", "sheet", "price_type", "frequency", "year", "period", "period_label",
                "period_start", "period_end", "location_raw", "location", "district",
                "commodity_raw", "commodity", "price_raw", "price_lkr_per_kg", "unit",
                "quality_flag", "is_outlier_flag", "ref_retail_median", "ratio_to_national_retail",
            ]
        ].sort_values(["price_type", "location", "commodity", "year", "period"])
        out[frequency] = part.reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xlsx", type=Path, default=LOCAL_EXCEL, help="Path to Vegetable Prices.xlsx")
    args = parser.parse_args()
    out_dir = processed_dir()
    tables = build(args.xlsx)
    for frequency, df in tables.items():
        path = out_dir / f"harti_prices_{frequency}_long.csv"
        df.to_csv(path, index=False)
        flags = df["quality_flag"].value_counts().to_dict()
        print(f"{path}: {len(df):,} rows; flags={flags}")


if __name__ == "__main__":
    main()
