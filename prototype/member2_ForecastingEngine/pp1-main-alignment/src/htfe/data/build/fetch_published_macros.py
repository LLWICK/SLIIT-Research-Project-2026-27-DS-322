"""Download leak-safe diesel, USD/LKR, and CCPI year-on-year series.

Nothing is filled in by hand. Each request uses curl with a 20 second cap.
A series is written only after published anchor values match the parse.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd
import pdfplumber

from htfe.config import DATA_ROOT, processed_dir

PUBLISHED_PATH = processed_dir() / "macros_published.csv"
RAW_DIR = DATA_ROOT / "raw" / "macros"
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
ANCHORS = {
    "usd_lkr": ("2016-01-01", 144.0716),
    "diesel_lkr_per_litre": ("2026-10-01", 392.0),
    "inflation_2016_01": ("2016-01-31", 1.7),
    "inflation_2022_01": ("2022-01-31", 14.2),
    "inflation_2022_12": ("2022-12-31", 57.2),
    "inflation_2024_01": ("2024-01-31", 6.4),
    "inflation_2025_01": ("2025-01-31", -4.0),
}
INFLATION_LAG_DAYS = 21
INFLATION_AVAILABLE_DATE_RULE = "reference_month_end_plus_21_days"


def _curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["curl.exe", "--max-time", "20", "-fsSL", "-A", "SLIIT-DS322-research", "-o", str(dest), url],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not dest.exists() or dest.stat().st_size < 50:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"fetch failed for {url}: {detail}")


def _parse_cpc_date(text: str) -> pd.Timestamp | None:
    parts = re.split(r"[.\-/]", text.strip())
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    first, second, third = (int(part) for part in parts)
    if len(parts[0]) == 4:
        year, month, day = first, second, third
    elif len(parts[2]) == 4:
        day, month, year = first, second, third
    else:
        return None
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return None


def parse_diesel(html: str) -> pd.DataFrame:
    """Lanka Auto Diesel (LAD) revisions. The revision date is the available date."""
    start = html.lower().find('<tbody id="fueltbody">')
    end = html.lower().find("</tbody>", start)
    if start < 0 or end < 0:
        raise RuntimeError("CPC page has no fuelTbody table")
    rows = []
    for date_text, _lp95, _lp92, lad in re.findall(
        r"<tr><td>([^<]+)</td><td>([^<]*)</td><td>([^<]*)</td><td>([^<]*)</td>",
        html[start:end],
        flags=re.I,
    ):
        available = _parse_cpc_date(date_text)
        price = pd.to_numeric(str(lad).replace(",", ""), errors="coerce")
        if available is None or pd.isna(price):
            continue
        rows.append({"available_date": available, "diesel_lkr_per_litre": float(price)})
    if not rows:
        raise RuntimeError("CPC historical-prices page had no parseable LAD rows")
    frame = pd.DataFrame(rows).drop_duplicates("available_date", keep="first")
    frame = frame.sort_values("available_date").reset_index(drop=True)
    if frame["diesel_lkr_per_litre"].le(0).any() or frame["diesel_lkr_per_litre"].gt(800).any():
        raise RuntimeError("a parsed diesel price is outside 0-800 LKR per litre")
    return frame


def parse_usd_lkr(csv_path: Path) -> pd.DataFrame:
    """CBSL indicative USD rate via the Frankfurter CBSL provider. Rate date is the available date."""
    raw = pd.read_csv(csv_path)
    needed = {"date", "base", "quote", "rate"}
    if not needed.issubset(raw.columns):
        raise RuntimeError(f"FX file columns are {list(raw.columns)}")
    lkr = raw[raw["quote"].astype(str).str.upper().eq("LKR") & raw["base"].astype(str).str.upper().eq("USD")].copy()
    lkr["available_date"] = pd.to_datetime(lkr["date"])
    lkr["usd_lkr"] = pd.to_numeric(lkr["rate"], errors="coerce")
    lkr = lkr.dropna(subset=["available_date", "usd_lkr"])
    lkr = lkr.sort_values("available_date").drop_duplicates("available_date", keep="last")
    if lkr.empty:
        raise RuntimeError("CBSL USD/LKR extract is empty")
    if lkr["usd_lkr"].lt(80).any() or lkr["usd_lkr"].gt(500).any():
        raise RuntimeError("a parsed USD/LKR rate is outside 80-500")
    return lkr[["available_date", "usd_lkr"]].reset_index(drop=True)


def _numeric_lines(text: str | None) -> list[float]:
    values = []
    for line in (text or "").splitlines():
        token = line.strip().replace("%", "")
        if not token or not re.fullmatch(r"-?\d+(?:\.\d+)?", token):
            continue
        values.append(float(token))
    return values


def _month_stamps(years: list[int], months: list[str]) -> list[pd.Timestamp]:
    stamps = []
    month_index = {name: number for number, name in enumerate(MONTHS, start=1)}
    cursor = 0
    for year in years:
        while cursor < len(months):
            name = months[cursor]
            cursor += 1
            if name not in month_index:
                continue
            stamps.append(pd.Timestamp(year=year, month=month_index[name], day=1) + pd.offsets.MonthEnd(0))
            if name == "December":
                break
        else:
            break
    return stamps


def _pdf_blobs(path: Path) -> list[str]:
    blobs: list[str] = []
    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != 1 and path.name.startswith("ccpi2013"):
            raise RuntimeError(f"{path.name} page count changed")
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                for row in table:
                    for cell in row:
                        if cell:
                            blobs.append(cell)
    return blobs


def _blob_starting(blobs: list[str], prefix: str) -> str:
    for blob in blobs:
        if blob.startswith(prefix):
            return blob
    raise RuntimeError(f"no PDF cell starts with {prefix}")


def parse_ccpi_yoy(base_2013: Path, base_2021: Path) -> pd.DataFrame:
    """Published CCPI headline year-on-year.

    The reference month end is not the publication date. The movements PDFs and
    the DCS monthly page do not list a historical release calendar, so the
    available date is the reference month end plus 21 days. Base 2013=100 is
    used through January 2023. Base 2021=100 is used from February 2023. The
    index is not rescaled from one base onto the other.
    """
    old_blobs = _pdf_blobs(base_2013)
    years = [int(token) for token in _blob_starting(old_blobs, "2014").splitlines() if token.strip().isdigit()]
    months = [token.strip() for token in _blob_starting(old_blobs, "January").splitlines() if token.strip() in MONTHS]
    index_blob = _blob_starting(old_blobs, "104.2")
    extra = next(blob for blob in old_blobs if blob.startswith("114.7"))
    index = _numeric_lines(index_blob) + _numeric_lines(extra)
    yoy = _numeric_lines(_blob_starting(old_blobs, "3.8"))
    stamps = _month_stamps(years, months)
    if len(stamps) != len(index):
        raise RuntimeError(f"base 2013 months {len(stamps)} != index {len(index)}")
    if len(yoy) != len(stamps) - 12:
        raise RuntimeError(f"base 2013 YoY {len(yoy)} is not 12 months shorter than the index")
    old = pd.DataFrame(
        {
            "reference_month_end": stamps[12:],
            "inflation_yoy": yoy,
            "inflation_base": "CCPI_2013",
        }
    )

    new_blobs = _pdf_blobs(base_2021)
    year_blob = _blob_starting(new_blobs, "Year")
    new_years = [int(token) for token in year_blob.splitlines() if token.strip().isdigit()]
    month_blob = _blob_starting(new_blobs, "Month")
    new_months = [token.strip() for token in month_blob.splitlines() if token.strip() in MONTHS]
    index_blob = next(blob for blob in new_blobs if blob.startswith("124.3"))
    new_index = _numeric_lines(index_blob)
    new_yoy = _numeric_lines(next(blob for blob in new_blobs if blob.startswith("50.6")))
    new_stamps = _month_stamps(new_years, new_months)
    if len(new_stamps) != len(new_index):
        raise RuntimeError(f"base 2021 months {len(new_stamps)} != index {len(new_index)}")
    # The sheet's first year-on-year figure is February 2023, not January 2023.
    if new_stamps[13] != pd.Timestamp("2023-02-28"):
        raise RuntimeError(new_stamps[13])
    if len(new_yoy) != len(new_stamps) - 13:
        raise RuntimeError(f"base 2021 YoY {len(new_yoy)} does not start in February 2023")
    new = pd.DataFrame(
        {
            "reference_month_end": new_stamps[13:],
            "inflation_yoy": new_yoy,
            "inflation_base": "CCPI_2021",
        }
    )
    combined = pd.concat([old[old["reference_month_end"] <= "2023-01-31"], new], ignore_index=True)
    combined = combined.sort_values("reference_month_end").drop_duplicates("reference_month_end", keep="last")
    if combined["reference_month_end"].duplicated().any():
        raise RuntimeError("duplicate inflation months")
    combined["available_date"] = pd.to_datetime(combined["reference_month_end"]) + pd.Timedelta(days=INFLATION_LAG_DAYS)
    combined["inflation_available_date_rule"] = INFLATION_AVAILABLE_DATE_RULE
    if combined["available_date"].duplicated().any():
        raise RuntimeError("inflation available dates collide after the publication lag")
    return combined.reset_index(drop=True)


def _anchor(frame: pd.DataFrame, column: str, day: str, expected: float, date_column: str = "available_date") -> None:
    stamp = pd.Timestamp(day)
    hit = frame.loc[pd.to_datetime(frame[date_column]).eq(stamp), column]
    if hit.empty or abs(float(hit.iloc[0]) - expected) > 1e-6:
        raise RuntimeError(f"{column} on {day} is {None if hit.empty else float(hit.iloc[0])}, expected {expected}")


def inflation_rule_ok(frame: pd.DataFrame) -> bool:
    """True only when inflation is lagged 21 days after the reference month end."""
    needed = {"available_date", "reference_month_end", "inflation_yoy", "inflation_available_date_rule"}
    if not needed.issubset(frame.columns):
        return False
    inf = frame.dropna(subset=["inflation_yoy", "reference_month_end", "available_date"])
    if inf.empty:
        return False
    if not inf["inflation_available_date_rule"].eq(INFLATION_AVAILABLE_DATE_RULE).all():
        return False
    lag = (pd.to_datetime(inf["available_date"]) - pd.to_datetime(inf["reference_month_end"])).dt.days
    return bool(lag.eq(INFLATION_LAG_DAYS).all())


def _source_paths() -> tuple[Path, Path, Path, Path]:
    return (
        RAW_DIR / "cpc_historical_prices.html",
        RAW_DIR / "cbsl_usd_rates.csv",
        RAW_DIR / "ccpi_movements_base2013.pdf",
        RAW_DIR / "ccpi_movements_base2021.pdf",
    )


def build_published_table(download: bool = False) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cpc_path, fx_path, ccpi_2013, ccpi_2021 = _source_paths()
    if download or not all(path.exists() for path in (cpc_path, fx_path, ccpi_2013, ccpi_2021)):
        _curl("https://ceypetco.gov.lk/historical-prices/", cpc_path)
        _curl(
            "https://api.frankfurter.dev/v2/rates.csv?base=usd&providers=cbsl&from=2016-01-01",
            fx_path,
        )
        _curl(
            "https://www.statistics.gov.lk/Resource/en/InflationAndPrices/CCPI/MOVEMENTS_of_CCPI_with_MV_Base2013-100.pdf",
            ccpi_2013,
        )
        _curl(
            "https://www.statistics.gov.lk/Resource/en/InflationAndPrices/CCPI/MOVEMENTS_of_CCPI_with_MV_Base2021.pdf",
            ccpi_2021,
        )
    diesel = parse_diesel(cpc_path.read_text(encoding="utf-8", errors="replace"))
    if diesel.loc[diesel["available_date"] <= "2016-01-04"].empty:
        raise RuntimeError("no diesel revision is available on or before the first forecast Monday")
    usd = parse_usd_lkr(fx_path)
    inflation = parse_ccpi_yoy(ccpi_2013, ccpi_2021)
    _anchor(usd, "usd_lkr", *ANCHORS["usd_lkr"])
    _anchor(diesel, "diesel_lkr_per_litre", *ANCHORS["diesel_lkr_per_litre"])
    for key in (
        "inflation_2016_01",
        "inflation_2022_01",
        "inflation_2022_12",
        "inflation_2024_01",
        "inflation_2025_01",
    ):
        _anchor(inflation, "inflation_yoy", *ANCHORS[key], date_column="reference_month_end")
    if not inflation_rule_ok(inflation):
        raise RuntimeError("inflation available_date is not reference month-end plus 21 days")
    wide = diesel.merge(usd, on="available_date", how="outer").merge(inflation, on="available_date", how="outer")
    wide = wide.sort_values("available_date").reset_index(drop=True)
    return wide


def ensure_published_macros(refresh: bool = False) -> dict:
    """Write the macro file when all three series parse. Leave it absent on failure."""
    if PUBLISHED_PATH.exists() and not refresh:
        frame = pd.read_csv(PUBLISHED_PATH, parse_dates=["available_date"])
        if "reference_month_end" in frame.columns:
            frame["reference_month_end"] = pd.to_datetime(frame["reference_month_end"])
        try:
            anchors_ok = True
            _anchor(frame, "usd_lkr", *ANCHORS["usd_lkr"])
            _anchor(frame, "diesel_lkr_per_litre", *ANCHORS["diesel_lkr_per_litre"])
            _anchor(frame, "inflation_yoy", *ANCHORS["inflation_2025_01"], date_column="reference_month_end")
        except (RuntimeError, KeyError):
            anchors_ok = False
        if anchors_ok and inflation_rule_ok(frame):
            return {"status": "already_on_disk", "path": str(PUBLISHED_PATH), "rows": int(len(frame))}
    frame = build_published_table(download=False)
    PUBLISHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PUBLISHED_PATH, index=False)
    return {
        "status": "rebuilt",
        "path": str(PUBLISHED_PATH),
        "rows": int(len(frame)),
        "inflation_available_date_rule": INFLATION_AVAILABLE_DATE_RULE,
    }


if __name__ == "__main__":
    print(ensure_published_macros(refresh=True))
