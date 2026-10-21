"""Load the 2026 CBSL daily prices that overlap the project scope.

Pettah is stored as Colombo. Leeks, Meegoda, Nuwara Eliya, and Narahenpita
are not in this source, so they are left out. The series stays separate from HARTI.
"""

import duckdb
import pandas as pd

from data_layer.paths import CBSL_LONG, DUCKDB_PATH, EXPORT_DIR, WAREHOUSE_DIR

CROP_ALIASES = {"Carrot": "carrot", "Tomato": "tomato"}
MARKET_ALIASES = {"Pettah": "colombo", "Dambulla": "dambulla"}


def price_frame(path=CBSL_LONG) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CBSL price file not found: {path}")
    raw = pd.read_csv(path)
    raw["crop_code"] = raw["commodity"].map(CROP_ALIASES)
    raw["market_code"] = raw["market"].map(MARKET_ALIASES)
    scoped = raw[raw["crop_code"].notna() & raw["market_code"].notna()].copy()
    scoped["report_date"] = pd.to_datetime(scoped["report_date"]).dt.date
    scoped["price"] = pd.to_numeric(scoped["today"], errors="coerce")
    scoped["baseline_price"] = pd.to_numeric(scoped["baseline"], errors="coerce")
    scoped["is_missing"] = scoped["price"].isna()
    scoped["source"] = "cbsl"
    scoped["unit"] = "Rs/kg"
    scoped = scoped.rename(columns={"market": "source_market_name", "commodity": "source_crop_name"})
    duplicate = scoped.duplicated(["report_date", "price_type", "market_code", "crop_code"])
    if duplicate.any():
        raise ValueError(f"{int(duplicate.sum())} duplicate CBSL price keys")
    return scoped[
        [
            "source",
            "report_date",
            "price_type",
            "market_code",
            "crop_code",
            "price",
            "baseline_price",
            "is_missing",
            "unit",
            "source_market_name",
            "source_crop_name",
        ]
    ].sort_values(["report_date", "price_type", "market_code", "crop_code"])


def load_cbsl() -> dict[str, int]:
    frame = price_frame()
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS curated")
        con.register("cbsl_price_daily", frame)
        con.execute("CREATE OR REPLACE TABLE curated.cbsl_price_daily AS SELECT * FROM cbsl_price_daily")
        export_path = str(EXPORT_DIR / "cbsl_price_daily.parquet").replace("\\", "/")
        con.execute(f"COPY curated.cbsl_price_daily TO '{export_path}' (FORMAT PARQUET)")
        missing = con.execute(
            "SELECT count(*) FROM curated.cbsl_price_daily WHERE is_missing"
        ).fetchone()[0]
    finally:
        con.close()
    return {"rows": len(frame), "missing": missing}


def main() -> None:
    counts = load_cbsl()
    print(f"warehouse: {DUCKDB_PATH}")
    print(f"cbsl_price_daily rows: {counts['rows']:,}  missing: {counts['missing']:,}")


if __name__ == "__main__":
    main()
