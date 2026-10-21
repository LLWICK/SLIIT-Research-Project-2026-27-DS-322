"""Build the local DuckDB warehouse and a Parquet copy of the price panel."""

import duckdb
import pandas as pd

from data_layer.harti import weekly_panel
from data_layer.paths import DUCKDB_PATH, EXPORT_DIR, HARTI_WORKBOOK, WAREHOUSE_DIR
from data_layer.scope import CROPS, MARKET_DISTRICT, MARKETS


def build() -> dict[str, int]:
    if not HARTI_WORKBOOK.exists():
        raise FileNotFoundError(f"HARTI workbook not found: {HARTI_WORKBOOK}")

    prices = weekly_panel(HARTI_WORKBOOK)
    markets = pd.DataFrame(
        [
            {
                "market_code": code,
                "market_name": name,
                "district": MARKET_DISTRICT[code],
                "supply_rule": "national_extent",
            }
            for code, name in MARKETS.items()
        ]
    )
    crops = pd.DataFrame(
        [{"crop_code": code, "crop_name": name} for code, name in CROPS.items()]
    )

    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS curated")
        con.register("prices", prices)
        con.register("markets", markets)
        con.register("crops", crops)
        con.execute("CREATE OR REPLACE TABLE curated.dim_market AS SELECT * FROM markets")
        con.execute("CREATE OR REPLACE TABLE curated.dim_crop AS SELECT * FROM crops")
        con.execute(
            "CREATE OR REPLACE TABLE curated.harti_price_weekly AS SELECT * FROM prices"
        )
        export_path = str(EXPORT_DIR / "harti_price_weekly.parquet").replace("\\", "/")
        con.execute(
            f"COPY curated.harti_price_weekly TO '{export_path}' (FORMAT PARQUET)"
        )
        counts = {
            "prices": con.execute("SELECT count(*) FROM curated.harti_price_weekly").fetchone()[0],
            "missing": con.execute(
                "SELECT count(*) FROM curated.harti_price_weekly WHERE is_missing"
            ).fetchone()[0],
        }
    finally:
        con.close()

    return counts


def main() -> None:
    counts = build()
    print(f"warehouse: {DUCKDB_PATH}")
    print(f"rows: {counts['prices']:,}  missing: {counts['missing']:,}")


if __name__ == "__main__":
    main()
