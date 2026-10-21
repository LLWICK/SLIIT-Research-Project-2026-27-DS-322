"""Build the Maha/Yala calendar used by the Department of Agriculture reports."""

from datetime import date, timedelta

import duckdb
import pandas as pd

from data_layer.paths import DUCKDB_PATH, EXPORT_DIR, WAREHOUSE_DIR

START = date(2015, 1, 1)
END = date(2026, 12, 31)


def season_for(day: date) -> tuple[str, str]:
    """Maha is October-March. Yala is April-September."""
    if day.month >= 10:
        return "maha", f"{day.year}/{str(day.year + 1)[-2:]} Maha"
    if day.month <= 3:
        return "maha", f"{day.year - 1}/{str(day.year)[-2:]} Maha"
    return "yala", f"{day.year} Yala"


def calendar_frame() -> pd.DataFrame:
    rows = []
    day = START
    while day <= END:
        code, label = season_for(day)
        rows.append(
            {
                "calendar_date": day,
                "calendar_year": day.year,
                "month": day.month,
                "season_code": code,
                "season_label": label,
            }
        )
        day += timedelta(days=1)
    return pd.DataFrame(rows)


def load_calendar() -> int:
    frame = calendar_frame()
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS curated")
        con.register("season_calendar", frame)
        con.execute("CREATE OR REPLACE TABLE curated.season_calendar AS SELECT * FROM season_calendar")
        export_path = str(EXPORT_DIR / "season_calendar.parquet").replace("\\", "/")
        con.execute(f"COPY curated.season_calendar TO '{export_path}' (FORMAT PARQUET)")
        count = con.execute("SELECT count(*) FROM curated.season_calendar").fetchone()[0]
    finally:
        con.close()
    return count


def main() -> None:
    count = load_calendar()
    print(f"warehouse: {DUCKDB_PATH}")
    print(f"season_calendar rows: {count:,}")


if __name__ == "__main__":
    main()
