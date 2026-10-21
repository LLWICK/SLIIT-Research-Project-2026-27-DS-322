"""File locations. Generated warehouse files stay inside this package."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

SOURCE_DIR = PACKAGE_DIR / "source"
HARTI_WORKBOOK = SOURCE_DIR / "Vegetable Prices.xlsx"
CBSL_LONG = SOURCE_DIR / "cbsl_2026_long.csv"

WAREHOUSE_DIR = PACKAGE_DIR / "warehouse"
DUCKDB_PATH = WAREHOUSE_DIR / "cobweb.duckdb"
EXPORT_DIR = PACKAGE_DIR / "export"
