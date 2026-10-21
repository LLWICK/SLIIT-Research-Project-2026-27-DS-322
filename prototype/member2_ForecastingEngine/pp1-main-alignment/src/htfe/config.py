"""Paths for Member 2.

Member 1's four contract files are used when they are all in data/member1.
Until then, the current external data folder is used. The monthly cultivation
file is optional and is read from one path, never by scanning the data disk.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MEMBER1_DIR = REPO_ROOT / "data" / "member1"
OUTPUTS = REPO_ROOT / "outputs"
STUDIO_DATA = REPO_ROOT / "app" / "data"
ARTIFACTS = OUTPUTS

FALLBACK_ROOT = Path(os.environ.get("COBWEB_DATA_ROOT", r"D:\SLIIT\4th Year\Reaserch\data sets"))
DATA_ROOT = FALLBACK_ROOT
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
AUDIT_DIR = PROCESSED_DIR / "audit"
LOCAL_EXCEL = DATA_ROOT / "Vegetable Prices.xlsx"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36 "
    "SLIIT-DS322-research-crawler (academic use)"
)
REQUEST_TIMEOUT = 60
MIN_DELAY_SECONDS = 1.0

CONTRACT = {
    "prices": "harti_prices_weekly_long.csv",
    "weather": "weather_weekly_open_meteo.csv",
    "macros": "macros_published.csv",
    "extent": "dcs_highland_seasonal.csv",
}
CULTIVATION_NAME = "cultivation_monthly.csv"


def member1_ready() -> bool:
    return all((MEMBER1_DIR / name).is_file() for name in CONTRACT.values())


def processed_dir() -> Path:
    """Folder that holds the four source files. Built tables go to outputs/panel."""
    if member1_ready():
        return MEMBER1_DIR
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR


def contract_path(key: str) -> Path:
    return processed_dir() / CONTRACT[key]


def cultivation_monthly_path() -> Path | None:
    direct = MEMBER1_DIR / CULTIVATION_NAME
    if direct.is_file():
        return direct
    fallback = PROCESSED_DIR / CULTIVATION_NAME
    if fallback.is_file():
        return fallback
    return None


def built_file(name: str) -> Path:
    local = OUTPUTS / "panel" / name
    if local.exists():
        return local
    fallback = PROCESSED_DIR / "latest" / name
    if fallback.exists():
        return fallback
    return local


def origin_map_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "maps" / "origin_map.yaml"


def raw_dir(source: str) -> Path:
    path = RAW_DIR / source
    path.mkdir(parents=True, exist_ok=True)
    return path
