"""Shared paths and HTTP settings for the Member 1 data pipeline.

Raw and processed data live outside the git repo. Override the root with the
COBWEB_DATA_ROOT environment variable on another machine.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("COBWEB_DATA_ROOT", r"D:\SLIIT\4th Year\Reaserch\data sets"))
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


def raw_dir(source: str) -> Path:
    path = RAW_DIR / source
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dir() -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR
