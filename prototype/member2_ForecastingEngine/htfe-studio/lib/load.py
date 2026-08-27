"""Load frozen HTFE artifacts and (optionally) detect Member 1's extract."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"
OUTPUT_DIR = APP_DIR / "outputs"

STUDIO_FILES = (
    "meta",
    "coverage",
    "split",
    "data_dictionary",
    "origin_map",
    "series",
    "cobweb",
    "comparison",
    "ablation",
    "commitment_weeks",
    "importance",
    "live_forecast",
    "models",
)


def find_repo_root(start: Path | None = None) -> Path | None:
    here = start or Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data" / "v1" / "analysis_ready_price.csv").exists():
            return parent
    return None


def _read_json(name: str) -> Any:
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing demo artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_studio() -> dict[str, Any]:
    data = {key: _read_json(key) for key in STUDIO_FILES}
    data["dictionary"] = data.pop("data_dictionary")
    data["originMap"] = data.pop("origin_map")
    data["weeks"] = data.pop("commitment_weeks")
    data["live"] = data.pop("live_forecast")
    return data


@st.cache_data(show_spinner=False)
def load_predictions() -> list[dict[str, Any]]:
    payload = _read_json("predictions")
    if isinstance(payload, dict):
        return list(payload.get("lightgbm") or [])
    return list(payload)


@st.cache_data(show_spinner=False)
def member1_status() -> dict[str, Any]:
    root = find_repo_root()
    if root is None:
        return {"found": False}

    price_path = root / "data" / "v1" / "analysis_ready_price.csv"
    commit_path = root / "data" / "v1" / "analysis_ready_commitment.csv"
    weather_path = root / "data" / "v1" / "weather_data.json"

    price = pd.read_csv(price_path)
    wholesale = price[price["price_type"].astype(str).str.lower() == "wholesale"]
    crops = {"carrot", "leek", "tomato"}
    markets = {"colombo", "badulla", "nuwara eliya"}
    scoped = wholesale[
        wholesale["crop"].astype(str).str.lower().isin(crops)
        & wholesale["district"].astype(str).str.lower().isin(markets)
    ]

    return {
        "found": True,
        "price_rows": int(len(price)),
        "wholesale_rows": int(len(wholesale)),
        "scoped_rows": int(len(scoped)),
        "commitment_rows": int(sum(1 for _ in open(commit_path, encoding="utf-8"))) - 1 if commit_path.exists() else 0,
        "has_weather": weather_path.exists(),
        "price_path": str(price_path),
    }


def write_forecast_packet(request: dict[str, Any], packet: dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "forecast_packet.json"
    envelope = {
        "from_member": 2,
        "to_member": 3,
        "engine": "Hybrid Temporal Forecasting Engine",
        "request": request,
        "response": packet,
    }
    last = st.session_state.get("_last_packet")
    if last == envelope:
        return path
    path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    st.session_state["_last_packet"] = envelope
    return path


def title_case(value: str) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


def lkr(value: float, digits: int = 0) -> str:
    return f"Rs. {value:,.{digits}f}"
