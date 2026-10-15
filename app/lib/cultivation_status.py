"""Read the known cultivation files for the studio. Does not train."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from lib.load import call_cached

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _file_token(path: Path | None) -> tuple[str, int, int]:
    if path is None or not path.is_file():
        return ("" if path is None else str(path), 0, 0)
    stat = path.stat()
    return (str(path), int(stat.st_mtime_ns), int(stat.st_size))


def _cultivation_token() -> tuple[tuple[str, int, int], ...]:
    from htfe.data.build.cultivation_progress import doa_progress_path, find_monthly_table

    status = Path(__file__).resolve().parents[2] / "outputs" / "experiment_c" / "status.json"
    return (
        _file_token(doa_progress_path()),
        _file_token(find_monthly_table()),
        _file_token(status),
    )


@st.cache_data(show_spinner=False)
def _load_cultivation_status(token: tuple[tuple[str, int, int], ...]) -> dict:
    del token
    from htfe.data.build.cultivation_progress import inspect_cultivation_sources

    return inspect_cultivation_sources()


def load_cultivation_status() -> dict:
    return call_cached(_load_cultivation_status, (_cultivation_token(),), "Opening the cultivation file")


def render_cultivation_status(*, compact: bool = False) -> None:
    from lib.experiment_c import render_cultivation_audit

    render_cultivation_audit(compact=compact)
