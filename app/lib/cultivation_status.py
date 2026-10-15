"""Read the known cultivation files for the studio. Does not train."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@st.cache_data(show_spinner=False)
def load_cultivation_status() -> dict:
    from htfe.data.build.cultivation_progress import inspect_cultivation_sources

    return inspect_cultivation_sources()


def render_cultivation_status(*, compact: bool = False) -> None:
    from lib.experiment_c import render_cultivation_audit

    render_cultivation_audit(compact=compact)
