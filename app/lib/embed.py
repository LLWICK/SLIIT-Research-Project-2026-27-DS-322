"""Open another member's existing Streamlit app without editing their files."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import streamlit as st

PROTOTYPE = Path(__file__).resolve().parents[3]

MEMBER_APPS = {
    "3": PROTOTYPE / "member3_MSRS_DCVS" / "app.py",
    "4": PROTOTYPE / "member4_AgenticSimulation" / "veggie-abm" / "app.py",
}

_PATCHED = False


def _silence_set_page_config() -> None:
    """Their apps call set_page_config. Ours already did — do not touch their source."""
    global _PATCHED
    if _PATCHED:
        return
    st.set_page_config = lambda *args, **kwargs: None  # type: ignore[method-assign]
    _PATCHED = True


def run_existing_app(member_id: str) -> None:
    app_path = MEMBER_APPS.get(member_id)
    if app_path is None:
        st.error(f"No existing app is registered for Member {member_id}.")
        return
    if not app_path.exists():
        st.warning(f"Member {member_id}'s app is not on this branch yet.")
        st.code(str(app_path), language="text")
        return

    _silence_set_page_config()
    app_dir = app_path.parent
    previous_cwd = Path.cwd()
    inserted_path = False
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
        inserted_path = True

    try:
        os.chdir(app_dir)
        runpy.run_path(str(app_path), run_name=f"member{member_id}_app")
    except ModuleNotFoundError as exc:
        st.error(f"Member {member_id}'s app needs its own packages ({exc.name}).")
        st.code(f"pip install -r {app_dir / 'requirements.txt'}", language="bash")
        st.caption(f"Or run theirs separately: `streamlit run {app_path}`")
    except FileNotFoundError as exc:
        st.error(f"Member {member_id}'s app could not open a local file: {exc}")
        st.caption("Their relative paths are resolved from their own folder. No files were changed.")
    except Exception as exc:  # noqa: BLE001 — surface their app error, do not rewrite it
        st.exception(exc)
    finally:
        os.chdir(previous_cwd)
        if inserted_path and sys.path and sys.path[0] == str(app_dir):
            sys.path.pop(0)
