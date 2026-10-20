"""Replace an official output folder only after a run has finished writing."""
from __future__ import annotations

import shutil
from pathlib import Path

from htfe.config import OUTPUTS

STAGING = OUTPUTS / "_staging"


def begin_staging(name: str) -> Path:
    """Empty temporary directory for one experiment. A failed run deletes this directory."""
    stage = STAGING / name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    return stage


def discard_staging(stage: Path) -> None:
    if stage.exists():
        shutil.rmtree(stage)
    if STAGING.exists() and not any(STAGING.iterdir()):
        STAGING.rmdir()


def publish_staging(stage: Path, official: Path) -> None:
    """Move a finished run into place. The previous official folder is removed only after the move."""
    if not stage.is_dir():
        raise RuntimeError(f"Staged run is missing: {stage}")
    official.parent.mkdir(parents=True, exist_ok=True)
    retiring = official.with_name(official.name + ".__retiring")
    if retiring.exists():
        shutil.rmtree(retiring)
    if official.exists():
        official.rename(retiring)
    try:
        stage.rename(official)
    except Exception:
        if retiring.exists() and not official.exists():
            retiring.rename(official)
        raise
    if retiring.exists():
        shutil.rmtree(retiring)
    if STAGING.exists() and not any(STAGING.iterdir()):
        STAGING.rmdir()
