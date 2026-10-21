"""Measured scores from the studio evaluation file. 90% is the intended level."""
from __future__ import annotations

import json
from pathlib import Path

EVALUATION = Path(__file__).resolve().parents[1] / "data" / "evaluation.json"


def find_row(evaluation: dict, experiment: str, model: str, horizon: int) -> dict | None:
    for row in evaluation.get("rows") or []:
        if row.get("experiment") == experiment and row.get("model") == model and int(row.get("horizon", -1)) == horizon:
            return row
    return None


def coverage_words(picp: float | None) -> str:
    if picp is None:
        return "not scored on this run"
    value = float(picp)
    if value >= 90:
        return f"{value:.2f}%, at or above the 90% intended level"
    return f"{value:.2f}%, below the 90% intended level"


def b4_row(evaluation: dict | None = None) -> dict | None:
    if evaluation is None:
        if not EVALUATION.is_file():
            return None
        evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    return find_row(evaluation, "B", "lightgbm", 4)


def b4_coverage_sentence(evaluation: dict | None = None) -> str:
    row = b4_row(evaluation)
    picp = None if row is None else row.get("picp_cqr")
    return f"Experiment B LightGBM horizon 4 calibrated coverage is {coverage_words(picp)}."
