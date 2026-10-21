"""Prove the observed-cultivation loader on a labelled fixture.

The fixture is not a dataset. It is not written into the training table, and
it is not scored. It lives under artifacts/fixture so the data-root search
cannot mistake it for a real monthly file.
"""
from __future__ import annotations

import json

import pandas as pd

from htfe.training.config import ARTIFACTS
from htfe.data.build.cultivation_progress import assert_formula, attach_progress

OUT = ARTIFACTS / "fixture"
NOTE = "FIXTURE_NOT_OBSERVED. These hectares were typed to test the loader. They are not Census or Department of Agriculture records."


def main() -> None:
    assert_formula()
    monthly = pd.DataFrame(
        [
            {"year": 2024, "month": 4, "crop": "carrot", "district": "Nuwara Eliya", "target_ha": 100, "achieved_ha": 40, "available_date": "2024-04-30"},
            {"year": 2024, "month": 4, "crop": "carrot", "district": "Badulla", "target_ha": 50, "achieved_ha": 25, "available_date": "2024-04-30"},
            {"year": 2024, "month": 5, "crop": "carrot", "district": "Nuwara Eliya", "target_ha": 100, "achieved_ha": 80, "available_date": "2024-05-31"},
            {"year": 2024, "month": 5, "crop": "carrot", "district": "Badulla", "target_ha": 50, "achieved_ha": 40, "available_date": "2024-05-31"},
            {"year": 2024, "month": 6, "crop": "carrot", "district": "Nuwara Eliya", "target_ha": 100, "achieved_ha": 140, "available_date": "2024-06-30"},
            {"year": 2024, "month": 6, "crop": "carrot", "district": "Badulla", "target_ha": 50, "achieved_ha": 70, "available_date": "2024-06-30"},
        ]
    )
    monthly["note"] = NOTE
    panel = pd.DataFrame(
        [
            {"crop": "carrot", "market": "Dambulla", "week_start": "2024-04-01", "forecast_issue_week": "2024-04-01"},
            {"crop": "carrot", "market": "Dambulla", "week_start": "2024-05-06", "forecast_issue_week": "2024-05-06"},
            {"crop": "carrot", "market": "Dambulla", "week_start": "2024-06-03", "forecast_issue_week": "2024-06-03"},
            {"crop": "carrot", "market": "Dambulla", "week_start": "2024-07-08", "forecast_issue_week": "2024-07-08"},
        ]
    )
    origin = {"carrot": {"Dambulla": {"supply_districts": ["Nuwara Eliya", "Badulla"]}}}
    checked = attach_progress(panel, _ready(monthly), origin)
    issue = pd.to_datetime(checked["forecast_issue_week"])
    available = pd.to_datetime(checked["cultivation_asof_date"])
    leaked = checked["cultivation_progress"].notna() & available.gt(issue)
    if bool(leaked.any()):
        raise AssertionError("a fixture report dated after the forecast Monday was used")
    june = checked.loc[pd.to_datetime(checked["week_start"]).eq(pd.Timestamp("2024-07-08"))].iloc[0]
    if float(june["cultivation_progress"]) <= 1 or not bool(june["cultivation_target_exceeded"]):
        raise AssertionError("a fixture ratio above 1 was clipped or not flagged")
    early = checked.loc[pd.to_datetime(checked["week_start"]).eq(pd.Timestamp("2024-04-01"))].iloc[0]
    if pd.notna(early["cultivation_progress"]):
        raise AssertionError("April's report was used before 30 April")
    OUT.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUT / "monthly_target_achieved_FIXTURE.csv", index=False)
    checked.to_csv(OUT / "asof_join_FIXTURE.csv", index=False)
    rows = []
    for _, row in checked.iterrows():
        progress = row["cultivation_progress"]
        exceeded = row["cultivation_target_exceeded"]
        available = row["cultivation_asof_date"]
        rows.append(
            {
                "week_start": pd.Timestamp(row["week_start"]).strftime("%Y-%m-%d"),
                "cultivation_progress": None if pd.isna(progress) else float(progress),
                "cultivation_target_exceeded": None if pd.isna(exceeded) else bool(exceeded),
                "cultivation_asof_date": None if pd.isna(available) else str(pd.Timestamp(available).date()),
            }
        )
    report = {
        "status": "loader_ok_not_a_result",
        "observed_cultivation_data_available": False,
        "note": NOTE,
        "rows": rows,
    }
    (OUT / "fixture_report.json").write_text(json.dumps(report, indent=2, default=str, allow_nan=False), encoding="utf-8")
    print(f"wrote {OUT / 'fixture_report.json'}", flush=True)


def _ready(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["crop_key"] = out["crop"].str.lower()
    out["district_key"] = out["district"].str.casefold()
    out["available_date"] = pd.to_datetime(out["available_date"])
    return out


if __name__ == "__main__":
    main()
