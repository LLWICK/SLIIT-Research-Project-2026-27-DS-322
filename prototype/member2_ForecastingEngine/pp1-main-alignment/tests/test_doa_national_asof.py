"""National as-of rules on a tiny fixture. The full DOA file is not read."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htfe.data.build.doa_national_progress import (
    assert_season_parser,
    attach_national_asof,
    audit_progress,
)


def _reports() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "available_date": "2024-04-15",
                "observation_date": "2024-04-01",
                "season": "Yala 2024",
                "crop": "carrot",
                "district": "NATIONAL",
                "scope_level": "national",
                "target_ha": 100,
                "achieved_ha": 140,
                "source_report": "fixture-april",
            },
            {
                "available_date": "2024-05-01",
                "observation_date": "2024-05-01",
                "season": "Yala 2024",
                "crop": "carrot",
                "district": "NATIONAL",
                "scope_level": "national",
                "target_ha": 0,
                "achieved_ha": 10,
                "source_report": "fixture-zero-target",
            },
            {
                "available_date": "2024-05-01",
                "observation_date": "2024-05-01",
                "season": "Yala 2024",
                "crop": "carrot",
                "district": "Nuwara Eliya",
                "scope_level": "district",
                "target_ha": 10,
                "achieved_ha": 90,
                "source_report": "fixture-district",
            },
            {
                "available_date": "2024-05-01",
                "observation_date": "2024-05-01",
                "season": "Maha 2023/24",
                "crop": "carrot",
                "district": "NATIONAL",
                "scope_level": "national",
                "target_ha": 80,
                "achieved_ha": 40,
                "source_report": "fixture-wrong-season",
            },
            {
                "available_date": "2024-06-15",
                "observation_date": "2024-06-01",
                "season": "Yala 2024",
                "crop": "carrot",
                "district": "NATIONAL",
                "scope_level": "national",
                "target_ha": 100,
                "achieved_ha": 50,
                "source_report": "fixture-june",
            },
            {
                "available_date": "2021-04-15",
                "observation_date": "2021-04-01",
                "season": "Yala 2021",
                "crop": "carrot",
                "district": "NATIONAL",
                "scope_level": "national",
                "target_ha": 100,
                "achieved_ha": 20,
                "source_report": "fixture-2021",
            },
        ]
    )


def _issue(*dates: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "crop_id": ["carrot"] * len(dates),
            "market_id": ["dambulla"] * len(dates),
            "forecast_date": list(dates),
        }
    )


def test_future_report_ignored_and_ratio_above_one_kept() -> None:
    assert_season_parser()
    audited = audit_progress(_reports())
    zero = audited[audited["source_report"].eq("fixture-zero-target")].iloc[0]
    if bool(zero["valid_progress_row"]) or pd.notna(zero["cultivation_progress_ratio"]):
        raise AssertionError("target_ha <= 0 was turned into a progress ratio")
    if float(zero["target_ha"]) != 0:
        raise AssertionError("the zero target was replaced")

    asof = attach_national_asof(
        _issue("2024-04-01", "2024-05-06", "2024-06-15", "2024-07-08", "2021-05-03"),
        audited,
    )
    asof["forecast_date"] = pd.to_datetime(asof["forecast_date"])
    asof = asof.set_index("forecast_date")

    early = asof.loc[pd.Timestamp("2024-04-01")]
    if pd.notna(early["cultivation_progress_ratio_real"]):
        raise AssertionError("a report dated after 1 April was used")

    may = asof.loc[pd.Timestamp("2024-05-06")]
    ratio = float(may["cultivation_progress_ratio_real"])
    if abs(ratio - 1.4) > 1e-9 or ratio <= 1:
        raise AssertionError(f"ratio above 1 was clipped or the wrong report was used: {ratio}")
    if int(may["cultivation_target_exceeded_real"]) != 1:
        raise AssertionError("target exceeded was not kept")
    if pd.Timestamp(may["cultivation_report_date"]) != pd.Timestamp("2024-04-15"):
        raise AssertionError("May used a report other than 15 April")
    if float(may["cultivation_target_ha_real"]) <= 0:
        raise AssertionError("a non-positive target was attached")
    if str(may["cultivation_source_scope"]) != "national":
        raise AssertionError(may["cultivation_source_scope"])

    same_day = asof.loc[pd.Timestamp("2024-06-15")]
    if abs(float(same_day["cultivation_progress_ratio_real"]) - 0.5) > 1e-9:
        raise AssertionError("the report published on the forecast Monday was ignored")

    july = asof.loc[pd.Timestamp("2024-07-08")]
    if abs(float(july["cultivation_progress_ratio_real"]) - 0.5) > 1e-9:
        raise AssertionError("July did not keep the June ratio")
    if float(july["cultivation_progress_ratio_real"]) > 1:
        raise AssertionError("a later ratio was replaced with the earlier ratio above 1")

    year_2021 = asof.loc[pd.Timestamp("2021-05-03")]
    if pd.notna(year_2021["cultivation_progress_ratio_real"]):
        raise AssertionError("2021 received a cultivation value")
    if str(year_2021["cultivation_source_scope"]) != "excluded_issue_year":
        raise AssertionError(year_2021["cultivation_source_scope"])

    future = asof["cultivation_report_date"].notna() & (
        pd.to_datetime(asof["cultivation_report_date"]) > asof.index
    )
    if bool(future.any()):
        raise AssertionError("a future-dated report was attached")


def test_wrong_season_is_not_attached() -> None:
    reports = _reports()
    extra = pd.DataFrame(
        [
            {
                "available_date": "2023-12-20",
                "observation_date": "2023-12-01",
                "season": "Yala 2024",
                "crop": "carrot",
                "district": "NATIONAL",
                "scope_level": "national",
                "target_ha": 100,
                "achieved_ha": 10,
                "source_report": "fixture-yala-too-early",
            },
            {
                "available_date": "2023-12-15",
                "observation_date": "2023-12-01",
                "season": "Maha 2023/24",
                "crop": "carrot",
                "district": "NATIONAL",
                "scope_level": "national",
                "target_ha": 80,
                "achieved_ha": 40,
                "source_report": "fixture-maha",
            },
        ]
    )
    audited = audit_progress(pd.concat([reports, extra], ignore_index=True))
    asof = attach_national_asof(_issue("2024-01-08"), audited)
    row = asof.iloc[0]
    if abs(float(row["cultivation_progress_ratio_real"]) - 0.5) > 1e-9:
        raise AssertionError(row["cultivation_progress_ratio_real"])
    if str(row["cultivation_source_report"]) != "fixture-maha":
        raise AssertionError(row["cultivation_source_report"])
    if pd.Timestamp(row["cultivation_report_date"]) != pd.Timestamp("2023-12-15"):
        raise AssertionError("the Yala report was used on a Maha Monday")


if __name__ == "__main__":
    test_future_report_ignored_and_ratio_above_one_kept()
    test_wrong_season_is_not_attached()
    print("national as-of checks passed")
