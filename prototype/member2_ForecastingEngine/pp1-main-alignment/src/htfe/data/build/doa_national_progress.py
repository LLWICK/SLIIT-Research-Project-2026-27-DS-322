"""National exact-season as-of signal from doa_cultivation_progress.csv.

Scored Experiment C uses national rows only. District rows stay in the audit.
A report is attached only when the crop matches, the official source season
matches the forecast Monday's operational season, and available_date is on or
before that Monday. Ratios are achieved hectares / target hectares. They are
not clipped at 1. A non-positive target is kept in the audit and is not turned
into 1. No hectares are invented.

This module does not read cultivation_monthly.csv and does not call the
district-month loader.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from htfe.data.build.cultivation_progress import doa_progress_path

REQUIRED = (
    "available_date",
    "observation_date",
    "season",
    "crop",
    "target_ha",
    "achieved_ha",
    "source_report",
)
CROPS = ("carrot", "leeks", "tomato")
ALLOWED_ISSUE_YEARS = frozenset({2018, 2019, 2022, 2023, 2024, 2025})
EXCLUDED_ISSUE_YEARS = frozenset({2016, 2017, 2020, 2021})
PARSER_EXAMPLES = {
    "Yala 2018": "Yala 2018",
    "Maha 2024/25": "Maha 2024",
    "Maha 2024/2025": "Maha 2024",
}


def norm_crop(value) -> str:
    key = str(value).strip().lower()
    return {
        "carrot": "carrot",
        "carrots": "carrot",
        "leek": "leeks",
        "leeks": "leeks",
        "tomato": "tomato",
        "tomatoes": "tomato",
    }.get(key, key)


def season_family(value) -> str | float:
    text = str(value).casefold()
    if "maha" in text:
        return "Maha"
    if "yala" in text:
        return "Yala"
    return np.nan


def official_season_id(value) -> str | float:
    """Parse the source label. Maha 2024/25 stays Maha 2024. The date is not used."""
    text = str(value).strip()
    family = season_family(text)
    if pd.isna(family):
        return np.nan
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    if match is None:
        return np.nan
    return f"{family} {int(match.group(1))}"


def operational_issue_season_id(value) -> str:
    """Operational season of a forecast Monday. Oct–Mar is Maha, Apr–Sep is Yala."""
    stamp = pd.Timestamp(value)
    month = int(stamp.month)
    year = int(stamp.year)
    if month in (10, 11, 12):
        return f"Maha {year}"
    if month in (1, 2, 3):
        return f"Maha {year - 1}"
    return f"Yala {year}"


def assert_season_parser() -> None:
    for raw_label, expected in PARSER_EXAMPLES.items():
        parsed = official_season_id(raw_label)
        if parsed != expected:
            raise AssertionError(f"{raw_label!r} parsed as {parsed!r}, expected {expected!r}")


def load_doa(path: Path | None = None) -> pd.DataFrame:
    found = path or doa_progress_path()
    if found is None:
        raise FileNotFoundError(
            "doa_cultivation_progress.csv was not found in data/member1 or the processed data folder."
        )
    frame = pd.read_csv(found)
    frame.attrs["source_path"] = str(found)
    return frame


def audit_progress(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the extract and compute ratios. Invalid targets stay in the frame."""
    assert_season_parser()
    progress = frame.copy()
    progress.columns = [str(column).strip() for column in progress.columns]
    missing = [name for name in REQUIRED if name not in progress.columns]
    if missing:
        raise ValueError(f"doa_cultivation_progress.csv is missing columns: {missing}")

    for column, default in {
        "district": "NATIONAL",
        "source_page": "",
        "scope_level": "",
        "notes": "",
    }.items():
        if column not in progress.columns:
            progress[column] = default

    for column in ("available_date", "observation_date"):
        progress[column] = pd.to_datetime(progress[column], format="mixed", errors="raise").dt.normalize()

    progress["crop"] = progress["crop"].map(norm_crop)
    progress = progress[progress["crop"].isin(CROPS)].copy()
    if progress.empty:
        raise ValueError("No carrot, leeks, or tomato rows remain in the DOA extract.")

    progress["district"] = progress["district"].fillna("NATIONAL").astype(str).str.strip()
    progress["scope_level"] = progress["scope_level"].fillna("").astype(str).str.strip().str.lower()
    blank_scope = progress["scope_level"].eq("")
    progress.loc[blank_scope, "scope_level"] = np.where(
        progress.loc[blank_scope, "district"].str.casefold().eq("national"),
        "national",
        "district",
    )

    for column in ("target_ha", "achieved_ha"):
        progress[column] = pd.to_numeric(progress[column], errors="coerce")
    bad_numeric = progress["target_ha"].isna() | progress["achieved_ha"].isna()
    if bool(bad_numeric.any()):
        raise ValueError(f"{int(bad_numeric.sum())} rows have non-numeric target_ha or achieved_ha.")
    if bool(progress["achieved_ha"].lt(0).any()):
        raise ValueError("achieved_ha cannot be negative.")

    # A non-positive target is real source data. It is not replaced with 1.
    progress["valid_progress_row"] = (
        progress["target_ha"].notna()
        & progress["achieved_ha"].notna()
        & progress["target_ha"].gt(0)
        & progress["achieved_ha"].ge(0)
    )
    progress["cultivation_progress_ratio"] = np.where(
        progress["valid_progress_row"],
        progress["achieved_ha"] / progress["target_ha"],
        np.nan,
    )
    progress["target_exceeded"] = np.where(
        progress["cultivation_progress_ratio"].notna(),
        progress["cultivation_progress_ratio"].gt(1.0).astype(int),
        np.nan,
    )
    progress["season_family"] = progress["season"].map(season_family)
    progress["official_season_id"] = progress["season"].map(official_season_id)
    progress["exact_season_id"] = progress["official_season_id"].astype("string")
    progress["observation_date_operational_season_id"] = progress["observation_date"].map(operational_issue_season_id)
    progress["source_date_season_mismatch"] = (
        progress["exact_season_id"].notna()
        & progress["observation_date_operational_season_id"].notna()
        & progress["exact_season_id"].ne(progress["observation_date_operational_season_id"])
    )
    unknown = progress["official_season_id"].isna()
    if bool(unknown.any()):
        raise ValueError(
            "Experiment C needs an official Maha or Yala label with a four-digit start year. "
            f"{int(unknown.sum())} rows do not have one."
        )
    return progress.reset_index(drop=True)


def national_scoring_rows(audited: pd.DataFrame) -> pd.DataFrame:
    """Valid national rows only. District rows are not returned."""
    national = audited[
        (audited["scope_level"].eq("national") | audited["district"].astype(str).str.casefold().eq("national"))
        & audited["valid_progress_row"].astype(bool)
        & audited["cultivation_progress_ratio"].notna()
        & audited["exact_season_id"].notna()
    ].copy()
    if national.empty:
        return national
    key = ["crop", "available_date", "exact_season_id"]
    grouped = (
        national.groupby(key, dropna=False)
        .agg(
            row_count=("crop", "size"),
            target_values=("target_ha", "nunique"),
            achieved_values=("achieved_ha", "nunique"),
        )
        .reset_index()
    )
    conflicts = grouped[
        grouped["row_count"].gt(1)
        & (grouped["target_values"].gt(1) | grouped["achieved_values"].gt(1))
    ]
    if len(conflicts):
        raise ValueError("Conflicting national rows exist for the same crop, report date, and exact season.")
    return national.sort_values(key, kind="mergesort").drop_duplicates(key, keep="first").reset_index(drop=True)


def _signal_from_join(joined: pd.DataFrame) -> pd.DataFrame:
    """Copy attached hectares into the as-of columns. Do not clip the ratio."""
    ratio = pd.to_numeric(joined["cultivation_progress_ratio"], errors="coerce")
    target = pd.to_numeric(joined["target_ha"], errors="coerce")
    achieved = pd.to_numeric(joined["achieved_ha"], errors="coerce")
    comparable = ratio.notna() & target.gt(0)
    if bool(comparable.any()):
        direct = achieved / target
        gap = (ratio[comparable] - direct[comparable]).abs()
        if float(gap.max()) > 1e-9:
            raise AssertionError("attached ratio does not equal achieved_ha / target_ha")
    if bool((ratio.notna() & ~target.gt(0)).any()):
        raise AssertionError("a non-positive target was used in a progress ratio")
    future = joined["available_date"].notna() & joined["available_date"].gt(joined["forecast_date"])
    if bool(future.any()):
        raise AssertionError("a report dated after the forecast Monday was attached")
    out = joined.copy()
    out["cultivation_report_date"] = pd.to_datetime(out["available_date"])
    out["cultivation_observation_date"] = pd.to_datetime(out["observation_date"])
    out["cultivation_exact_season_id"] = out["exact_season_id"].astype("string")
    out["cultivation_target_ha_real"] = target
    out["cultivation_achieved_ha_real"] = achieved
    out["cultivation_progress_ratio_real"] = ratio
    exceeded = pd.Series(np.nan, index=out.index, dtype="float")
    known = ratio.notna()
    exceeded.loc[known] = ratio.loc[known].gt(1.0).astype(float)
    out["cultivation_target_exceeded_real"] = exceeded
    age = (out["forecast_date"] - out["cultivation_report_date"]).dt.days
    out["cultivation_report_age_days"] = age.where(known)
    out["cultivation_source_scope"] = np.where(known, "national", "no_report_available")
    out["cultivation_source_report"] = out["source_report"].where(known)
    out["c_issue_year_allowed"] = True
    return out


def _empty_signal(issue: pd.DataFrame, scope: str) -> pd.DataFrame:
    out = issue.copy()
    out["cultivation_report_date"] = pd.NaT
    out["cultivation_observation_date"] = pd.NaT
    out["cultivation_exact_season_id"] = out["exact_season_id"].astype("string")
    out["cultivation_target_ha_real"] = np.nan
    out["cultivation_achieved_ha_real"] = np.nan
    out["cultivation_progress_ratio_real"] = np.nan
    out["cultivation_target_exceeded_real"] = np.nan
    out["cultivation_report_age_days"] = np.nan
    out["cultivation_source_scope"] = scope
    out["cultivation_source_report"] = pd.NA
    out["c_issue_year_allowed"] = out["issue_year"].isin(ALLOWED_ISSUE_YEARS)
    return out


def attach_national_asof(issue: pd.DataFrame, audited: pd.DataFrame) -> pd.DataFrame:
    """Latest same-crop, same-season national report with available_date <= forecast Monday.

    ``issue`` needs crop_id, market_id, and forecast_date. One row per
    crop-market-Monday is returned. Horizons are joined later.
    """
    work = issue[["crop_id", "market_id", "forecast_date"]].drop_duplicates().copy()
    work["forecast_date"] = pd.to_datetime(work["forecast_date"], format="mixed").dt.normalize().astype("datetime64[ns]")
    work["crop_id"] = work["crop_id"].map(norm_crop)
    work["market_id"] = work["market_id"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    work["crop"] = work["crop_id"]
    work["issue_year"] = work["forecast_date"].dt.year.astype(int)
    work["exact_season_id"] = work["forecast_date"].map(operational_issue_season_id)
    work["c_year_allowed"] = work["issue_year"].isin(ALLOWED_ISSUE_YEARS)

    blocked = _empty_signal(work.loc[~work["c_year_allowed"]].copy(), "excluded_issue_year")
    allowed = work.loc[work["c_year_allowed"]].copy()
    national = national_scoring_rows(audited)

    if allowed.empty or national.empty:
        matched = _empty_signal(allowed, "no_report_available")
    else:
        right = national[
            [
                "crop",
                "exact_season_id",
                "available_date",
                "observation_date",
                "target_ha",
                "achieved_ha",
                "cultivation_progress_ratio",
                "source_report",
            ]
        ].copy()
        right["exact_season_id"] = right["exact_season_id"].astype(str)
        right["crop"] = right["crop"].astype(str)
        right["available_date"] = pd.to_datetime(right["available_date"]).dt.normalize().astype("datetime64[ns]")
        allowed["exact_season_id"] = allowed["exact_season_id"].astype(str)
        allowed["crop"] = allowed["crop"].astype(str)
        # merge_asof checks that the date key itself is sorted. Group first so
        # each crop-season pair has its own ordered dates.
        grouped: list[pd.DataFrame] = []
        for (crop, season), left_group in allowed.groupby(["crop", "exact_season_id"], sort=False):
            left_group = left_group.sort_values("forecast_date", kind="mergesort")
            right_group = right[
                right["crop"].eq(crop) & right["exact_season_id"].eq(season)
            ].sort_values("available_date", kind="mergesort")
            if right_group.empty:
                grouped.append(_empty_signal(left_group, "no_report_available"))
                continue
            left_group = left_group.copy()
            right_group = right_group.copy()
            left_group["forecast_date"] = pd.to_datetime(left_group["forecast_date"]).astype("datetime64[ns]")
            right_group["available_date"] = pd.to_datetime(right_group["available_date"]).astype("datetime64[ns]")
            joined = pd.merge_asof(
                left_group,
                right_group.drop(columns=["crop", "exact_season_id"]),
                left_on="forecast_date",
                right_on="available_date",
                direction="backward",
                allow_exact_matches=True,
            )
            grouped.append(_signal_from_join(joined))
        matched = pd.concat(grouped, ignore_index=True) if grouped else _empty_signal(allowed, "no_report_available")

    columns = [
        "crop_id",
        "market_id",
        "forecast_date",
        "issue_year",
        "cultivation_report_date",
        "cultivation_observation_date",
        "cultivation_exact_season_id",
        "cultivation_target_ha_real",
        "cultivation_achieved_ha_real",
        "cultivation_progress_ratio_real",
        "cultivation_target_exceeded_real",
        "cultivation_report_age_days",
        "cultivation_source_scope",
        "cultivation_source_report",
        "c_issue_year_allowed",
    ]
    pieces = [piece for piece in (matched, blocked) if not piece.empty]
    if not pieces:
        return pd.DataFrame(columns=columns)
    asof = pd.concat(pieces, ignore_index=True)
    asof = asof[columns]
    leaked = asof["cultivation_report_date"].notna() & asof["cultivation_report_date"].gt(asof["forecast_date"])
    if bool(leaked.any()):
        raise AssertionError("a future-dated national report survived the as-of join")
    valid = asof["cultivation_progress_ratio_real"].notna()
    if bool(valid.any()):
        operational = asof.loc[valid, "forecast_date"].map(operational_issue_season_id)
        attached = asof.loc[valid, "cultivation_exact_season_id"].astype(str)
        if not bool(operational.astype(str).eq(attached).all()):
            raise AssertionError("a cultivation report was attached outside its exact season")
        excluded = asof["issue_year"].isin(EXCLUDED_ISSUE_YEARS) & valid
        if bool(excluded.any()):
            raise AssertionError("an excluded issue year received a cultivation value")
        above = asof.loc[valid & asof["cultivation_progress_ratio_real"].gt(1.0), "cultivation_progress_ratio_real"]
        if len(above) and bool(above.le(1.0).any()):
            raise AssertionError("a ratio above 1 was clipped")
    return asof.reset_index(drop=True)


def coverage_tables(asof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    yearly = (
        asof.groupby("issue_year", as_index=False)
        .agg(
            rows=("crop_id", "size"),
            rows_with_progress=("cultivation_progress_ratio_real", lambda series: int(series.notna().sum())),
        )
    )
    yearly["coverage_pct"] = (yearly["rows_with_progress"] / yearly["rows"] * 100).round(2)
    known = asof[asof["cultivation_progress_ratio_real"].notna()]
    if known.empty:
        season = pd.DataFrame(columns=["cultivation_exact_season_id", "rows_with_progress", "report_dates"])
    else:
        season = known.groupby("cultivation_exact_season_id", as_index=False).agg(
            rows_with_progress=("crop_id", "size"),
            report_dates=("cultivation_report_date", "nunique"),
        )
    return yearly, season


def audit_summary(audited: pd.DataFrame, asof: pd.DataFrame, source_path: str | None) -> dict:
    national_mask = audited["scope_level"].eq("national") | audited["district"].astype(str).str.casefold().eq("national")
    district = audited[audited["scope_level"].eq("district")]
    valid_national = national_scoring_rows(audited)
    ratios = pd.to_numeric(valid_national["cultivation_progress_ratio"], errors="coerce") if len(valid_national) else pd.Series(dtype=float)
    yearly, season = coverage_tables(asof)
    signal = int(asof["cultivation_progress_ratio_real"].notna().sum()) if len(asof) else 0
    return {
        "source_path": source_path,
        "spatial_scope_for_scoring": "national",
        "spatial_scope_note": (
            "National cultivation progress is not district supply progress. "
            "District rows stay in the audit and are not features."
        ),
        "district_rows_used_for_scoring": False,
        "ratio_rule": "achieved_ha / target_ha when target_ha > 0. Ratios above 1 are kept. target_ha <= 0 is not replaced with 1.",
        "asof_rule": "same crop, official source season equals the forecast Monday operational season, available_date <= forecast Monday",
        "report_age_cutoff_days": None,
        "allowed_issue_years": sorted(ALLOWED_ISSUE_YEARS),
        "excluded_issue_years": sorted(EXCLUDED_ISSUE_YEARS),
        "n_source_rows_after_crop_filter": int(len(audited)),
        "n_national_rows": int(national_mask.sum()),
        "n_district_rows": int(len(district)),
        "n_valid_national_rows": int(len(valid_national)),
        "n_nonpositive_target_rows": int(audited["target_ha"].le(0).sum()),
        "n_national_ratios_above_one": int(ratios.gt(1.0).sum()) if len(ratios) else 0,
        "n_source_date_season_mismatches": int(audited["source_date_season_mismatch"].sum()),
        "asof_rows": int(len(asof)),
        "asof_rows_with_national_signal": signal,
        "coverage_by_issue_year": yearly.to_dict(orient="records"),
        "coverage_by_exact_season": season.to_dict(orient="records"),
        "wired_through_district_month_loader": False,
        "renamed_to_cultivation_monthly": False,
    }
