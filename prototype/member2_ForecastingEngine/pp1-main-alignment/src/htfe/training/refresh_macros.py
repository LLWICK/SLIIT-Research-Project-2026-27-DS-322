"""Rebuild macro availability and refresh training-table macro columns.

Usage from prototype/member2_ForecastingEngine:
  python -u -m training.refresh_macros
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import htfe.training.config  # noqa: F401  # puts prototype/ on sys.path
from htfe.data.build.fetch_published_macros import (
    INFLATION_AVAILABLE_DATE_RULE,
    INFLATION_LAG_DAYS,
    ensure_published_macros,
)
from htfe.data.build.macros import MACRO_COLUMNS, asof_macros, write_macro_note
from htfe.config import processed_dir

from htfe.config import OUTPUTS
REPO = OUTPUTS
AUDIT_PATH = OUTPUTS / "notes" / "Macro-Availability-Audit-IT23415836.md"
RENAME = {
    "diesel_lkr_per_litre": "diesel_price_lkr_litre",
    "usd_lkr": "usd_lkr_rate",
    "inflation_yoy": "inflation_rate_pct",
}


def _value_on(series: pd.DataFrame, day: str) -> float:
    hit = series.loc[pd.to_datetime(series["reference_month_end"]).eq(pd.Timestamp(day)), "inflation_yoy"]
    if hit.empty:
        raise RuntimeError(f"no CCPI year-on-year for reference month {day}")
    return float(hit.iloc[0])


def _example_block(published: pd.DataFrame, table: pd.DataFrame) -> list[str]:
    inflation = published.dropna(subset=["inflation_yoy", "reference_month_end"]).copy()
    inflation["reference_month_end"] = pd.to_datetime(inflation["reference_month_end"])
    inflation["available_date"] = pd.to_datetime(inflation["available_date"])
    january = _value_on(inflation, "2024-01-31")
    february = _value_on(inflation, "2024-02-29")
    march = _value_on(inflation, "2024-03-31")
    checks = {
        "2024-03-04": january,
        "2024-03-18": january,
        "2024-03-25": february,
    }
    lines = [
        "A forecast issued on Monday 2024-03-04 is in the first three weeks of March.",
        "March 2024 CCPI has reference month-end 2024-03-31 and available date 2024-04-21, so that issue cannot see it.",
        "February 2024 CCPI has reference month-end 2024-02-29 and available date 2024-03-21, so that issue cannot see it either.",
        f"January 2024 CCPI, value {january:.1f}, is available on 2024-02-21, so the 2024-03-04 forecast can see it.",
        f"The same January value is still the latest print on 2024-03-18. On 2024-03-25 the February value {february:.1f} is available. The March value {march:.1f} stays hidden until 2024-04-21.",
    ]
    for day, expected in checks.items():
        stamp = pd.Timestamp(day)
        rows = table.loc[pd.to_datetime(table["forecast_date"]).eq(stamp), "inflation_rate_pct"]
        if rows.empty:
            raise AssertionError(f"training table has no forecast issued on {day}")
        got = float(rows.iloc[0])
        if abs(got - expected) > 1e-6:
            raise AssertionError(f"inflation on {day} is {got}, expected {expected}")
        if abs(got - march) < 1e-6:
            raise AssertionError(f"forecast on {day} can see March 2024 CCPI")
    early = table.loc[pd.to_datetime(table["forecast_date"]).eq(pd.Timestamp("2024-03-04")), "inflation_rate_pct"]
    if bool((early - february).abs().le(1e-6).any()):
        raise AssertionError("a 2024-03-04 forecast can see February 2024 CCPI")
    lines.append("The rebuilt training table matches these three issue dates.")
    return lines


def refresh_training_table() -> dict:
    from htfe.config import built_file
    path = built_file("training_table.parquet")
    table = pd.read_parquet(path)
    table["_row"] = range(len(table))
    merged, meta = asof_macros(table, "forecast_date")
    write_macro_note(meta)
    if meta.get("status") != "joined" or len(meta.get("columns") or []) < len(MACRO_COLUMNS):
        raise RuntimeError(f"macro join failed: {meta}")
    merged = merged.sort_values("_row")
    for source, target in RENAME.items():
        merged[target] = pd.to_numeric(merged[source], errors="coerce")
    merged = merged.drop(columns=[name for name in list(MACRO_COLUMNS) + ["_row"] if name in merged.columns])
    merged["diesel_missing_flag"] = merged["diesel_price_lkr_litre"].isna()
    merged["usd_missing_flag"] = merged["usd_lkr_rate"].isna()
    merged["inflation_missing_flag"] = merged["inflation_rate_pct"].isna()
    if not bool((pd.to_datetime(merged["forecast_date"]) == pd.to_datetime(table["forecast_date"])).all()):
        raise AssertionError("macro refresh reordered forecast dates")
    merged.to_parquet(path, index=False)
    shares = {target: round(float(merged[target].notna().mean()), 4) for target in RENAME.values()}
    return {"path": str(path), "rows": int(len(merged)), "non_null_share": shares, "table": merged}


def write_audit(published: pd.DataFrame, table: pd.DataFrame, shares: dict) -> None:
    inflation = published.dropna(subset=["inflation_yoy"]).copy()
    example = _example_block(published, table)
    text = "\n".join(
        [
            "# Macro availability audit",
            "",
            "Each series is joined as-of: `available_date` must be on or before `forecast_issue_date`. No macro value was invented.",
            "",
            "| Series | Column | Available-date rule |",
            "| --- | --- | --- |",
            "| Diesel | `diesel_price_lkr_litre` | CPC Lanka Auto Diesel effective/revision date, the date printed on the Ceypetco historical-price row. |",
            "| USD/LKR | `usd_lkr_rate` | CBSL indicative-rate observation date from the Frankfurter CBSL series. |",
            f"| Inflation | `inflation_rate_pct` | `{INFLATION_AVAILABLE_DATE_RULE}`. Reference month-end plus {INFLATION_LAG_DAYS} days. |",
            "",
            "The DCS monthly CCPI page and the movements PDFs do not list a historical release calendar. A 20-second fetch of the DCS monthly CCPI page returned the department site shell, not release dates. The reference month-end is therefore not used as the available date.",
            "",
            "## Inflation example",
            "",
            *example,
            "",
            "## Coverage after the lag",
            "",
            f"Training-table non-null shares: {shares}.",
            f"Inflation rows in `macros_published.csv`: {int(len(inflation))}.",
            "",
            "CCPI base 2013=100 is used through January 2023, and base 2021=100 from February 2023. The index was not rescaled. That base change is a property of the published series, not a model failure.",
            "",
        ]
    )
    AUDIT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    rebuilt = ensure_published_macros(refresh=True)
    print(rebuilt, flush=True)
    published = pd.read_csv(rebuilt["path"], parse_dates=["available_date", "reference_month_end"])
    refreshed = refresh_training_table()
    table = refreshed.pop("table")
    print(refreshed, flush=True)
    write_audit(published, table, refreshed["non_null_share"])
    print(f"wrote {AUDIT_PATH}", flush=True)


if __name__ == "__main__":
    main()
