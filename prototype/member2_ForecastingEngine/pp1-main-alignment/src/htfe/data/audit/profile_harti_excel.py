"""Profile the HARTI price workbook and its tidy long outputs.

Run the ingester first:
  python -m htfe.data.ingest.harti_excel_prices
then (from the ``prototype`` folder):
  python -m htfe.data.audit.profile_harti_excel

Writes:
  PROCESSED_DIR/harti_coverage_by_series.csv
  PROCESSED_DIR/harti_weekly_completeness_matrix.csv
  AUDIT_DIR/harti_excel_profile.md
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from htfe.config import AUDIT_DIR, LOCAL_EXCEL, PROCESSED_DIR, processed_dir

WINDOW = (2016, 2025)


def _longest_gap(present: np.ndarray) -> int:
    best = run = 0
    for p in present:
        run = 0 if p else run + 1
        best = max(best, run)
    return best


def series_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_periods = 52 if df["frequency"].iloc[0] == "weekly" else 12
    grid_years = range(WINDOW[0], WINDOW[1] + 1)
    for (pt, loc, com), g in df.groupby(["price_type", "location", "commodity"]):
        valid = g[g["price_lkr_per_kg"].notna()]
        in_win = valid[valid["year"].between(*WINDOW) & (valid["period"] <= n_periods)]
        have = set(zip(in_win["year"], in_win["period"]))
        present = np.array([(y, p) in have for y in grid_years for p in range(1, n_periods + 1)])
        first = valid.loc[valid["period_start"].idxmin(), "period_start"] if len(valid) else None
        last = valid.loc[valid["period_start"].idxmax(), "period_start"] if len(valid) else None
        rows.append(
            {
                "price_type": pt,
                "frequency": g["frequency"].iloc[0],
                "location": loc,
                "district": g["district"].iloc[0],
                "commodity": com,
                "first_period": first,
                "last_period": last,
                "n_obs_all_years": len(valid),
                "n_obs_2016_2025": int(present.sum()),
                "n_expected_2016_2025": len(present),
                "completeness_2016_2025": round(present.mean(), 4),
                "longest_gap_periods_2016_2025": _longest_gap(present),
                "min": round(valid["price_lkr_per_kg"].min(), 2) if len(valid) else None,
                "median": round(valid["price_lkr_per_kg"].median(), 2) if len(valid) else None,
                "max": round(valid["price_lkr_per_kg"].max(), 2) if len(valid) else None,
                "cv": round(valid["price_lkr_per_kg"].std() / valid["price_lkr_per_kg"].mean(), 3) if len(valid) > 1 else None,
                "n_flagged": int((g["quality_flag"] != "ok").sum()),
            }
        )
    return pd.DataFrame(rows)


def week_alignment_test(weekly: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the published monthly retail means from the weekly sheet under
    competing definitions of W1, assigning each week to the month of its start
    day. The definition that reproduces the monthly sheet exactly is the one
    HARTI used."""
    w = weekly[(weekly["price_type"] == "retail") & weekly["price_lkr_per_kg"].notna()].copy()
    m = monthly[(monthly["price_type"] == "retail") & monthly["price_lkr_per_kg"].notna()]
    m = m.set_index(["location", "commodity", "year", "period"])["price_lkr_per_kg"]
    year, week = w["year"].astype(int), w["period"].astype(int)
    jan1 = pd.to_datetime(year.astype(str) + "-01-01")
    candidates = {
        "day_of_year_blocks (Jan 1 + 7(w-1))": jan1 + pd.to_timedelta((week - 1) * 7, unit="D"),
        "monday_on_or_before_jan1": jan1 - pd.to_timedelta(jan1.dt.weekday, unit="D") + pd.to_timedelta((week - 1) * 7, unit="D"),
        "iso_week_monday": pd.Series(
            [pd.Timestamp.fromisocalendar(int(y), int(min(k, 52)), 1) for y, k in zip(year, week)], index=w.index
        ),
        "monday_on_or_after_jan1 (adopted)": jan1 + pd.to_timedelta((7 - jan1.dt.weekday) % 7, unit="D")
        + pd.to_timedelta((week - 1) * 7, unit="D"),
    }
    out = []
    for name, start in candidates.items():
        tmp = w.assign(my=start.dt.year, mm=start.dt.month)
        rebuilt = tmp.groupby(["location", "commodity", "my", "mm"])["price_lkr_per_kg"].mean()
        rebuilt.index.names = ["location", "commodity", "year", "period"]
        joined = pd.concat([rebuilt.rename("from_weekly"), m.rename("published")], axis=1, join="inner")
        ape = (joined["from_weekly"] - joined["published"]).abs() / joined["published"]
        exact = (ape < 0.001)
        row = {"w1_definition": name, "n_pairs": len(joined), "exact_match_share": round(exact.mean(), 3),
               "median_ape": round(ape.median(), 4)}
        row.update({str(y): round(v, 2) for y, v in exact.groupby(level="year").mean().items()})
        out.append(row)
    return pd.DataFrame(out)


def workbook_facts() -> pd.DataFrame:
    rows = []
    for sheet, raw in pd.read_excel(LOCAL_EXCEL, sheet_name=None, header=None).items():
        title = next((v for v in raw.iloc[:4, 0].tolist() if isinstance(v, str)), "")
        rows.append({"sheet": sheet, "raw_rows": raw.shape[0], "raw_cols": raw.shape[1], "title_cell": title,
                     "header_row_excel": 5, "footer": str(raw.iloc[-1, 0]) if isinstance(raw.iloc[-1, 0], str) else ""})
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        vals = []
        for v in r.tolist():
            if isinstance(v, float):
                vals.append("" if np.isnan(v) else format(v, floatfmt))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    processed_dir()
    weekly = pd.read_csv(PROCESSED_DIR / "harti_prices_weekly_long.csv", low_memory=False)
    monthly = pd.read_csv(PROCESSED_DIR / "harti_prices_monthly_long.csv", low_memory=False)

    cov = pd.concat(
        [series_coverage(weekly[weekly["price_type"] == t]) for t in ("retail", "wholesale")]
        + [series_coverage(monthly[monthly["price_type"] == t]) for t in ("retail", "wholesale")],
        ignore_index=True,
    )
    cov.to_csv(PROCESSED_DIR / "harti_coverage_by_series.csv", index=False)

    wk = cov[cov["frequency"] == "weekly"]
    matrix = wk.pivot_table(index=["price_type", "commodity"], columns="location",
                            values="completeness_2016_2025", aggfunc="first")
    matrix.to_csv(PROCESSED_DIR / "harti_weekly_completeness_matrix.csv")

    align = week_alignment_test(weekly, monthly)

    sections = ["# HARTI `Vegetable Prices.xlsx` profile (auto-generated)", ""]
    sections += ["## Workbook", _md(workbook_facts()), ""]
    for name, df in (("weekly", weekly), ("monthly", monthly)):
        sections += [f"## {name.title()} long table", f"- rows: {len(df):,}",
                     f"- years: {df['year'].min()}–{df['year'].max()}",
                     f"- locations: {df['location'].nunique()} | commodities: {df['commodity'].nunique()}",
                     "- quality flags: " + ", ".join(f"{k}={v:,}" for k, v in df["quality_flag"].value_counts().items()), ""]
        per_year = df[df["price_lkr_per_kg"].notna()].pivot_table(index="year", columns="price_type",
                                                                  values="price_lkr_per_kg", aggfunc="count")
        sections += ["Valid observations per year:", _md(per_year.reset_index(), ".0f"), ""]
    sections += ["## Week-1 definition test (weekly sheet rebuilt into months vs published monthly sheet, retail)",
                 "Share of (location, commodity, month) values reproduced exactly (<0.1%), overall and per year.",
                 _md(align), ""]

    ratio = (
        weekly[weekly["price_lkr_per_kg"].notna()]
        .pivot_table(index=["location", "commodity", "year", "period"], columns="price_type", values="price_lkr_per_kg")
        .dropna()
    )
    ratio["ws_over_rt"] = ratio["wholesale"] / ratio["retail"]
    margin = ratio.groupby(level="commodity")["ws_over_rt"].describe()[["count", "25%", "50%", "75%"]].round(3)
    sections += ["## Wholesale / retail ratio at the same location and week", _md(margin.reset_index()), ""]

    for pt in ("retail", "wholesale"):
        sub = wk[wk["price_type"] == pt]
        by_com = sub.groupby("commodity")["completeness_2016_2025"].agg(["count", "median", "max"]).round(3)
        by_loc = sub.groupby("location")["completeness_2016_2025"].agg(["count", "median", "max"]).round(3)
        sections += [f"## Weekly {pt}: completeness 2016–2025 by commodity (across locations)", _md(by_com.reset_index()), "",
                     f"## Weekly {pt}: completeness 2016–2025 by location (across commodities)", _md(by_loc.reset_index()), ""]
    top = wk[wk["completeness_2016_2025"] >= 0.9].sort_values(["price_type", "commodity", "completeness_2016_2025"],
                                                              ascending=[True, True, False])
    sections += [f"## Weekly series with ≥90% completeness 2016–2025: {len(top)}",
                 _md(top.groupby(["price_type", "commodity"]).size().rename("n_locations").reset_index()), ""]

    report = AUDIT_DIR / "harti_excel_profile.md"
    report.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
