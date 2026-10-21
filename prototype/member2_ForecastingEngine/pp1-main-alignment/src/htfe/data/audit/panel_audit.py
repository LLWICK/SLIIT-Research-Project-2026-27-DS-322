"""Summarise the analysis-ready panel.

Usage (from the prototype folder):
  python -m htfe.data.audit.panel_audit
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from htfe.data.build.scope import EVAL_MARKETS
from htfe.config import AUDIT_DIR, processed_dir


def summarise(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (crop, market), group in panel.groupby(["crop", "market"], sort=True):
        observed = group["is_observed"]
        weather_ok = group["weather_tmean_c"].notna() if "weather_tmean_c" in group else pd.Series(False, index=group.index)
        rows.append(
            {
                "crop": crop,
                "market": market,
                "role": "eval" if market in EVAL_MARKETS else "train_pool",
                "n_rows": int(len(group)),
                "n_observed": int(observed.sum()),
                "pct_observed": round(float(observed.mean()), 3),
                "week_start_min": str(pd.to_datetime(group["week_start"]).min().date()),
                "week_start_max": str(pd.to_datetime(group["week_start"]).max().date()),
                "n_disruption_covid": int((group["disruption_flag"] == "covid").sum()),
                "n_disruption_2017": int((group["disruption_flag"] == "collection_gap_2017").sum()),
                "n_imputed": int(group["is_imputed"].sum()),
                "weather_join_pct_observed": round(float(weather_ok[observed].mean()) if observed.any() else 0.0, 3),
            }
        )
    return pd.DataFrame(rows)


def write_report(panel: pd.DataFrame, summary: pd.DataFrame) -> Path:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = AUDIT_DIR / "panel_series_summary.csv"
    md_path = AUDIT_DIR / "panel_audit.md"
    summary.to_csv(csv_path, index=False)
    eval_rows = summary[summary["role"] == "eval"]
    lines = [
        "# Analysis-ready panel audit",
        "",
        f"Rows: **{len(panel):,}**. Observed prices: **{int(panel['is_observed'].sum()):,}**.",
        f"Crops: {', '.join(sorted(panel['crop'].unique()))}.",
        "",
        "Decision-plan wholesale completeness for carrot / leeks / tomato on Dambulla, Colombo, Meegoda and Nuwara Eliya is about 0.91–0.94 of a 520-week 2016–2025 grid. The percentages below use the HARTI weeks present in this extract (including gaps), so they should sit in that range.",
        "",
        "## Series",
        "",
        summary.to_markdown(index=False),
        "",
        "## Eval-market check",
        "",
        f"Minimum observed share on eval markets: **{eval_rows['pct_observed'].min():.3f}**.",
        f"Minimum weather join on observed eval rows: **{eval_rows['weather_join_pct_observed'].min():.3f}**.",
        "",
        "Long gaps (early 2017, COVID 2020–21) stay unfilled. `price` is null wherever HARTI did not publish a usable value. `price_feature` may fill isolated 1–2 week holes for lag features only.",
        "",
    ]
    # Spot-check the as-of rule on Yala 2024 (weeks 14–39).
    yala = panel[(panel["year"] == 2024) & (panel["week"].between(14, 39))]
    if not yala.empty:
        labels = sorted(yala["supply_season_label"].dropna().unique())
        lines.append(f"Yala 2024 supply labels in use: {', '.join(labels)} (expected prior Maha, not Yala_2024).")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    panel = pd.read_parquet(processed_dir() / "latest" / "analysis_ready_panel.parquet")
    summary = summarise(panel)
    path = write_report(panel, summary)
    print(f"wrote {path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
