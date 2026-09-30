"""Progress-presentation figures from the real panel and scored forecasts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from training.config import EVAL_MARKETS


def write_figures(
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    importance: dict[str, float],
    origin_map: dict,
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    eval_panel = panel[panel["market"].isin(EVAL_MARKETS)]
    _coverage(eval_panel, destination / "fig01_coverage_heatmap.png")
    _timeline(eval_panel, destination / "fig02_dambulla_carrot_timeline.png")
    _missingness(eval_panel, destination / "fig03_missingness_calendar.png")
    _week_rule(destination / "fig04_harti_week_rule.png")
    _mape(metrics, destination / "fig05_mape_by_horizon.png")
    _splits(destination / "fig06_split_calendar.png")
    _table(metrics, destination / "fig07_model_comparison_table.png")
    _origins(origin_map, destination / "fig08_origin_map.png")
    _importance(importance, destination / "fig09_importance_or_acf.png")
    _limits(destination / "fig10_limitations.png")


def _coverage(panel: pd.DataFrame, path: Path) -> None:
    table = panel.groupby(["crop", "market"])["is_observed"].mean().unstack("market")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    image = ax.imshow(table.to_numpy(dtype=float), vmin=0.8, vmax=1, cmap="YlGn")
    ax.set_xticks(range(len(table.columns)), list(table.columns), rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)), list(table.index))
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table.iloc[i, j]:.0%}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.03)
    ax.set_title("Observed wholesale weeks, 2016–2025")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _timeline(panel: pd.DataFrame, path: Path) -> None:
    series = panel[(panel["crop"] == "carrot") & (panel["market"] == "Dambulla")].sort_values("week_start")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(series["week_start"], series["price"], color="#0f6b4c", lw=1)
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), color="#f4a261", alpha=0.25, label="2022 level shift")
    ax.axvspan(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31"), color="#e76f51", alpha=0.3, label="2024 Q1 spike")
    ax.set_ylabel("LKR/kg")
    ax.set_title("Dambulla carrot wholesale")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _missingness(panel: pd.DataFrame, path: Path) -> None:
    series = panel[(panel["crop"] == "carrot") & (panel["market"] == "Dambulla")]
    pivot = series.assign(missing=~series["is_observed"]).pivot_table(
        index="year", columns="week", values="missing", aggfunc="max"
    )
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot.index)), list(pivot.index))
    ax.set_xlabel("HARTI week")
    ax.set_title("Dambulla carrot missing weeks (dark = unpublished)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _week_rule(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "HARTI week 1 = first Monday on or after 1 January\n"
        "Rebuilding the published monthly sheet from these weeks matches about 97%.",
        ha="center",
        va="center",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _mape(metrics: pd.DataFrame, path: Path) -> None:
    test = metrics[metrics["slice"].eq("test")]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for model, group in test.groupby("model"):
        group = group.sort_values("horizon")
        ax.plot(group["horizon"], group["mape"], marker="o", label=model)
    ax.set_xticks([1, 2, 4, 8, 12])
    ax.set_xlabel("Horizon (weeks)")
    ax.set_ylabel("MAPE %")
    ax.set_title("2024–2025 test MAPE by horizon")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _splits(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 1.8))
    spans = [("Train ≤2022", 2016, 2023, "#2a9d8f"), ("Calibrate 2023", 2023, 2024, "#e9c46a"), ("Test 2024–25", 2024, 2026, "#e76f51")]
    for label, start, end, color in spans:
        ax.barh(0, end - start, left=start, color=color, label=label)
    ax.set_yticks([])
    ax.set_xlim(2016, 2026)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.45))
    ax.set_title("One calendar split for every series")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _table(metrics: pd.DataFrame, path: Path) -> None:
    test = metrics[metrics["slice"].eq("test") & metrics["horizon"].eq(1)][
        ["model", "n_scored", "mae", "rmse", "mape"]
    ]
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.axis("off")
    shown = ax.table(cellText=test.round(2).to_numpy(), colLabels=list(test.columns), loc="center")
    shown.auto_set_font_size(False)
    shown.set_fontsize(8)
    ax.set_title("1-week-ahead test metrics", pad=16)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _origins(origin_map: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.axis("off")
    lines = ["Origin districts behind scored markets"]
    for crop in ("carrot", "leeks", "tomato"):
        for market in EVAL_MARKETS:
            spec = origin_map.get(crop, {}).get(market, {})
            districts = ", ".join(spec.get("supply_districts") or [])
            lines.append(f"{crop} · {market} ← {districts}")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _importance(importance: dict[str, float], path: Path) -> None:
    items = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    if items:
        names = [name for name, _ in items][::-1]
        values = [value for _, value in items][::-1]
        ax.barh(names, values, color="#0f6b4c")
    ax.set_title("LightGBM gain, summed across horizons (q=0.50)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _limits(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axis("off")
    ax.text(
        0.02,
        0.9,
        "\n".join(
            [
                "Not in this run",
                "• Potato, red onion, and local big onion — missing or too incomplete",
                "• No live HARTI cultivation commitments (historical Census proxy only)",
                "• Intervals are raw 5/95% quantiles, not conformal yet",
                "• Central Bank daily prices are not the training grain",
                "• OpenWeather History was not used (paid); weather is Open-Meteo",
            ]
        ),
        va="top",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
