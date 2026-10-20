"""Plots for the studio, drawn from this run's experiment files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from htfe.config import OUTPUTS, processed_dir

PLOT = OUTPUTS / "figures"
HORIZONS = (1, 2, 4, 8, 12)


def _save(fig: plt.Figure, name: str) -> Path:
    PLOT.mkdir(parents=True, exist_ok=True)
    path = PLOT / name
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)
    return path


def _price_plots() -> None:
    weekly = pd.read_csv(processed_dir() / "harti_prices_weekly_long.csv", low_memory=False)
    weekly["period_start"] = pd.to_datetime(weekly["period_start"])
    weekly["price_lkr_per_kg"] = pd.to_numeric(weekly["price_lkr_per_kg"], errors="coerce")
    score = ["Dambulla", "Colombo", "Meegoda", "Nuwara Eliya"]
    crops = ["carrot", "leeks", "tomato"]
    wholesale = weekly[
        weekly["price_type"].eq("wholesale")
        & weekly["commodity"].isin(crops)
        & weekly["location"].isin(score)
    ].copy()
    missing = (
        wholesale.assign(year=wholesale["period_start"].dt.year, missing=wholesale["price_lkr_per_kg"].isna())
        .groupby("year", as_index=False)["missing"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(missing["year"], missing["missing"], color="black")
    ax.set_title("Wholesale weekly missing share by year")
    ax.set_ylabel("share missing")
    _save(fig, "wholesale_missing_share_by_year.png")

    for crop in crops:
        part = wholesale[wholesale["commodity"].eq(crop) & wholesale["location"].eq("Dambulla")].sort_values("period_start")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(part["period_start"], part["price_lkr_per_kg"], color="black", lw=1)
        ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"), color="0.85", label="2020-2021")
        ax.set_title(f"Wholesale {crop}, Dambulla, LKR per kg")
        ax.set_ylabel("LKR/kg")
        ax.legend()
        _save(fig, f"price_{crop}_dambulla.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    carrot = wholesale[wholesale["commodity"].eq("carrot")]
    for ax, market in zip(axes.ravel(), score):
        part = carrot[carrot["location"].eq(market)].sort_values("period_start")
        ax.plot(part["period_start"], part["price_lkr_per_kg"], color="black", lw=1)
        ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"), color="0.85")
        ax.set_title(f"Wholesale carrot, {market}, LKR per kg")
    _save(fig, "price_carrot_four_markets.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for crop in crops:
        part = wholesale[wholesale["commodity"].eq(crop) & wholesale["location"].eq("Dambulla")].sort_values("period_start")
        observed = part.dropna(subset=["price_lkr_per_kg"])
        observed = observed.assign(week=observed["period_start"].dt.isocalendar().week.astype(int))
        profile = observed.groupby("week")["price_lkr_per_kg"].mean()
        ax.plot(profile.index, profile.values, label=crop)
    ax.set_title("Seasonal level: mean wholesale price by week of year, Dambulla")
    ax.set_xlabel("week of year")
    ax.set_ylabel("LKR/kg")
    ax.legend()
    _save(fig, "price_season_week_of_year.png")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for crop in crops:
        part = wholesale[wholesale["commodity"].eq(crop) & wholesale["location"].eq("Dambulla")].sort_values("period_start")
        change = part["price_lkr_per_kg"].pct_change() * 100
        ax.plot(part["period_start"], change, lw=0.8, label=crop)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Week-to-week wholesale price movement, Dambulla")
    ax.set_ylabel("percent change")
    ax.legend()
    _save(fig, "price_week_to_week_change.png")

    both = weekly[weekly["commodity"].eq("carrot") & weekly["location"].eq("Dambulla") & weekly["price_lkr_per_kg"].notna()]
    retail = both[both["price_type"].eq("retail")][["period_start", "price_lkr_per_kg"]].rename(columns={"price_lkr_per_kg": "retail"})
    whole = both[both["price_type"].eq("wholesale")][["period_start", "price_lkr_per_kg"]].rename(columns={"price_lkr_per_kg": "wholesale"})
    ratio = whole.merge(retail, on="period_start")
    if not ratio.empty:
        ratio["ratio"] = ratio["wholesale"] / ratio["retail"]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(ratio["period_start"], ratio["ratio"], color="black", lw=1)
        ax.axhline(1, color="red", lw=1)
        ax.set_title("Dambulla carrot, wholesale divided by retail")
        ax.set_ylabel("ratio")
        _save(fig, "carrot_dambulla_wholesale_over_retail.png")


def _mae_lines(frame: pd.DataFrame, model_column: str, value_column: str, title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, part in frame.groupby(model_column):
        part = part.sort_values("horizon")
        ax.plot(part["horizon"], part[value_column], marker="o", label=model)
    ax.set_title(title)
    ax.set_xlabel("horizon (weeks)")
    ax.set_ylabel("MAE (LKR/kg)")
    ax.set_xticks(list(HORIZONS))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    _save(fig, name)


def _experiment_plots() -> None:
    a_path = OUTPUTS / "experiment_a" / "metrics_by_model_horizon.csv"
    b_path = OUTPUTS / "experiment_b" / "metrics_by_model_horizon.csv"
    if a_path.is_file():
        metrics = pd.read_csv(a_path)
        test = metrics[metrics["slice"].eq("test")]
        _mae_lines(test, "model", "mae", "Experiment A test MAE by horizon, LKR/kg", "experiment_a_mae.png")
    if a_path.is_file() and b_path.is_file():
        compared = pd.read_csv(OUTPUTS / "experiment_b" / "ab_comparison.csv")
        fig, ax = plt.subplots(figsize=(8, 4))
        for model in ("lightgbm", "xgboost", "sarimax"):
            part = compared[compared["model"].eq(model)].sort_values("horizon")
            if part.empty:
                continue
            ax.plot(part["horizon"], part["mae_a"], marker="o", linestyle="--", label=f"{model} A")
            ax.plot(part["horizon"], part["mae_b"], marker="o", label=f"{model} B")
        ax.set_title("Test MAE, Experiment A and Experiment B, LKR/kg")
        ax.set_xlabel("horizon (weeks)")
        ax.set_ylabel("MAE")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        _save(fig, "experiment_b_vs_a_mae.png")
    benchmark = OUTPUTS / "evaluation" / "common_key_benchmark.csv"
    if benchmark.is_file():
        frame = pd.read_csv(benchmark)
        keep = frame[frame["model"].isin(["naive_last", "sarimax", "lightgbm", "xgboost"])]
        _mae_lines(keep, "model", "mae", "Held-out MAE on identical test observations", "final_model_benchmark_mae.png")
        _mae_lines(keep, "model", "mae", "Fair held-out MAE on identical test observations", "common_key_benchmark_mae.png")


def _interval_plots() -> None:
    predictions = pd.read_csv(OUTPUTS / "experiment_b" / "predictions.csv")
    cqr = json.loads((OUTPUTS / "experiment_b" / "cqr.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(OUTPUTS / "experiment_b" / "metrics_by_model_horizon.csv")
    light = metrics[metrics["slice"].eq("test") & metrics["model"].eq("lightgbm")].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(light["horizon"], light["picp_cqr"], marker="o")
    ax.axhline(90, linestyle="--", label="90% intended level")
    ax.set_title("LightGBM B test PICP vs horizon")
    ax.set_xlabel("horizon (weeks)")
    ax.set_ylabel("calibrated PICP (%)")
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, "picp_vs_horizon.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(light["horizon"], light["interval_width_cqr"], marker="o")
    ax.set_title("LightGBM B mean interval width vs horizon")
    ax.set_xlabel("horizon (weeks)")
    ax.set_ylabel("width (LKR/kg)")
    ax.grid(alpha=0.2)
    _save(fig, "width_vs_horizon.png")

    band = predictions[
        predictions["model"].eq("lightgbm")
        & predictions["split"].eq("test")
        & predictions["horizon"].eq(4)
        & predictions["crop"].astype(str).str.lower().eq("carrot")
        & predictions["market"].astype(str).str.lower().eq("dambulla")
    ].sort_values("target_week")
    qhat = float(cqr["qhat_by_model"]["lightgbm"]["4"])
    if not band.empty:
        low = np.minimum.reduce([band["y_pred_q05"], band["y_pred"], band["y_pred_q95"]]) - qhat
        high = np.maximum.reduce([band["y_pred_q05"], band["y_pred"], band["y_pred_q95"]]) + qhat
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(pd.to_datetime(band["target_week"]), low, high, color="0.8", label="ordered interval plus qhat")
        ax.plot(pd.to_datetime(band["target_week"]), band["y_true"], color="black", lw=1, label="actual")
        ax.plot(pd.to_datetime(band["target_week"]), band["y_pred"], color="tab:blue", lw=1, label="median")
        ax.set_title("Carrot, Dambulla, horizon 4, LightGBM B, test weeks")
        ax.legend(fontsize=8)
        _save(fig, "carrot_dambulla_h4_band.png")
    h4 = predictions[predictions["model"].eq("lightgbm") & predictions["split"].eq("test") & predictions["horizon"].eq(4)]
    if not h4.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(h4["y_true"] - h4["y_pred"], bins=35, color="black")
        ax.axvline(0, linestyle="--")
        ax.set_title("LightGBM B horizon 4 test residuals, actual minus median, LKR/kg")
        ax.set_xlabel("residual (LKR/kg)")
        _save(fig, "h4_residuals.png")


def _c_plots() -> None:
    comp_path = OUTPUTS / "experiment_c" / "b_star_vs_c.csv"
    if not comp_path.is_file():
        return
    comp = pd.read_csv(comp_path)
    primary = comp[comp["model"].eq("lightgbm")].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(primary["horizon"], primary["mae_b_star"], marker="o", label="B* — same real-progress rows")
    ax.plot(primary["horizon"], primary["mae_c"], marker="o", label="C — B + real progress")
    ax.set_title("Real cultivation-progress ablation: LightGBM B* vs C")
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("MAE (LKR/kg)")
    ax.set_xticks(list(HORIZONS))
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, "experiment_c_bstar_vs_c_mae.png")

    eligible_path = OUTPUTS / "experiment_c" / "c_eligible_table.parquet"
    if eligible_path.is_file():
        table = pd.read_parquet(eligible_path)
        fig, ax = plt.subplots(figsize=(8, 4))
        for split, group in table.groupby("split"):
            values = pd.to_numeric(group["cultivation_progress_ratio_real"], errors="coerce").dropna()
            if len(values):
                ax.hist(values, bins=25, alpha=0.4, label=split)
        ax.axvline(1.0, linestyle="--", label="target = 1.0")
        ax.set_title("Real cultivation-progress ratios used by Experiment C")
        ax.set_xlabel("achieved hectares / target hectares")
        ax.set_ylabel("row count")
        ax.legend()
        _save(fig, "experiment_c_progress_distribution.png")

    for label, path, filename in (
        ("crop", OUTPUTS / "experiment_c" / "lightgbm_h4_by_crop.csv", "experiment_c_h4_by_crop.png"),
        ("market", OUTPUTS / "experiment_c" / "lightgbm_h4_by_market.csv", "experiment_c_h4_by_market.png"),
    ):
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(frame))
        ax.bar(x - 0.15, frame["mae_b_star"], width=0.3, label="B*")
        ax.bar(x + 0.15, frame["mae_c"], width=0.3, label="C")
        ax.set_xticks(x)
        ax.set_xticklabels(frame[label].astype(str))
        ax.set_title(f"LightGBM horizon 4 MAE by {label}, matched B* and C rows")
        ax.set_ylabel("MAE (LKR/kg)")
        ax.legend()
        _save(fig, filename)

    metrics = pd.read_csv(OUTPUTS / "experiment_c" / "metrics_c.csv")
    light = metrics[metrics["model"].eq("lightgbm")].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(light["horizon"], light["picp_cqr"], marker="o", label="Empirical PICP")
    ax.axhline(90, linestyle="--", label="90% intended level")
    ax.set_title("LightGBM C calibrated interval coverage")
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("PICP (%)")
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, "final_c_picp_by_horizon.png")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(light["horizon"], light["interval_width_cqr"], marker="o")
    ax.set_title("LightGBM C calibrated interval width")
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("Mean interval width (LKR/kg)")
    ax.grid(alpha=0.2)
    _save(fig, "final_c_interval_width_by_horizon.png")

    c_pred = pd.read_csv(OUTPUTS / "experiment_c" / "predictions_c.csv")
    c_h4 = c_pred[c_pred["model"].eq("lightgbm") & c_pred["split"].eq("test") & c_pred["horizon"].eq(4)].copy()
    if c_h4.empty:
        return
    c_h4["residual"] = c_h4["y_true"] - c_h4["y_pred"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(c_h4["y_true"], c_h4["y_pred"], alpha=0.45, s=20)
    lower = float(min(c_h4["y_true"].min(), c_h4["y_pred"].min()))
    upper = float(max(c_h4["y_true"].max(), c_h4["y_pred"].max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_title("LightGBM C predicted versus actual price, horizon 4")
    ax.set_xlabel("Actual price (LKR/kg)")
    ax.set_ylabel("Predicted price (LKR/kg)")
    ax.grid(alpha=0.2)
    _save(fig, "final_c_h4_predicted_vs_actual.png")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(c_h4["y_pred"], c_h4["residual"], alpha=0.45, s=20)
    ax.axhline(0, linestyle="--")
    ax.set_title("LightGBM C residuals versus predicted price, horizon 4")
    ax.set_xlabel("Predicted price (LKR/kg)")
    ax.set_ylabel("Residual: actual - predicted (LKR/kg)")
    ax.grid(alpha=0.2)
    _save(fig, "final_c_h4_residual_vs_predicted.png")
    time_residual = c_h4.groupby("target_week", as_index=False)["residual"].mean().sort_values("target_week")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(pd.to_datetime(time_residual["target_week"]), time_residual["residual"])
    ax.axhline(0, linestyle="--")
    ax.set_title("Mean LightGBM C residual through the held-out period, horizon 4")
    ax.set_xlabel("Target week")
    ax.set_ylabel("Mean residual (LKR/kg)")
    ax.grid(alpha=0.2)
    _save(fig, "final_c_h4_residual_over_time.png")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(c_h4["residual"], bins=35)
    ax.axvline(0, linestyle="--")
    ax.set_title("LightGBM C residual distribution, horizon 4")
    ax.set_xlabel("Residual: actual - predicted (LKR/kg)")
    ax.set_ylabel("Observation count")
    _save(fig, "final_c_h4_residual_histogram.png")
    shock_n = int(c_h4["is_shock"].astype(str).str.lower().eq("true").sum()) if "is_shock" in c_h4.columns else 0
    note = {
        "horizon": 4,
        "model": "lightgbm",
        "feature_set": "C",
        "shock_rows": shock_n,
        "evaluable": shock_n > 0,
        "note": (
            f"2024 Q1 shock horizon-4 Experiment C rows in this run: {shock_n}."
            if shock_n
            else "2024 Q1 shock at horizon 4 is not evaluable: this run has zero Experiment C horizon-4 rows in that window."
        ),
    }
    (PLOT / "shock_note.json").write_text(json.dumps(note, indent=2), encoding="utf-8")


def _shap() -> None:
    model_path = OUTPUTS / "experiment_c" / "models" / "lightgbm_C_h4_q50.joblib"
    eligible_path = OUTPUTS / "experiment_c" / "c_eligible_table.parquet"
    if not model_path.is_file() or not eligible_path.is_file():
        return
    try:
        import shap
    except ImportError:
        print("shap is not installed; SHAP plot was not drawn", flush=True)
        return
    import joblib

    from htfe.training.reforecast_c import apply_categories
    from htfe.training.run_experiment_a import EVAL_IDS, _design

    bundle = joblib.load(model_path)
    table = pd.read_parquet(eligible_path)
    subset = table[
        table["forecast_horizon_weeks"].eq(4)
        & table["split"].eq("test")
        & table["market_id"].astype(str).isin(EVAL_IDS)
    ].copy()
    if subset.empty:
        return
    subset = apply_categories(subset, bundle["categories"])
    design, names = _design(subset, list(bundle["feature_columns"]))
    design = design[list(bundle["columns"])]
    sample = design.sample(n=min(200, len(design)), random_state=42)
    explainer = shap.TreeExplainer(bundle["model"])
    values = np.asarray(explainer.shap_values(sample))
    if values.ndim == 3:
        values = values[0]
    mean_abs = np.abs(values).mean(axis=0)
    ranked = pd.DataFrame({"feature": list(design.columns), "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    out = OUTPUTS / "experiment_c" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out / "shap_lightgbm_c_h4.csv", index=False)
    top = ranked.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["mean_abs_shap"], color="black")
    ax.set_title("LightGBM C horizon 4 mean absolute SHAP")
    ax.set_xlabel("mean absolute SHAP")
    _save(fig, "shap_lightgbm_c_h4_top_features.png")
    progress = ranked[ranked["feature"].eq("cultivation_progress_ratio_real")]
    if not progress.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(["cultivation_progress_ratio_real"], progress["mean_abs_shap"], color="black")
        ax.set_title("SHAP for the cultivation-progress feature, horizon 4")
        _save(fig, "shap_cultivation_progress_h4.png")


def main() -> None:
    _price_plots()
    _experiment_plots()
    if (OUTPUTS / "experiment_b" / "predictions.csv").is_file():
        _interval_plots()
    _c_plots()
    _shap()
    print(f"figures in {PLOT}", flush=True)


if __name__ == "__main__":
    main()
