"""Train Experiment B and compare it with the saved Experiment A metrics.

Usage from prototype/member2_ForecastingEngine:
  python -u -m training.run_experiment_b

B is price lags, seasonal features, origin-aligned historical weather, real
as-of macros, and lagged finished-season extent. It does not refit SARIMAX,
does not train C or S, and does not impute macros or cultivation.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from htfe.training.align_experiment_a import _cqr
from htfe.training.config import ARTIFACTS, CAL_END, EVAL_MARKETS, HORIZONS, TRAIN_END
from htfe.training.evaluate import _block, metrics_tables
from htfe.features.sets import (
    EXTENT_FEATURES,
    MACRO_FEATURES,
    PRICE_HISTORY,
    PROGRESS_FEATURE,
    SEASON_FEATURES,
    WEATHER_FEATURES,
    feature_sets,
    macros_ready,
    progress_available,
)
from htfe.training.run_experiment_a import (
    RANDOM_STATE,
    _assert_split,
    _eligible,
    _fit_horizon,
    _reuse_sarimax,
    _tune,
)

OUT = ARTIFACTS / "experiment_b"
A_DIR = ARTIFACTS / "experiment_a"
A_METRICS = A_DIR / "metrics_by_model_horizon.csv"
A_CQR = A_DIR / "cqr.json"
GAP_PATH = A_DIR / "sarimax_gap_predictions.csv"
from htfe.config import OUTPUTS
REPO = OUTPUTS
DOC_PATH = OUTPUTS / "notes" / "Experiment-AB-Comparison-IT23415836.md"
PROTECTED = (
    ARTIFACTS / "experiment_a",
    ARTIFACTS / "latest",
    ARTIFACTS / "methodology",
)
EXPECTED = (
    list(PRICE_HISTORY)
    + list(SEASON_FEATURES)
    + list(WEATHER_FEATURES)
    + list(MACRO_FEATURES)
    + list(EXTENT_FEATURES)
)
FORBIDDEN = {
    PROGRESS_FEATURE,
    "cultivation_target_exceeded",
    "origin_target_hectares",
    "origin_achieved_hectares",
    "achieved_hectares_change",
    "cultivation_report_age_days",
    "cultivation_report_month",
    "cultivation_scenario",
    "cultivation_is_synthetic",
    "cultivation_source",
    "cultivation_intensity",
    "cultivation_progress",
}
METRIC_FIELDS = ("mae", "rmse", "mape", "pinball", "picp", "interval_width")


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    if not root.exists():
        return found
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            found[str(path.resolve())] = (stat.st_mtime_ns, stat.st_size)
    return found


def _training_rows(table: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for horizon in HORIZONS:
        subset = _eligible(table, horizon)
        parts.append(subset.loc[subset["split"].eq("train")])
    return pd.concat(parts, ignore_index=True)


def _write_status(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "experiment_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _unavailable(reason: str, protected: dict) -> None:
    payload = {
        "A": {
            "status": "already_trained",
            "reason": "Not retrained. Metrics stay in artifacts/experiment_a.",
        },
        "B": {"status": "unavailable", "reason": reason},
        "C": {
            "status": "unavailable",
            "reason": "observed_cultivation_data_available=false. dcs_highland_seasonal.csv and synthetic_proxy were not used as current progress.",
        },
        "S": {"status": "not_run", "reason": "S was not run."},
        "observed_cultivation_data_available": False,
        "inflation_available_date_rule": "reference_month_end_plus_21_days",
    }
    _write_status(payload)
    _assert_protected(protected)
    print(reason, flush=True)
    print(f"wrote {OUT / 'experiment_status.json'}", flush=True)


def _assert_protected(before: dict) -> None:
    for root in PROTECTED:
        after = _snapshot(root)
        if after != before[str(root.resolve())]:
            raise RuntimeError(f"{root} changed during Experiment B")


def _assert_macros(table: pd.DataFrame, used: pd.DataFrame) -> dict[str, int]:
    """Stop when a required macro is all-null. Do not drop partial nulls."""
    all_null = []
    for name in MACRO_FEATURES:
        series = table[name] if name in table.columns else pd.Series(dtype=float)
        used_series = used[name] if name in used.columns else pd.Series(dtype=float)
        table_empty = name not in table.columns or not bool(series.notna().any())
        used_empty = used.empty or name not in used.columns or not bool(used_series.notna().any())
        if table_empty or used_empty:
            all_null.append(name)
    if all_null:
        raise SystemExit(
            "Experiment B is unavailable: diesel, USD/LKR, and inflation are not all non-null "
            f"on a real as-of series. All-null or missing: {', '.join(all_null)}."
        )
    nulls = {}
    for name in MACRO_FEATURES:
        count = int(used[name].isna().sum())
        nulls[name] = count
        if count:
            raise AssertionError(
                f"{name} is null on {count} training rows. Those rows were not dropped and were not imputed."
            )
    return nulls


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def _num(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _pct(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return round((new - old) / old * 100, 2)


def _cqr_table(payload: dict) -> pd.DataFrame:
    rows = payload["by_model_horizon"]
    return pd.DataFrame(rows)[["model", "horizon", "pinball", "picp_raw", "interval_width_raw", "picp_cqr", "interval_width_cqr", "qhat"]]


def _comparison(a_metrics: pd.DataFrame, b_metrics: pd.DataFrame, a_cqr: pd.DataFrame, b_cqr: pd.DataFrame) -> pd.DataFrame:
    left = a_metrics[a_metrics["slice"].eq("test") & a_metrics["target_form"].eq("price")].copy()
    right = b_metrics[b_metrics["slice"].eq("test") & b_metrics["target_form"].eq("price")].copy()
    merged = left.merge(right, on=["model", "horizon"], how="outer", suffixes=("_a", "_b"))
    merged = merged.merge(a_cqr.add_suffix("_cqra").rename(columns={"model_cqra": "model", "horizon_cqra": "horizon"}), on=["model", "horizon"], how="left")
    merged = merged.merge(b_cqr.add_suffix("_cqrb").rename(columns={"model_cqrb": "model", "horizon_cqrb": "horizon"}), on=["model", "horizon"], how="left")
    rows = []
    for _, row in merged.sort_values(["model", "horizon"]).iterrows():
        item = {
            "model": row["model"],
            "horizon": int(row["horizon"]),
            "n_scored_a": None if pd.isna(row["n_scored_a"]) else int(row["n_scored_a"]),
            "n_scored_b": None if pd.isna(row["n_scored_b"]) else int(row["n_scored_b"]),
        }
        for name in METRIC_FIELDS:
            item[f"{name}_a"] = _num(row.get(f"{name}_a"))
            item[f"{name}_b"] = _num(row.get(f"{name}_b"))
        item["mae_pct_change"] = _pct(item["mae_b"], item["mae_a"])
        item["rmse_pct_change"] = _pct(item["rmse_b"], item["rmse_a"])
        item["picp_cqr_a"] = _num(row.get("picp_cqr_cqra"))
        item["picp_cqr_b"] = _num(row.get("picp_cqr_cqrb"))
        item["interval_width_cqr_a"] = _num(row.get("interval_width_cqr_cqra"))
        item["interval_width_cqr_b"] = _num(row.get("interval_width_cqr_cqrb"))
        rows.append(item)
    return pd.DataFrame(rows)


def _verdict(comparison: pd.DataFrame, model: str) -> str:
    label = {"lightgbm": "LightGBM", "xgboost": "XGBoost"}[model]
    hit = comparison[comparison["model"].eq(model) & comparison["horizon"].eq(4)]
    if hit.empty:
        raise RuntimeError(f"missing 4-week row for {model}")
    row = hit.iloc[0]
    a_mae = float(row["mae_a"])
    b_mae = float(row["mae_b"])
    change = row["mae_pct_change"]
    if b_mae < a_mae:
        outcome = "improves"
    elif b_mae > a_mae:
        outcome = "does not improve"
    else:
        outcome = "matches"
    change_text = "the same MAE" if change is None else f"{change:+.2f}% MAE"
    return (
        f"B {outcome} 4-week MAE for {label}: B is {b_mae:.2f} and A is {a_mae:.2f} ({change_text})."
    )


def _crop_market(a_predictions: pd.DataFrame, b_predictions: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in (1, 4, 12):
        pool = comparison[comparison["horizon"].eq(horizon) & comparison["model"].isin(["lightgbm", "xgboost", "sarimax"])]
        best = pool.sort_values(["mae_b", "model"]).iloc[0]
        model = str(best["model"])
        for label, frame in (("A", a_predictions), ("B", b_predictions)):
            part = frame[frame["model"].eq(model) & frame["split"].eq("test") & frame["horizon"].eq(horizon)]
            for (crop, market), group in part.groupby(["crop", "market"], sort=True):
                scored = _block(
                    group["y_true"].to_numpy(dtype=float),
                    group["y_pred"].to_numpy(dtype=float),
                    group["y_pred_q05"].to_numpy(dtype=float),
                    group["y_pred_q95"].to_numpy(dtype=float),
                )
                rows.append(
                    {
                        "horizon": horizon,
                        "model": model,
                        "crop": crop,
                        "market": market,
                        "side": label,
                        "n_scored": scored["n_scored"],
                        "mae": scored["mae"],
                        "mape": scored["mape"],
                    }
                )
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=["horizon", "model", "crop", "market"], columns="side", values=["mae", "mape", "n_scored"], aggfunc="first")
    wide.columns = [f"{metric}_{side.lower()}" for metric, side in wide.columns]
    return wide.reset_index()


def _write_doc(comparison: pd.DataFrame, verdicts: list[str], crops: pd.DataFrame, sarimax_note: str) -> None:
    header = (
        "| Model | H | MAE A | MAE B | MAE % | RMSE A | RMSE B | RMSE % | MAPE A | MAPE B | "
        "Pinball A | Pinball B | PICP A | PICP B | Width A | Width B | PICP CQR A | PICP CQR B | Width CQR A | Width CQR B |"
    )
    rule = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    body = []
    for _, row in comparison.iterrows():
        body.append(
            "| {model} | {horizon} | {mae_a} | {mae_b} | {mae_pct} | {rmse_a} | {rmse_b} | {rmse_pct} | {mape_a} | {mape_b} | "
            "{pinball_a} | {pinball_b} | {picp_a} | {picp_b} | {width_a} | {width_b} | {picp_cqr_a} | {picp_cqr_b} | {width_cqr_a} | {width_cqr_b} |".format(
                model=row["model"],
                horizon=int(row["horizon"]),
                mae_a=_fmt(row["mae_a"]),
                mae_b=_fmt(row["mae_b"]),
                mae_pct=_fmt(row["mae_pct_change"]),
                rmse_a=_fmt(row["rmse_a"]),
                rmse_b=_fmt(row["rmse_b"]),
                rmse_pct=_fmt(row["rmse_pct_change"]),
                mape_a=_fmt(row["mape_a"]),
                mape_b=_fmt(row["mape_b"]),
                pinball_a=_fmt(row["pinball_a"]),
                pinball_b=_fmt(row["pinball_b"]),
                picp_a=_fmt(row["picp_a"]),
                picp_b=_fmt(row["picp_b"]),
                width_a=_fmt(row["interval_width_a"]),
                width_b=_fmt(row["interval_width_b"]),
                picp_cqr_a=_fmt(row["picp_cqr_a"]),
                picp_cqr_b=_fmt(row["picp_cqr_b"]),
                width_cqr_a=_fmt(row["interval_width_cqr_a"]),
                width_cqr_b=_fmt(row["interval_width_cqr_b"]),
            )
        )
    crop_header = "| Horizon | Model | Crop | Market | MAE A | MAE B | MAPE A | MAPE B |"
    crop_rule = "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |"
    crop_body = []
    for _, row in crops.iterrows():
        crop_body.append(
            f"| {int(row['horizon'])} | {row['model']} | {row['crop']} | {row['market']} | {_fmt(row['mae_a'])} | {_fmt(row['mae_b'])} | {_fmt(row['mape_a'])} | {_fmt(row['mape_b'])} |"
        )
    text = "\n".join(
        [
            "# Experiment A vs B",
            "",
            "Test issue weeks are 2024-2025. Markets are Dambulla, Colombo, Meegoda, and Nuwara Eliya. Horizons are 1, 2, 4, 8, and 12 weeks. Scores use the same observed-price mask as Experiment A.",
            "",
            "Experiment A is price lags and seasonal features. Experiment B adds origin-aligned historical weather, real as-of diesel, USD/LKR, and CCPI inflation, and lagged finished-season extent (`previous_season_extent`, `same_season_previous_year_extent`, `historical_mean_season_extent`, `extent_change_vs_previous_season`). Lagged extent is historical production, not current cultivation progress. `cultivation_progress_ratio` is null and is not a feature. The synthetic calendar is not a feature. Experiment C is unavailable. S was not run.",
            "",
            "MAE and RMSE percent change is (B − A) / A × 100. A negative percent is a lower error for B. PICP and width are the raw 0.05/0.95 intervals. PICP CQR and width CQR apply the 2023 split-conformal adjustment to the 2024-2025 test. SARIMAX interval columns stay blank.",
            "",
            sarimax_note,
            "",
            "## 4-week MAE",
            "",
            verdicts[0],
            "",
            verdicts[1],
            "",
            "## 2024-2025 test",
            "",
            header,
            rule,
            *body,
            "",
            "## Crop and market, best point model",
            "",
            "The model at each of horizons 1, 4, and 12 is the one with the lowest pooled test MAE in Experiment B.",
            "",
            crop_header,
            crop_rule,
            *crop_body,
            "",
            "## Data caveats",
            "",
            "Published CCPI headline year-on-year uses base 2013=100 through January 2023 and base 2021=100 from February 2023. The index was not rescaled from one base onto the other. This is a break in the published inflation series, not a model failure.",
            "",
            "Inflation available dates use reference month-end plus 21 days. The reference month is not treated as the publication date. A forecast issued in the first three weeks of a month does not see that month's CCPI, and it does not see the previous month's CCPI until the 21-day lag has elapsed. Diesel uses the CPC Lanka Auto Diesel revision date. USD/LKR uses the CBSL observation date.",
            "",
        ]
    )
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(text, encoding="utf-8")


def _prepare_table() -> pd.DataFrame:
    from htfe.config import built_file

    path = built_file("training_table.parquet")
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_parquet(path)
    table["crop_id"] = pd.Categorical(table["crop_id"])
    table["market_id"] = pd.Categorical(table["market_id"])
    table["season"] = pd.Categorical(table["season"], categories=["Maha", "Yala"])
    return table


def _assert_inflation_rule(table: pd.DataFrame) -> str:
    from htfe.data.build.fetch_published_macros import (
        INFLATION_AVAILABLE_DATE_RULE,
        PUBLISHED_PATH,
        inflation_rule_ok,
    )

    published = pd.read_csv(PUBLISHED_PATH, parse_dates=["available_date", "reference_month_end"])
    if not inflation_rule_ok(published):
        raise RuntimeError("inflation available_date is still the reference month end; Experiment B will not train")
    early = table.loc[pd.to_datetime(table["forecast_date"]).eq(pd.Timestamp("2024-03-04")), "inflation_rate_pct"]
    if early.empty or abs(float(early.iloc[0]) - 6.4) > 1e-6:
        raise AssertionError(f"2024-03-04 inflation is {None if early.empty else float(early.iloc[0])}, expected lagged January CCPI 6.4")
    return INFLATION_AVAILABLE_DATE_RULE


def _aligned_sarimax(table: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    reused, note = _reuse_sarimax(table)
    if not GAP_PATH.exists():
        return reused, note
    gaps = pd.read_csv(GAP_PATH)
    gaps["forecast_issue_week"] = pd.to_datetime(gaps["forecast_issue_week"])
    gaps["target_week"] = pd.to_datetime(gaps["target_week"])
    both = pd.concat([reused, gaps], ignore_index=True)
    both = both.drop_duplicates(["crop", "market", "forecast_issue_week", "horizon"], keep="first")
    return both, (
        f"{note} Added {len(gaps)} refit gap rows from artifacts/experiment_a/sarimax_gap_predictions.csv. "
        "Matching rows stayed reused. artifacts/latest was not rewritten."
    )


def _test_identity(predictions: pd.DataFrame) -> dict:
    test = predictions[predictions["model"].eq("lightgbm") & predictions["split"].eq("test")].copy()
    test["forecast_issue_week"] = pd.to_datetime(test["forecast_issue_week"]).dt.strftime("%Y-%m-%d")
    keys = test[["crop", "market", "forecast_issue_week", "horizon"]].astype(str)
    blob = "\n".join(sorted(keys.agg("|".join, axis=1)))
    return {"test_row_count": int(len(test)), "test_row_sha256": hashlib.sha256(blob.encode()).hexdigest()}


def _boundaries(table: pd.DataFrame) -> dict:
    issue = pd.to_datetime(table["forecast_issue_week"])
    bounds = {}
    for split in ("train", "calibrate", "test"):
        part = issue[table["split"].eq(split)]
        bounds[f"{split}_issue_min"] = str(part.min().date())
        bounds[f"{split}_issue_max"] = str(part.max().date())
    return bounds


def main() -> None:
    if OUT.resolve() in {path.resolve() for path in PROTECTED}:
        raise RuntimeError("Experiment B output path collides with a protected artifact folder")
    protected = {str(path.resolve()): _snapshot(path) for path in PROTECTED}
    if OUT.exists():
        print("discarding Experiment B artifacts fitted with CCPI reference month-end dates", flush=True)
        shutil.rmtree(OUT)
    table = _prepare_table()
    _assert_split(table)
    inflation_rule = _assert_inflation_rule(table)
    ready, macro_reason = macros_ready(table)
    sets = feature_sets(table)
    columns = sets["B"]
    if not ready or columns is None:
        missing = [name for name in MACRO_FEATURES if name not in table.columns or not bool(table[name].notna().any())]
        if missing or not ready:
            _unavailable(macro_reason, protected)
            sys.exit(2)
        absent = [name for name in EXPECTED if name not in table.columns]
        raise RuntimeError(f"Experiment B columns are incomplete: {absent}")
    if columns != EXPECTED:
        raise RuntimeError(f"Experiment B features do not match the proposal: extra={set(columns) - set(EXPECTED)} missing={set(EXPECTED) - set(columns)}")
    banned = [name for name in columns if name in FORBIDDEN or "synthetic" in name]
    if banned:
        raise RuntimeError(f"Experiment B includes forbidden columns: {banned}")
    if PROGRESS_FEATURE in table.columns and bool(table[PROGRESS_FEATURE].notna().any()):
        raise RuntimeError("cultivation_progress_ratio is not all null; Experiment C was not trained and B will not use it until this is reviewed")
    used = _training_rows(table)
    try:
        nulls = _assert_macros(table, used)
    except SystemExit as stop:
        _unavailable(str(stop), protected)
        raise
    print(macro_reason, flush=True)
    print(f"training-row macro nulls {nulls}", flush=True)
    print(f"cultivation progress available={progress_available(table)}; C and S are not trained", flush=True)
    print("optuna horizon 4", flush=True)
    tuned = _tune(table, columns)
    print(tuned, flush=True)
    parts = []
    for horizon in HORIZONS:
        print(f"fitting horizon {horizon}", flush=True)
        subset = _eligible(table, horizon)
        for name in MACRO_FEATURES:
            train_nulls = int(subset.loc[subset["split"].eq("train"), name].isna().sum())
            if train_nulls:
                raise AssertionError(f"{name} is null on {train_nulls} horizon-{horizon} training rows")
        parts.append(_fit_horizon(subset, columns, tuned["lightgbm"], tuned["xgboost"]))
    sarimax, sarimax_note = _aligned_sarimax(table)
    print(sarimax_note, flush=True)
    if not sarimax.empty:
        parts.append(sarimax)
    predictions = pd.concat(parts, ignore_index=True)
    predictions.loc[predictions["model"].isin(["lightgbm", "xgboost"]), "feature_set"] = "B"
    reused_mask = predictions["model"].eq("sarimax") & ~predictions["feature_set"].eq("sarimax_gap_refit")
    predictions.loc[reused_mask, "feature_set"] = "reused_sarimax"
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    by_model, by_series = metrics_tables(predictions)
    _b_cqr_rows, b_cqr = _cqr(predictions)
    (OUT / "cqr.json").write_text(json.dumps(b_cqr, indent=2), encoding="utf-8")
    by_model = by_model.merge(
        _b_cqr_rows[["model", "horizon", "qhat", "picp_cqr", "interval_width_cqr"]],
        on=["model", "horizon"],
        how="left",
    )
    by_model.loc[~by_model["slice"].eq("test"), ["qhat", "picp_cqr", "interval_width_cqr"]] = np.nan
    by_model.to_csv(OUT / "metrics_by_model_horizon.csv", index=False)
    by_series.to_csv(OUT / "metrics_by_series.csv", index=False)
    a_saved = pd.read_csv(A_METRICS)
    a_trees = a_saved[a_saved["model"].isin(["lightgbm", "xgboost"])]
    a_sarimax_metrics, _ = metrics_tables(sarimax.assign(target_form="price"))
    a_metrics = pd.concat([a_trees, a_sarimax_metrics], ignore_index=True)
    a_cqr = _cqr_table(json.loads(A_CQR.read_text(encoding="utf-8")))
    comparison = _comparison(a_metrics, by_model, a_cqr, _cqr_table(b_cqr))
    verdicts = [_verdict(comparison, "lightgbm"), _verdict(comparison, "xgboost")]
    a_predictions = pd.read_csv(A_DIR / "predictions.csv")
    a_predictions = pd.concat([a_predictions[a_predictions["model"].ne("sarimax")], sarimax], ignore_index=True)
    crops = _crop_market(a_predictions, predictions, comparison)
    comparison.to_csv(OUT / "ab_comparison.csv", index=False)
    crops.to_csv(OUT / "ab_comparison_by_crop_market.csv", index=False)
    manifest = {
        "experiment": "B",
        "description": (
            "Price lags, seasonal features, origin-aligned historical weather, real as-of diesel, "
            "USD/LKR, and CCPI inflation, plus lagged finished-season extent. "
            "No current cultivation progress and no synthetic calendar."
        ),
        "features": columns,
        "excluded": [PROGRESS_FEATURE, "synthetic calendar columns"],
        "identity_codes": ["crop_id", "market_id"],
        "season_code": "season",
        "label": "target_price_lkr_kg at target_week",
        "model_target": "log(target_price / last_observed_price), inverted back to LKR/kg",
        "quantiles": [0.05, 0.50, 0.95],
        "random_state": RANDOM_STATE,
        "split": {
            "train_issue_through": TRAIN_END,
            "calibrate": "2023",
            "test": "2024-2025",
            "shuffle": False,
            "calibrate_end": CAL_END,
        },
        "score_markets": list(EVAL_MARKETS),
        "train_pool_markets": ["Kandy", "Keppetipola", "Thambuttegama"],
        "score_rule": "last observed price and target price both originally observed",
        "macro_nulls_on_training_rows": nulls,
        "macro_imputed": False,
        "extent_rows_dropped": False,
        "optuna": tuned,
        "sarimax": sarimax_note,
        "ccpi_caveat": (
            "Base 2013=100 through January 2023 and base 2021=100 from February 2023. "
            "The index was not rescaled. This is a published-series break, not a model failure."
        ),
        "inflation_available_date_rule": inflation_rule,
        "date_boundaries": _boundaries(table),
        "rows": int(len(table)),
        "training_rows_used": int(len(used)),
        **_test_identity(predictions),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    comparability = json.loads((A_DIR / "sarimax_comparability.json").read_text(encoding="utf-8"))
    sarimax_doc = comparability["decision"]
    _write_doc(comparison, verdicts, crops, sarimax_doc)
    identity = _test_identity(predictions)
    payload = {
        "observed_cultivation_data_available": False,
        "inflation_available_date_rule": inflation_rule,
        "A": {
            "status": "already_trained",
            "reason": "Not retrained. Quantile and CQR files were added beside the saved predictions.",
        },
        "B": {
            "status": "trained",
            "reason": "Experiment B trained on price lags, season, origin weather, real as-of macros, and lagged finished-season extent after the inflation publication lag.",
        },
        "C": {
            "status": "unavailable",
            "reason": "observed_cultivation_data_available=false. dcs_highland_seasonal.csv and synthetic_proxy were not used as current progress.",
        },
        "S": {"status": "not_run", "reason": "S was not run."},
        "features": columns,
        "date_boundaries": manifest["date_boundaries"],
        **identity,
        "sarimax": sarimax_note,
        "sarimax_exact_match_before_gap_fill": comparability["exact_match_before_gap_fill"],
        "sarimax_exact_match_after_gap_fill": comparability["exact_match_after_gap_fill"],
        "four_week_mae": verdicts,
        "artifacts": {
            "predictions": str(OUT / "predictions.csv"),
            "metrics": str(OUT / "metrics_by_model_horizon.csv"),
            "manifest": str(OUT / "manifest.json"),
            "comparison_csv": str(OUT / "ab_comparison.csv"),
            "comparison_md": str(DOC_PATH),
        },
    }
    _write_status(payload)
    _assert_protected(protected)
    test = by_model[by_model["slice"].eq("test")][
        ["model", "horizon", "mae", "rmse", "mape", "pinball", "picp", "interval_width", "n_scored"]
    ]
    print(test.to_string(index=False), flush=True)
    for line in verdicts:
        print(line, flush=True)
    print(f"wrote {OUT / 'experiment_status.json'}", flush=True)
    print(f"wrote {DOC_PATH}", flush=True)


if __name__ == "__main__":
    main()
