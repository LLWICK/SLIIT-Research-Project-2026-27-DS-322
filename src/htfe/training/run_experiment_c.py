"""Train matched-row B* and national Experiment C.

B* is Experiment B's feature list on the rows that have a real national
cultivation report. C is those same rows plus the as-of progress ratio and the
target-exceeded flag. LightGBM hyperparameters are read from
outputs/experiment_b/manifest.json (seed 42). Optuna is not run again.

This script writes only outputs/experiment_c/. It does not rewrite Experiment A
or Experiment B score files, and it does not rewrite app/data.

Usage from the repo root:
  python -u -m htfe.training.run_experiment_c
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from htfe.config import OUTPUTS, REPO_ROOT, built_file
from htfe.data.build.doa_national_progress import (
    EXCLUDED_ISSUE_YEARS,
    attach_national_asof,
    audit_progress,
    audit_summary,
    coverage_tables,
    load_doa,
)
from htfe.features.sets import PROGRESS_FEATURE
from htfe.training.config import CAL_END, HORIZONS, TRAIN_END
from htfe.training.evaluate import _block
from htfe.training.reforecast_c import MODEL_VERSION, write_reforecast
from htfe.training.run_experiment_a import (
    EVAL_IDS,
    MARKET_NAME,
    RANDOM_STATE,
    _assert_split,
    _design,
    _eligible,
    _score_frame,
)

OUT = OUTPUTS / "experiment_c"
B_MANIFEST = OUTPUTS / "experiment_b" / "manifest.json"
FEATURES_C_EXTRA = ("cultivation_progress_ratio_real", "cultivation_target_exceeded_real")
PROTECTED = (
    OUTPUTS / "experiment_a" / "metrics_by_model_horizon.csv",
    OUTPUTS / "experiment_a" / "manifest.json",
    OUTPUTS / "experiment_a" / "predictions.csv",
    OUTPUTS / "experiment_a" / "cqr.json",
    OUTPUTS / "experiment_b" / "metrics_by_model_horizon.csv",
    OUTPUTS / "experiment_b" / "manifest.json",
    OUTPUTS / "experiment_b" / "predictions.csv",
    OUTPUTS / "experiment_b" / "cqr.json",
    REPO_ROOT / "app" / "data" / "evaluation.json",
    REPO_ROOT / "app" / "data" / "comparison.json",
    REPO_ROOT / "app" / "data" / "ablation.json",
    REPO_ROOT / "app" / "data" / "models.json",
    REPO_ROOT / "app" / "data" / "predictions.json",
)


def _hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_hashes() -> dict[str, str | None]:
    return {str(path): _hash_file(path) for path in PROTECTED}


def _assert_unchanged(before: dict[str, str | None]) -> None:
    after = _protected_hashes()
    changed = [path for path, digest in before.items() if after.get(path) != digest]
    if changed:
        raise RuntimeError("Protected A/B or studio score files changed: " + ", ".join(changed))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _lock_categories(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = frame.copy()
    spec = {
        "crop_id": sorted(out["crop_id"].astype(str).unique().tolist()),
        "market_id": sorted(out["market_id"].astype(str).unique().tolist()),
        "season": ["Maha", "Yala"],
    }
    out["crop_id"] = pd.Categorical(out["crop_id"].astype(str), categories=spec["crop_id"])
    out["market_id"] = pd.Categorical(out["market_id"].astype(str), categories=spec["market_id"])
    out["season"] = pd.Categorical(out["season"].astype(str), categories=spec["season"])
    return out, spec


def _attach(table: pd.DataFrame, audited: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    issue = table[["crop_id", "market_id", "forecast_date"]].drop_duplicates()
    asof = attach_national_asof(issue, audited)
    asof["forecast_date"] = pd.to_datetime(asof["forecast_date"]).dt.normalize().astype("datetime64[ns]")
    out = table.copy()
    out["forecast_date"] = pd.to_datetime(out["forecast_date"]).dt.normalize().astype("datetime64[ns]")
    out["crop_id"] = out["crop_id"].astype(str)
    out["market_id"] = out["market_id"].astype(str)
    merged = out.merge(
        asof,
        on=["crop_id", "market_id", "forecast_date"],
        how="left",
        validate="many_to_one",
    )
    merged["c_progress_available"] = (
        merged["cultivation_progress_ratio_real"].notna()
        & merged["cultivation_source_scope"].eq("national")
    )
    return merged, asof


def _eligible_national(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    subset = _eligible(frame, horizon)
    keep = subset["c_progress_available"].astype(bool) & subset["cultivation_source_scope"].eq("national")
    kept = subset.loc[keep].copy()
    if len(kept) and bool(kept["issue_year"].isin(EXCLUDED_ISSUE_YEARS).any()):
        raise AssertionError("an excluded issue year entered the C eligible rows")
    if len(kept) and not bool(kept["cultivation_source_scope"].eq("national").all()):
        raise AssertionError("a non-national row entered the C eligible rows")
    return kept


def _readiness(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        subset = _eligible_national(frame, horizon)
        for split in ("train", "calibrate", "test"):
            rows.append(
                {
                    "forecast_horizon_weeks": int(horizon),
                    "split": split,
                    "rows": int(subset["split"].eq(split).sum()),
                }
            )
    return pd.DataFrame(rows)


def _ready(counts: pd.DataFrame) -> bool:
    pivot = counts.pivot(index="forecast_horizon_weeks", columns="split", values="rows").fillna(0)
    for split in ("train", "calibrate", "test"):
        if split not in pivot.columns or not bool((pivot[split] > 0).all()):
            return False
    return bool(pivot.reindex(list(HORIZONS)).notna().all().all())


def _metrics(predictions: pd.DataFrame, cqr_payload: dict) -> pd.DataFrame:
    from htfe.training.evaluate import project_interval_around_median

    rows = []
    qhats = cqr_payload["qhat_by_model"]
    test = predictions[predictions["split"].eq("test")]
    for (model, horizon), group in test.groupby(["model", "horizon"]):
        if model not in qhats or str(int(horizon)) not in qhats[model]:
            continue
        low, mid, high = project_interval_around_median(
            group["y_pred_q05"].to_numpy(dtype=float),
            group["y_pred"].to_numpy(dtype=float),
            group["y_pred_q95"].to_numpy(dtype=float),
        )
        qhat = float(qhats[model][str(int(horizon))])
        y_true = group["y_true"].to_numpy(dtype=float)
        raw = _block(y_true, mid, low, high)
        calibrated = _block(y_true, mid, low - qhat, high + qhat)
        rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "n_scored": raw["n_scored"],
                "mae": raw["mae"],
                "rmse": raw["rmse"],
                "mape": raw["mape"],
                "pinball": raw["pinball"],
                "picp_raw": raw["picp"],
                "interval_width_raw": raw["interval_width"],
                "qhat": round(qhat, 4),
                "picp_cqr": calibrated["picp"],
                "interval_width_cqr": calibrated["interval_width"],
            }
        )
    return pd.DataFrame(rows)


def _comparison(b_metrics: pd.DataFrame, c_metrics: pd.DataFrame) -> pd.DataFrame:
    merged = b_metrics.merge(c_metrics, on=["model", "horizon"], suffixes=("_b_star", "_c"))
    merged["mae_change_pct"] = ((merged["mae_c"] - merged["mae_b_star"]) / merged["mae_b_star"] * 100).round(2)
    merged["mape_change_pct"] = ((merged["mape_c"] - merged["mape_b_star"]) / merged["mape_b_star"] * 100).round(2)
    return merged.sort_values("horizon").reset_index(drop=True)


def _statements(comparison: pd.DataFrame) -> list[str]:
    lines = [
        "B* and C use the same eligible rows. B* does not see current cultivation progress. C adds the national as-of ratio and the target-exceeded flag.",
        "Spatial scope is national. National cultivation progress is not district supply progress. District rows were not used as features.",
        "The full-sample Experiment A and Experiment B table is a different row set and is not mixed with these rows.",
    ]
    for row in comparison.itertuples(index=False):
        horizon = int(row.horizon)
        model = str(row.model)
        lines.append(
            f"{model} horizon {horizon} MAE: B* {row.mae_b_star:.2f}, C {row.mae_c:.2f}, on {int(row.n_scored_c)} matched test rows."
        )
        lines.append(
            f"{model} horizon {horizon} MAPE: B* {row.mape_b_star:.2f}%, C {row.mape_c:.2f}%."
        )
        if float(row.mae_change_pct) < 0:
            lines.append(
                f"At horizon {horizon}, {model} C MAE is lower than B* by {abs(float(row.mae_change_pct)):.2f}% on this matched-row test."
            )
        elif float(row.mae_change_pct) > 0:
            lines.append(
                f"At horizon {horizon}, {model} C MAE is higher than B* by {float(row.mae_change_pct):.2f}% on this matched-row test."
            )
        else:
            lines.append(f"At horizon {horizon}, {model} C MAE matches B*.")
        picp = float(row.picp_cqr_c)
        if picp < 90:
            lines.append(
                f"{model} horizon {horizon} calibrated PICP for C is {picp:.2f}%. That is below 90%."
            )
        else:
            lines.append(
                f"{model} horizon {horizon} calibrated PICP for C is {picp:.2f}%. The nominal interval is 90%. This is the measured share of test prices inside the interval."
            )
    return lines


def _h4_breakdown(b_pred: pd.DataFrame, c_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["crop", "market", "forecast_issue_week", "target_week", "horizon"]
    left = b_pred[b_pred["model"].eq("lightgbm") & b_pred["split"].eq("test") & b_pred["horizon"].eq(4)]
    right = c_pred[c_pred["model"].eq("lightgbm") & c_pred["split"].eq("test") & c_pred["horizon"].eq(4)]
    pair = left[key + ["y_true", "y_pred"]].merge(
        right[key + ["y_pred"]],
        on=key,
        suffixes=("_b_star", "_c"),
        validate="one_to_one",
    )
    pair["ae_b_star"] = (pair["y_pred_b_star"] - pair["y_true"]).abs()
    pair["ae_c"] = (pair["y_pred_c"] - pair["y_true"]).abs()

    def _agg(column: str) -> pd.DataFrame:
        frame = pair.groupby(column, as_index=False).agg(
            n=("y_true", "size"),
            mae_b_star=("ae_b_star", "mean"),
            mae_c=("ae_c", "mean"),
        )
        frame["mae_change_pct"] = ((frame["mae_c"] - frame["mae_b_star"]) / frame["mae_b_star"] * 100).round(2)
        frame[["mae_b_star", "mae_c"]] = frame[["mae_b_star", "mae_c"]].round(2)
        return frame

    return _agg("crop"), _agg("market")


def _status_base(summary: dict, counts: pd.DataFrame, ready: bool) -> dict:
    return {
        "status": "ready_to_train" if ready else "not_ready_for_locked_split",
        "experiment_c_trained": False,
        "spatial_scope": "national",
        "district_rows_used_for_scoring": False,
        "exact_season_matching": True,
        "report_age_cutoff_days": None,
        "no_synthetic_values": True,
        "excluded_issue_years": sorted(EXCLUDED_ISSUE_YEARS),
        "required_horizons": list(HORIZONS),
        "readiness_rule": (
            "Every horizon must contain at least one real national-progress row "
            "in train, 2023 calibration, and the 2024-2025 test."
        ),
        "readiness": counts.to_dict(orient="records"),
        "audit": summary,
        "runtime_model_note": "The dashboard forecast is the runtime model. It is not Experiment B and it is not Experiment C.",
    }


def _prepare() -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    if not B_MANIFEST.is_file():
        raise FileNotFoundError(B_MANIFEST)
    b_manifest = json.loads(B_MANIFEST.read_text(encoding="utf-8"))
    if int(b_manifest.get("random_state", -1)) != RANDOM_STATE:
        raise RuntimeError("Experiment B manifest is not seed 42")
    features_b = list(b_manifest["features"])
    if PROGRESS_FEATURE in features_b or any(name in features_b for name in FEATURES_C_EXTRA):
        raise RuntimeError("Experiment B feature list already contains current cultivation progress")
    table = pd.read_parquet(built_file("training_table.parquet"))
    missing = [name for name in features_b if name not in table.columns]
    if missing:
        raise RuntimeError(f"training table is missing B features: {missing}")
    source = load_doa()
    source_path = str(source.attrs.get("source_path"))
    audited = audit_progress(source)
    attached, asof = _attach(table, audited)
    summary = audit_summary(audited, asof, source_path)
    summary["b_features"] = features_b
    summary["lightgbm_params_source"] = "notebook fixed hyperparameters; Optuna was not run"
    return attached, summary, {"features_b": features_b, "audited": audited}, asof


def _annotate(predictions: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    keys = eligible.copy()
    keys["crop"] = keys["crop_id"].astype(str)
    keys["market_name"] = keys["market_id"].astype(str).map(MARKET_NAME)
    keys["forecast_issue_week"] = pd.to_datetime(keys["forecast_issue_week"])
    keys["horizon"] = keys["forecast_horizon_weeks"].astype(int)
    keep = keys[
        [
            "crop",
            "market_name",
            "forecast_issue_week",
            "horizon",
            "market_id",
            "cultivation_progress_ratio_real",
            "cultivation_target_exceeded_real",
            "cultivation_report_date",
            "cultivation_exact_season_id",
        ]
    ].drop_duplicates(["crop", "market_name", "forecast_issue_week", "horizon"])
    out = predictions.copy()
    out["forecast_issue_week"] = pd.to_datetime(out["forecast_issue_week"])
    out = out.merge(
        keep,
        left_on=["crop", "market", "forecast_issue_week", "horizon"],
        right_on=["crop", "market_name", "forecast_issue_week", "horizon"],
        how="left",
    )
    out["market_id"] = out["market_id"].astype(str)
    out["cultivation_source_scope"] = "national"
    return out


def _train(attached: pd.DataFrame, summary: dict, bundle: dict) -> None:
    from htfe.training.conformal import cqr_tables
    from htfe.training.run_experiment_a import fit_quantile_trees

    features_b = bundle["features_b"]
    features_c = features_b + list(FEATURES_C_EXTRA)
    counts = _readiness(attached)
    ready = _ready(counts)
    print(counts.to_string(index=False), flush=True)
    parts = []
    for horizon in HORIZONS:
        parts.append(_eligible_national(attached, horizon))
    eligible = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(eligible):
        _assert_split(eligible)
        if not bool(eligible["c_progress_available"].astype(bool).all()):
            raise AssertionError("eligible rows include a week without real national progress")
    eligible, spec = _lock_categories(eligible) if len(eligible) else (eligible, {})
    OUT.mkdir(parents=True, exist_ok=True)
    audit_dir = OUT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    bundle["audited"].to_csv(audit_dir / "doa_cultivation_progress_audited.csv", index=False)
    mismatch = bundle["audited"][bundle["audited"]["source_date_season_mismatch"]]
    mismatch.to_csv(audit_dir / "doa_source_vs_observation_season_mismatch.csv", index=False)
    asof_keys = attached.drop_duplicates(["crop_id", "market_id", "forecast_date"])
    yearly, season = coverage_tables(asof_keys)
    yearly.to_csv(audit_dir / "doa_progress_asof_year_coverage.csv", index=False)
    season.to_csv(audit_dir / "doa_progress_asof_exact_season_coverage.csv", index=False)
    counts.to_csv(OUT / "readiness_counts.csv", index=False)
    if len(eligible):
        eligible.to_parquet(OUT / "c_eligible_table.parquet", index=False)
    _write_json(OUT / "audit_summary.json", summary)
    score_ready = True
    score_gaps = []
    for horizon in HORIZONS:
        subset = eligible[eligible["forecast_horizon_weeks"].eq(horizon)] if len(eligible) else eligible
        scored = subset[subset["market_id"].astype(str).isin(EVAL_IDS)] if len(subset) else subset
        for split in ("calibrate", "test"):
            n_split = int(scored["split"].eq(split).sum()) if len(scored) else 0
            if n_split <= 0:
                score_ready = False
                score_gaps.append(f"h{horizon} {split}")
    status = _status_base(summary, counts, ready and score_ready)
    status["eligible_rows"] = int(len(eligible))
    status["score_market_gaps"] = score_gaps
    if score_gaps:
        status["reason"] = (
            "Evaluation markets are missing 2023 or 2024-2025 rows with real national progress: "
            + ", ".join(score_gaps)
        )
    _write_json(OUT / "status.json", status)
    if not ready or not score_ready:
        print("C is not ready on the locked split. No model was fit.", flush=True)
        if score_gaps:
            print("missing scored rows: " + ", ".join(score_gaps), flush=True)
        return

    print("notebook protocol for B* and C: 2022 early stop, 2023 only for qhat", flush=True)
    b_pred, _b_records = fit_quantile_trees(eligible, features_b, "B_star", model_dir=OUT / "models_b_star")
    c_pred, _c_records = fit_quantile_trees(eligible, features_c, "C", model_dir=OUT / "models")
    b_pred = _annotate(b_pred, eligible)
    c_pred = _annotate(c_pred, eligible)
    key = ["crop", "market", "forecast_issue_week", "target_week", "horizon", "model"]
    b_keys = set(map(tuple, b_pred[key].itertuples(index=False, name=None)))
    c_keys = set(map(tuple, c_pred[key].itertuples(index=False, name=None)))
    if b_keys != c_keys:
        raise RuntimeError(f"B* and C keys differ: B*={len(b_keys)} C={len(c_keys)} shared={len(b_keys & c_keys)}")

    b_pred.to_csv(OUT / "predictions_b_star.csv", index=False)
    c_pred.to_csv(OUT / "predictions_c.csv", index=False)
    _b_rows, b_cqr = cqr_tables(b_pred)
    _c_rows, c_cqr = cqr_tables(c_pred)
    b_cqr["models"] = ["lightgbm", "xgboost"]
    c_cqr["models"] = ["lightgbm", "xgboost"]
    b_cqr["note"] = "qhat uses the 2023 calibration rows only. Tree counts were selected on 2022."
    c_cqr["note"] = b_cqr["note"]
    _write_json(OUT / "cqr_b_star.json", b_cqr)
    _write_json(OUT / "cqr.json", c_cqr)
    b_metrics = _metrics(b_pred, b_cqr)
    c_metrics = _metrics(c_pred, c_cqr)
    b_metrics.to_csv(OUT / "metrics_b_star.csv", index=False)
    c_metrics.to_csv(OUT / "metrics_c.csv", index=False)
    comparison = _comparison(b_metrics, c_metrics)
    if not bool(comparison["n_scored_b_star"].eq(comparison["n_scored_c"]).all()):
        raise RuntimeError("B* and C scored-row counts differ")
    comparison.to_csv(OUT / "b_star_vs_c.csv", index=False)
    by_crop, by_market = _h4_breakdown(b_pred, c_pred)
    by_crop.to_csv(OUT / "lightgbm_h4_by_crop.csv", index=False)
    by_market.to_csv(OUT / "lightgbm_h4_by_market.csv", index=False)
    statements = _statements(comparison)
    shock = c_pred[
        c_pred["model"].eq("lightgbm")
        & c_pred["split"].eq("test")
        & c_pred["horizon"].eq(4)
        & c_pred["is_shock"].astype(bool)
    ]
    shock_n = int(len(shock))
    if shock_n == 0:
        statements.append(
            "2024 Q1 shock at horizon 4 is not evaluable: this run has zero Experiment C horizon-4 rows in that window."
        )
    else:
        statements.append(f"2024 Q1 shock horizon-4 Experiment C rows in this run: {shock_n}.")
    manifest = {
        "experiment": "C",
        "description": (
            "Matched-row test. B* uses Experiment B features without current cultivation progress. "
            "C adds real national as-of cultivation progress on those same rows."
        ),
        "spatial_scope": "national",
        "spatial_scope_note": "National cultivation progress is not district supply progress.",
        "district_rows_used_for_scoring": False,
        "features_b_star": features_b,
        "features_c": features_c,
        "categories": spec,
        "label": "target_price_lkr_kg at target_week",
        "model_target": "log(target_price / last_observed_price), inverted back to LKR/kg",
        "quantiles": [0.05, 0.50, 0.95],
        "random_state": RANDOM_STATE,
        "model_version": MODEL_VERSION,
        "hyperparameters": "fixed notebook defaults; n_estimators selected on 2022 targets",
        "hyperparameter_source": "Notebook 1 fixed LightGBM and XGBoost settings, seed 42",
        "optuna_rerun": False,
        "protocol": "notebook_2022_early_stop",
        "early_stopping": (
            "Selector training uses eligible targets through 2021-12-31. "
            "Early stopping uses 2022 target weeks, patience 50, cap 500. "
            "The saved trees are refit through 2022-12-31. 2023 is used only for qhat."
        ),
        "interval_calibration": "qhat is computed on 2023 issue weeks only and applied to 2024-2025 test intervals.",
        "test": "2024-2025 issue weeks",
        "xgboost": "fit with the same notebook protocol as LightGBM",
        "lstm": "not run",
        "eligible_rows": int(len(eligible)),
        "source_path": summary.get("source_path"),
        "statements": statements,
    }
    _write_json(OUT / "manifest.json", manifest)
    print("scoring the dynamic reforecast from the saved C models", flush=True)
    walk_note = write_reforecast(OUT)
    status.update(
        {
            "status": "trained_real_experiment_c",
            "experiment_c_trained": True,
            "final_status": "TRAINED",
            "model_version": MODEL_VERSION,
            "primary_model": "LightGBM quantile",
            "robustness_model": "XGBoost quantile",
            "shock_2024q1_h4_rows": shock_n,
            "calibration": "2023 only",
            "test": "2024-2025 only",
            "comparison": "B* and C use the exact same national-progress rows",
            "statements": statements,
            "lightgbm_mae_change_pct_c_vs_b_star": {
                str(int(row.horizon)): float(row.mae_change_pct) for row in comparison.itertuples(index=False)
            },
            "reforecast": walk_note,
            "interpretation": "A negative MAE change means C has a lower MAE than B* on that horizon. A positive change means C is worse.",
        }
    )
    _write_json(OUT / "status.json", status)
    print(comparison.to_string(index=False), flush=True)
    for line in statements:
        print(line, flush=True)


def main() -> None:
    global OUT
    from htfe.training.publish import begin_staging, discard_staging, publish_staging

    official = OUT
    before = _protected_hashes()
    stage = begin_staging(official.name)
    OUT = stage
    published = False
    try:
        print("auditing national cultivation progress", flush=True)
        attached, summary, bundle, _asof = _prepare()
        print(
            f"national signal rows {summary['asof_rows_with_national_signal']} of {summary['asof_rows']}",
            flush=True,
        )
        _train(attached, summary, bundle)
        publish_staging(stage, official)
        published = True
    finally:
        OUT = official
        if not published:
            discard_staging(stage)
        _assert_unchanged(before)
    print(f"wrote {official / 'status.json'}", flush=True)


if __name__ == "__main__":
    main()
