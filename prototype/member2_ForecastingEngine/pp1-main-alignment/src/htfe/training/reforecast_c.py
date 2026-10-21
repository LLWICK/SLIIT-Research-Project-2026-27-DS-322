"""Score one saved Experiment C target week at each earlier issue date.

The walk uses the fitted C models and the real national reports already
attached to those rows. It does not invent prices or cultivation values.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from htfe.training.config import HORIZONS
from htfe.training.run_experiment_a import EVAL_IDS, _design

MODEL_VERSION = "experiment_c_lightgbm_frozen_b_seed42"


def apply_categories(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    out = frame.copy()
    out["crop_id"] = pd.Categorical(out["crop_id"].astype(str), categories=list(spec["crop_id"]))
    out["market_id"] = pd.Categorical(out["market_id"].astype(str), categories=list(spec["market_id"]))
    out["season"] = pd.Categorical(out["season"].astype(str), categories=list(spec["season"]))
    return out


def select_walk(eligible: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pick one test crop-market target week that has real national progress."""
    search = eligible[
        eligible["split"].eq("test")
        & eligible["market_id"].astype(str).isin(EVAL_IDS)
        & eligible["cultivation_progress_ratio_real"].notna()
        & eligible["forecast_horizon_weeks"].isin(HORIZONS)
    ].copy()
    if search.empty:
        return search, {
            "found": False,
            "multiple_real_reports": False,
            "note": "No 2024-2025 test row with a real national progress value was available to reforecast.",
        }
    search["forecast_issue_week"] = pd.to_datetime(search["forecast_issue_week"]).dt.normalize()
    search["target_week"] = pd.to_datetime(search["target_week"]).dt.normalize()
    search["cultivation_report_date"] = pd.to_datetime(search["cultivation_report_date"]).dt.normalize()
    groups = search.groupby(["crop_id", "market_id", "target_week"], as_index=False).agg(
        information_states=("forecast_issue_week", "nunique"),
        distinct_real_reports=("cultivation_report_date", "nunique"),
    )
    multi = groups[groups["information_states"].ge(2) & groups["distinct_real_reports"].ge(2)]
    if len(multi):
        pool = multi
        note = {
            "found": True,
            "multiple_real_reports": True,
            "note": (
                "This target week has at least two issue dates and at least two different "
                "real national report dates. Each row is scored with the horizon model for that issue date."
            ),
        }
    else:
        pool = groups[groups["information_states"].ge(1)]
        note = {
            "found": True,
            "multiple_real_reports": False,
            "note": (
                "No 2024-2025 target week had two different real national report dates. "
                "The rows are the longest real issue-date sequence that could be scored. "
                "They are not a synthetic cultivation update."
            ),
        }
    choice = pool.sort_values(
        ["distinct_real_reports", "information_states", "target_week"],
        ascending=[False, False, True],
    ).iloc[0]
    states = search[
        search["crop_id"].astype(str).eq(str(choice["crop_id"]))
        & search["market_id"].astype(str).eq(str(choice["market_id"]))
        & search["target_week"].eq(pd.Timestamp(choice["target_week"]))
    ].copy()
    states = states.sort_values(["forecast_issue_week", "forecast_horizon_weeks"]).drop_duplicates(
        ["forecast_issue_week", "forecast_horizon_weeks"],
        keep="last",
    )
    note["crop"] = str(choice["crop_id"])
    note["market"] = str(choice["market_id"])
    note["target_week"] = pd.Timestamp(choice["target_week"]).strftime("%Y-%m-%d")
    note["information_states"] = int(choice["information_states"])
    note["distinct_real_reports"] = int(choice["distinct_real_reports"])
    return states.reset_index(drop=True), note


def _bundles(model_dir: Path, feature_set: str, horizon: int) -> dict:
    import joblib

    bundles = {}
    for label in ("q05", "q50", "q95"):
        path = model_dir / f"lightgbm_{feature_set}_h{int(horizon)}_{label}.joblib"
        if not path.is_file():
            raise FileNotFoundError(path)
        bundles[label] = joblib.load(path)
    return bundles


def _predict_prices(states: pd.DataFrame, bundles: dict, spec: dict) -> dict[str, np.ndarray]:
    saved_spec = bundles["q50"].get("categories")
    if saved_spec:
        spec = saved_spec
    frame = apply_categories(states, spec)
    saved = list(bundles["q50"]["columns"])
    design, names = _design(frame, list(bundles["q50"]["feature_columns"]))
    if names != saved:
        raise RuntimeError("reforecast column order does not match the saved C model")
    design = design[saved]
    origin = frame["last_observed_price_lkr_kg"].to_numpy(dtype=float)
    prices = {}
    for label, bundle in bundles.items():
        raw = np.asarray(bundle["model"].predict(design), dtype=float)
        prices[label] = np.maximum(origin * np.exp(raw), 0.01)
    return prices


def score_walk(states: pd.DataFrame, model_dir: Path, qhat_by_horizon: dict, spec: dict) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame()
    rows = []
    for horizon, group in states.groupby("forecast_horizon_weeks", sort=True):
        bundles = _bundles(model_dir, "C", int(horizon))
        prices = _predict_prices(group, bundles, spec)
        qhat = float(qhat_by_horizon[str(int(horizon))])
        low = np.minimum(prices["q05"], prices["q95"])
        high = np.maximum(prices["q05"], prices["q95"])
        lower = low - qhat
        upper = high + qhat
        median = prices["q50"]
        for position, (_, row) in enumerate(group.iterrows()):
            actual = float(row["target_price_lkr_kg"])
            rows.append(
                {
                    "crop": str(row["crop_id"]),
                    "market": str(row["market_id"]),
                    "target_week": pd.Timestamp(row["target_week"]).strftime("%Y-%m-%d"),
                    "forecast_issue_week": pd.Timestamp(row["forecast_issue_week"]).strftime("%Y-%m-%d"),
                    "horizon": int(horizon),
                    "cultivation_report_date": pd.Timestamp(row["cultivation_report_date"]).strftime("%Y-%m-%d"),
                    "progress": float(row["cultivation_progress_ratio_real"]),
                    "target_exceeded": int(float(row["cultivation_target_exceeded_real"])),
                    "lower": float(lower[position]),
                    "median": float(median[position]),
                    "upper": float(upper[position]),
                    "actual_price": actual,
                    "qhat": qhat,
                    "model_version": MODEL_VERSION,
                    "spatial_scope": "national",
                }
            )
    return pd.DataFrame(rows)


def verify_against_saved(walk: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    if walk.empty:
        return {"checked": 0, "max_abs_median_difference": None}
    saved = predictions[
        predictions["model"].eq("lightgbm") & predictions["feature_set"].eq("C")
    ].copy()
    saved["forecast_issue_week"] = pd.to_datetime(saved["forecast_issue_week"]).dt.strftime("%Y-%m-%d")
    saved["target_week"] = pd.to_datetime(saved["target_week"]).dt.strftime("%Y-%m-%d")
    saved["market_key"] = saved["market_id"].astype(str) if "market_id" in saved.columns else saved["market"].astype(str)
    gaps = []
    for row in walk.itertuples(index=False):
        match = saved[
            saved["horizon"].eq(int(row.horizon))
            & saved["crop"].astype(str).str.lower().eq(str(row.crop).lower())
            & saved["market_key"].eq(str(row.market))
            & saved["forecast_issue_week"].eq(row.forecast_issue_week)
            & saved["target_week"].eq(row.target_week)
        ]
        if len(match) != 1:
            raise RuntimeError(
                f"saved C prediction lookup failed for {row.crop}/{row.market}/"
                f"{row.forecast_issue_week}/h{row.horizon}: found {len(match)}"
            )
        gaps.append(abs(float(match["y_pred"].iloc[0]) - float(row.median)))
    worst = float(max(gaps)) if gaps else None
    if worst is not None and worst > 1e-4:
        raise RuntimeError(f"reforecast median differs from saved C predictions by {worst}")
    return {"checked": len(gaps), "max_abs_median_difference": worst}


def _rounded_walk(walk: pd.DataFrame) -> pd.DataFrame:
    if walk.empty:
        return walk
    out = walk.copy()
    out["progress"] = out["progress"].round(6)
    for column in ("lower", "median", "upper", "actual_price"):
        out[column] = out[column].round(2)
    return out


def build_member3_packet(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    qhat_by_horizon: dict,
    walk: pd.DataFrame,
    walk_note: dict,
) -> dict:
    light = metrics[metrics["model"].eq("lightgbm")].copy()
    measured = {
        str(int(row.horizon)): None if pd.isna(row.picp_cqr) else float(row.picp_cqr)
        for row in light.itertuples(index=False)
    }
    below = [horizon for horizon, picp in measured.items() if picp is not None and picp < 90]
    test = predictions[
        predictions["model"].eq("lightgbm")
        & predictions["feature_set"].eq("C")
        & predictions["split"].eq("test")
    ].copy()
    test["forecast_issue_week"] = pd.to_datetime(test["forecast_issue_week"])
    latest = (
        test.sort_values("forecast_issue_week")
        .groupby(["crop", "market_id", "horizon"], as_index=False)
        .tail(1)
    )
    forecasts = []
    for row in latest.itertuples(index=False):
        horizon = str(int(row.horizon))
        qhat = float(qhat_by_horizon[horizon])
        low = min(float(row.y_pred_q05), float(row.y_pred_q95)) - qhat
        high = max(float(row.y_pred_q05), float(row.y_pred_q95)) + qhat
        progress = row.cultivation_progress_ratio_real
        forecasts.append(
            {
                "crop": str(row.crop),
                "market": str(row.market_id),
                "horizon": int(row.horizon),
                "forecast_issue_week": pd.Timestamp(row.forecast_issue_week).strftime("%Y-%m-%d"),
                "target_week": pd.Timestamp(row.target_week).strftime("%Y-%m-%d"),
                "lower": round(low, 2),
                "median": round(float(row.y_pred), 2),
                "upper": round(high, 2),
                "progress": None if pd.isna(progress) else round(float(progress), 6),
                "model_version": MODEL_VERSION,
                "measured_picp": measured.get(horizon),
                "spatial_scope": "national",
            }
        )
    coverage_note = (
        "0.90 is the nominal interval. measured_picp is the share of 2024-2025 test prices "
        "inside the interval after the 2023 adjustment."
    )
    if below:
        coverage_note += " Measured calibrated PICP is below 90% at horizons " + ", ".join(below) + "."
    return {
        "project": "J26-DS-322",
        "from_member": 2,
        "to_member": 3,
        "model_version": MODEL_VERSION,
        "feature_set": "C",
        "spatial_scope": "national",
        "spatial_scope_note": "National cultivation progress is not district supply progress.",
        "nominal_interval": 0.90,
        "coverage_note": coverage_note,
        "measured_picp_pct_by_horizon": measured,
        "forecasts": forecasts,
        "reforecast_note": walk_note.get("note"),
        "reforecast": walk.to_dict(orient="records") if not walk.empty else [],
    }


def write_reforecast(out_dir: Path) -> dict:
    eligible = pd.read_parquet(out_dir / "c_eligible_table.parquet")
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    cqr = json.loads((out_dir / "cqr.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(out_dir / "predictions_c.csv")
    metrics = pd.read_csv(out_dir / "metrics_c.csv")
    spec = manifest["categories"]
    qhats = cqr["qhat_by_model"]["lightgbm"]
    states, note = select_walk(eligible)
    walk = score_walk(states, out_dir / "models", qhats, spec) if note.get("found") else pd.DataFrame()
    check = verify_against_saved(walk, predictions) if not walk.empty else {"checked": 0, "max_abs_median_difference": None}
    displayed = _rounded_walk(walk)
    displayed.to_csv(out_dir / "dynamic_reforecast.csv", index=False)
    packet = build_member3_packet(predictions, metrics, qhats, displayed, note)
    (out_dir / "member3_packet.json").write_text(json.dumps(packet, indent=2), encoding="utf-8")
    (out_dir / "dynamic_reforecast_note.json").write_text(json.dumps({**note, **check}, indent=2), encoding="utf-8")
    return {**note, **check}
