"""Evaluation computed from the saved Experiment C prediction files.

Pinball, MAE, and MAPE on the matched rows are already in metrics_c.csv.
This module adds the checks that were only in the notebooks: common-key
comparison, raw quantile crossing, conditional PICP, residual bias, the
2024 Q1 shock slice, horizon-4 breakdowns, and a moving-block bootstrap.
It does not refit models.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from htfe.config import STUDIO_DATA
from htfe.data.build.scope import SHOCK_END, SHOCK_START
from htfe.training.evaluate import (
    _block,
    interval_score_90,
    normalize_key,
    pinball_loss,
    project_interval_around_median,
)
from htfe.training.saved_inference import (
    OUT,
    REPRODUCTION_TOLERANCE,
    SCORE_MARKETS,
    design_matrix,
    load_bundles,
    load_json,
    predict_prices,
    raw_quantile_crossed,
    write_json,
)

EVAL = OUT / "evaluation"
KEY = ["crop_key", "market_key", "issue", "target", "horizon"]


def moving_block_ci(values: np.ndarray | pd.Series, block_length: int = 4, reps: int = 2000, seed: int = 42) -> tuple[float, float]:
    """95% interval for the mean, resampling contiguous blocks with wrap-around."""
    series = np.asarray(values, dtype=float)
    size = len(series)
    if size == 0:
        return (float("nan"), float("nan"))
    width = int(max(1, min(int(block_length), size)))
    generator = np.random.default_rng(seed)
    blocks = int(np.ceil(size / width))
    samples = np.empty(reps, dtype=float)
    offsets = np.arange(width)
    for rep in range(reps):
        starts = generator.integers(0, size, size=blocks)
        indices = np.concatenate([(start + offsets) % size for start in starts])[:size]
        samples[rep] = float(np.mean(series[indices]))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _with_keys(frame: pd.DataFrame, crop: str, market: str, horizon: str) -> pd.DataFrame:
    out = frame.copy()
    out["crop_key"] = out[crop].astype(str).str.strip().str.lower()
    out["market_key"] = (
        out[market].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False)
    )
    out["issue"] = pd.to_datetime(out["forecast_issue_week"], format="mixed").dt.strftime("%Y-%m-%d")
    out["target"] = pd.to_datetime(out["target_week"], format="mixed").dt.strftime("%Y-%m-%d")
    out["horizon"] = out[horizon].astype(int)
    return out


def _test_lightgbm(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["model"].astype(str).str.lower().eq("lightgbm") & frame["split"].astype(str).str.lower().eq("test")
    ].copy()


def _unique_keys(frame: pd.DataFrame, label: str) -> None:
    if frame.duplicated(KEY).any():
        raise RuntimeError(f"{label} has duplicate crop, market, issue, target, and horizon keys.")


def common_key_comparison(b_star: pd.DataFrame, experiment_c: pd.DataFrame) -> pd.DataFrame:
    left = _test_lightgbm(b_star)
    right = _test_lightgbm(experiment_c)
    left = left[left["feature_set"].astype(str).eq("B_star")]
    right = right[right["feature_set"].astype(str).eq("C")]
    _unique_keys(left, "B*")
    _unique_keys(right, "C")
    pair = left[KEY + ["y_true", "y_pred", "y_pred_q05", "y_pred_q95"]].merge(
        right[KEY + ["y_true", "y_pred", "y_pred_q05", "y_pred_q95"]],
        on=KEY,
        how="outer",
        suffixes=("_b_star", "_c"),
        indicator=True,
    )
    rows = []
    for horizon, group in pair.groupby("horizon", sort=True):
        both = group[group["_merge"].eq("both")]
        if not both.empty and not np.allclose(both["y_true_b_star"], both["y_true_c"], rtol=0, atol=1e-6):
            raise RuntimeError(f"B* and C actual prices differ at horizon {int(horizon)}.")
        b_metrics = _block(
            both["y_true_b_star"].to_numpy(dtype=float),
            both["y_pred_b_star"].to_numpy(dtype=float),
            both["y_pred_q05_b_star"].to_numpy(dtype=float),
            both["y_pred_q95_b_star"].to_numpy(dtype=float),
        )
        c_metrics = _block(
            both["y_true_c"].to_numpy(dtype=float),
            both["y_pred_c"].to_numpy(dtype=float),
            both["y_pred_q05_c"].to_numpy(dtype=float),
            both["y_pred_q95_c"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "horizon": int(horizon),
                "n_common": int(len(both)),
                "n_only_b_star": int(group["_merge"].eq("left_only").sum()),
                "n_only_c": int(group["_merge"].eq("right_only").sum()),
                "mae_b_star": b_metrics["mae"],
                "mae_c": c_metrics["mae"],
                "mape_b_star": b_metrics["mape"],
                "mape_c": c_metrics["mape"],
                "pinball_b_star": b_metrics["pinball"],
                "pinball_c": c_metrics["pinball"],
            }
        )
    return pd.DataFrame(rows)


def _point_row(frame: pd.DataFrame, **labels) -> dict:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    error = y_pred - y_true
    absolute = np.abs(error)
    populated = len(frame) > 0
    return {
        **labels,
        "n": int(len(frame)),
        "mae": round(float(np.mean(absolute)), 2) if populated else None,
        "rmse": round(float(np.sqrt(np.mean(error**2))), 2) if populated else None,
        "mape": round(float(np.mean(absolute / np.clip(np.abs(y_true), 1e-6, None)) * 100), 2) if populated else None,
        "mean_bias_pred_minus_actual": round(float(np.mean(error)), 2) if populated else None,
    }


def grouped_point_metrics(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = [_point_row(part, **{column: value}) for value, part in frame.groupby(column, dropna=False)]
    return pd.DataFrame(rows)


def shock_slice(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """2024 Q1 issue weeks. An empty slice is reported as not evaluable."""
    stamps = pd.to_datetime(frame["forecast_issue_week"], format="mixed")
    window = stamps.between(SHOCK_START, SHOCK_END, inclusive="both")
    shock = frame.loc[window].copy()
    outside_n = int((~window).sum())
    if shock.empty:
        table = pd.DataFrame(
            [
                {
                    "shock_status": "not_evaluable_no_real_C_rows_in_2024_Q1",
                    "n": 0,
                    "mae": None,
                    "rmse": None,
                    "mape": None,
                    "mean_bias_pred_minus_actual": None,
                }
            ]
        )
        note = (
            f"The 2024 Q1 shock slice is not evaluable for LightGBM C horizon 4. "
            f"No matched national-progress test row has an issue week from {SHOCK_START} to {SHOCK_END}. "
            f"Rows outside that window: {outside_n}."
        )
        return table, note
    shock["shock_status"] = "2024_Q1_shock"
    outside = frame.loc[~window].copy()
    outside["shock_status"] = "outside_shock"
    table = pd.concat(
        [grouped_point_metrics(shock, "shock_status"), grouped_point_metrics(outside, "shock_status")],
        ignore_index=True,
    )
    note = f"LightGBM C horizon 4 has {len(shock)} test rows with issue weeks in 2024 Q1."
    return table, note


def conditional_picp(predictions: pd.DataFrame, qhats: dict[str, float]) -> pd.DataFrame:
    work = _test_lightgbm(predictions)
    work = work[work["feature_set"].astype(str).eq("C")]
    rows = []
    scored = []
    for horizon, part in work.groupby("horizon", sort=True):
        qhat = float(qhats[str(int(horizon))])
        block = part.copy()
        low = np.minimum(block["y_pred_q05"].to_numpy(dtype=float), block["y_pred_q95"].to_numpy(dtype=float))
        high = np.maximum(block["y_pred_q05"].to_numpy(dtype=float), block["y_pred_q95"].to_numpy(dtype=float))
        block["lower_cqr"] = low - qhat
        block["upper_cqr"] = high + qhat
        scored.append(block)
    scored_frame = pd.concat(scored, ignore_index=True)
    for column, label in (("crop_key", "crop"), ("market_key", "market"), ("season", "season")):
        grouped = scored_frame.groupby(["model", "horizon", column], dropna=False)
        for (model, horizon, group_value), part in grouped:
            y_true = part["y_true"].to_numpy(dtype=float)
            lower = part["lower_cqr"].to_numpy(dtype=float)
            upper = part["upper_cqr"].to_numpy(dtype=float)
            covered = (y_true >= lower) & (y_true <= upper)
            rows.append(
                {
                    "dimension": label,
                    "group": str(group_value),
                    "model": str(model),
                    "horizon": int(horizon),
                    "n": int(len(part)),
                    "picp_cqr_pct": round(float(np.mean(covered) * 100), 2),
                    "interval_width_cqr": round(float(np.mean(upper - lower)), 2),
                }
            )
    return pd.DataFrame(rows)


def residual_summary(frame: pd.DataFrame) -> pd.DataFrame:
    residual = frame["y_true"].to_numpy(dtype=float) - frame["y_pred"].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "model": "lightgbm",
                "feature_set": "C",
                "horizon": 4,
                "n": int(len(frame)),
                "mean_residual_actual_minus_predicted": round(float(np.mean(residual)), 2),
                "median_residual": round(float(np.median(residual)), 2),
                "residual_std": round(float(np.std(residual, ddof=1)), 2),
            }
        ]
    )


def quantile_crossing(predictions: pd.DataFrame, eligible: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Stored tails plus a fresh predict() of the frozen models, with no fit."""
    work = _test_lightgbm(predictions)
    work = work[work["feature_set"].astype(str).eq("C")]
    rows = []
    for horizon, part in work.groupby("horizon", sort=True):
        bundles = load_bundles(OUT / "models", "C", int(horizon))
        feature_columns = list(bundles["q50"]["feature_columns"])
        needed = list(
            dict.fromkeys(
                feature_columns
                + [
                    "crop_id",
                    "market_id",
                    "season",
                    "last_observed_price_lkr_kg",
                    "crop_key",
                    "market_key",
                    "issue",
                    "target",
                    "horizon",
                ]
            )
        )
        source = eligible.loc[eligible["horizon"].eq(int(horizon)), needed]
        if source.duplicated(KEY).any():
            raise RuntimeError(f"Horizon {int(horizon)} eligible table has duplicate keys.")
        overlap = [name for name in source.columns if name in part.columns and name not in KEY]
        pair = part.drop(columns=overlap).merge(source, on=KEY, how="left", validate="many_to_one", indicator=True)
        missed = int(pair["_merge"].ne("both").sum())
        if missed:
            raise RuntimeError(f"Horizon {int(horizon)} has {missed} test rows without an eligible feature row.")
        design = design_matrix(pair, bundles, spec)
        origin = pair["last_observed_price_lkr_kg"].to_numpy(dtype=float)
        prices = predict_prices(design, origin, bundles)
        saved_gap = np.abs(prices["q50"] - pair["y_pred"].to_numpy(dtype=float))
        if float(np.max(saved_gap)) > REPRODUCTION_TOLERANCE:
            raise RuntimeError(
                f"Frozen h{int(horizon)} medians differ from predictions_c.csv by {float(np.max(saved_gap))}."
            )
        stored_low = pair["y_pred_q05"].to_numpy(dtype=float)
        stored_high = pair["y_pred_q95"].to_numpy(dtype=float)
        stored_mid = pair["y_pred"].to_numpy(dtype=float)
        stored_outside = (stored_low > stored_mid) | (stored_mid > stored_high)
        stored_swap = stored_low > stored_high
        raw_cross = np.fromiter(
            (
                raw_quantile_crossed(float(left), float(mid), float(right))
                for left, mid, right in zip(prices["q05"], prices["q50"], prices["q95"], strict=True)
            ),
            dtype=bool,
            count=len(pair),
        )
        rows.append(
            {
                "model": "lightgbm",
                "feature_set": "C",
                "horizon": int(horizon),
                "n_scored": int(len(pair)),
                "max_abs_median_difference_vs_saved": float(np.max(saved_gap)),
                "stored_tail_swap_count": int(stored_swap.sum()),
                "stored_median_outside_tails_count": int(stored_outside.sum()),
                "stored_median_outside_tails_pct": round(float(np.mean(stored_outside) * 100), 2),
                "raw_crossing_count": int(raw_cross.sum()),
                "raw_crossing_rate_pct": round(float(np.mean(raw_cross) * 100), 2),
            }
        )
    return pd.DataFrame(rows)


def block_bootstrap(b_star: pd.DataFrame, experiment_c: pd.DataFrame) -> pd.DataFrame:
    left = _test_lightgbm(b_star)
    right = _test_lightgbm(experiment_c)
    left = left[left["feature_set"].astype(str).eq("B_star")]
    right = right[right["feature_set"].astype(str).eq("C")]
    rows = []
    horizons = sorted(set(left["horizon"]).intersection(set(right["horizon"])))
    for horizon in horizons:
        b_rows = left[left["horizon"].eq(horizon)][KEY + ["y_true", "y_pred"]]
        c_rows = right[right["horizon"].eq(horizon)][KEY + ["y_true", "y_pred"]]
        pair = b_rows.merge(c_rows, on=KEY, suffixes=("_b_star", "_c"), validate="one_to_one")
        pair["ae_difference_c_minus_b_star"] = (pair["y_pred_c"] - pair["y_true_c"]).abs() - (
            pair["y_pred_b_star"] - pair["y_true_b_star"]
        ).abs()
        weekly = pair.groupby("target", as_index=False)["ae_difference_c_minus_b_star"].mean().sort_values("target")
        low, high = moving_block_ci(
            weekly["ae_difference_c_minus_b_star"],
            block_length=4,
            reps=2000,
            seed=42 + int(horizon),
        )
        rows.append(
            {
                "horizon": int(horizon),
                "n_matched_rows": int(len(pair)),
                "n_target_weeks": int(len(weekly)),
                "mean_ae_difference_c_minus_b_star": round(float(pair["ae_difference_c_minus_b_star"].mean()), 4),
                "bootstrap_95_ci_low": round(low, 4),
                "bootstrap_95_ci_high": round(high, 4),
                "ci_entirely_below_zero": bool(high < 0),
                "ci_entirely_above_zero": bool(low > 0),
            }
        )
    return pd.DataFrame(rows)


def protocol_note(comparison: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    evaluation = json.loads((STUDIO_DATA / "evaluation.json").read_text(encoding="utf-8"))
    rows = evaluation["rows"]
    b4 = next(row for row in rows if row["experiment"] == "B" and row["model"] == "lightgbm" and int(row["horizon"]) == 4)
    sarimax = next(row for row in rows if row["experiment"] == "A" and row["model"] == "sarimax" and int(row["horizon"]) == 1)
    mae = float(b4["mae"])
    mape = float(b4["mape"])
    picp = float(b4["picp_cqr"])
    width = float(b4["interval_width_cqr"])
    n_scored = int(b4["n_scored"])
    sarimax_mae = float(sarimax["mae"])
    sarimax_mape = float(sarimax["mape"])
    primary = comparison
    if "model" in comparison.columns:
        light = comparison[comparison["model"].astype(str).str.lower().eq("lightgbm")]
        if not light.empty:
            primary = light
    negative = sorted(int(row.horizon) for row in primary.itertuples(index=False) if float(row.mae_change_pct) < 0)
    higher = sorted(int(row.horizon) for row in primary.itertuples(index=False) if float(row.mae_change_pct) >= 0)
    below = [
        str(int(row.horizon))
        for row in metrics.itertuples(index=False)
        if str(row.model).lower() == "lightgbm" and float(row.picp_cqr) < 90
    ]
    coverage = (
        f"{picp:.2f}%, which is at or above the intended 90%"
        if picp >= 90
        else f"{picp:.2f}%, which is below the intended 90%"
    )
    if negative:
        change_bits = []
        for horizon in negative:
            row = primary.loc[primary["horizon"].eq(horizon)].iloc[0]
            change_bits.append(
                f"horizon {horizon} ({float(row.mae_change_pct):.2f}% MAE, B* {float(row.mae_b_star):.2f}, C {float(row.mae_c):.2f})"
            )
        c_sentence = "On this matched-row LightGBM run, C MAE is lower than B* at " + "; ".join(change_bits) + "."
        if higher:
            c_sentence += " C MAE is not lower at horizon " + ", ".join(str(item) for item in higher) + "."
    else:
        c_sentence = "On this matched-row LightGBM run, C MAE is not lower than B* at any scored horizon."
    sentences = [
        (
            "This run uses the notebook calendar. Selector training uses eligible targets through 2021-12-31. "
            "Early stopping uses 2022 target weeks. The trees are refit through 2022-12-31. "
            "2023 issue weeks set qhat only. Optuna is not used. The seed is 42."
        ),
        (
            f"Full-sample LightGBM B horizon 4 from the studio file: MAE {mae:.2f}, MAPE {mape:.2f}, "
            f"calibrated PICP {coverage}, interval width {width:.2f}, on {n_scored} test rows. "
            f"SARIMAX order is (1, 1, 1) x (1, 0, 0, 52). Horizon 1 MAE is {sarimax_mae:.2f} and MAPE is {sarimax_mape:.2f}."
        ),
        c_sentence + " These rows are not the full-sample A/B table.",
        (
            "Experiment C LightGBM calibrated PICP is below 90% at horizon "
            + (", ".join(below) if below else "none")
            + ". Report the measured share. The nominal interval is 90%."
        ),
    ]
    return {
        "notebook_protocol": "early stopping on 2022 target weeks after training on eligible targets through 2021-12-31; 2023 used only for conformal calibration",
        "repository_protocol": "same notebook calendar: 2021 selector fit, 2022 early stop, refit through 2022, 2023 only for qhat",
        "retrained_for_this_note": True,
        "tree_fit": "htfe.training.run_experiment_a.fit_quantile_trees",
        "notebook_protocol_produced_saved_cards": True,
        "sentences": sentences,
    }


def shap_table(eligible: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame | None, str]:
    try:
        import shap
    except ImportError:
        return None, "SHAP is not installed, so the Experiment C explainability table stays only in the notebook."
    bundle = load_bundles(OUT / "models", "C", 4)["q50"]
    test = eligible[
        eligible["horizon"].eq(4)
        & eligible["split"].astype(str).str.lower().eq("test")
        & eligible["market_key"].isin(SCORE_MARKETS)
    ]
    design = design_matrix(test, {"q50": bundle}, spec)
    sample = design if len(design) <= 400 else design.sample(400, random_state=42)
    try:
        values = shap.TreeExplainer(bundle["model"]).shap_values(sample)
    except Exception as exc:
        return None, f"SHAP did not run on the frozen horizon-4 median model: {exc}"
    mean_abs = np.abs(np.asarray(values)).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    names = list(bundle["columns"])
    rows = [
        {"feature": names[int(index)], "mean_abs_shap": round(float(mean_abs[int(index)]), 6), "rank": rank}
        for rank, index in enumerate(order[:20], start=1)
    ]
    return (
        pd.DataFrame(rows),
        "SHAP mean absolute value on a sample of horizon-4 test rows. This is explainability, not the B* versus C test.",
    )


def probabilistic_and_conditional(predictions: pd.DataFrame, eligible: pd.DataFrame, qhats: dict, metrics: pd.DataFrame) -> dict:
    """Notebook interval diagnostics on the saved C predictions. Does not replace metrics_c.csv."""
    work = predictions.copy()
    work["model"] = work["model"].astype(str).str.lower()
    work = work[work["model"].eq("lightgbm") & work["split"].astype(str).str.lower().eq("test")].copy()
    if "feature_set" in work.columns:
        work = work[work["feature_set"].astype(str).eq("C")]
    for column in ("forecast_issue_week", "target_week"):
        work[column] = pd.to_datetime(work[column], format="mixed").dt.normalize()
    prob_rows = []
    crossing_rows = []
    scored_parts = []
    for horizon, part in work.groupby("horizon", sort=True):
        horizon = int(horizon)
        part = part.copy()
        qhat = float(qhats[str(horizon)])
        raw_q05 = part["y_pred_q05"].to_numpy(dtype=float)
        q50 = part["y_pred"].to_numpy(dtype=float)
        raw_q95 = part["y_pred_q95"].to_numpy(dtype=float)
        raw_cross = (raw_q05 > q50) | (q50 > raw_q95)
        q05, mid, q95 = project_interval_around_median(raw_q05, q50, raw_q95)
        projected_cross = (q05 > mid) | (mid > q95)
        lower = q05 - qhat
        upper = q95 + qhat
        calibrated_cross = (lower > mid) | (mid > upper)
        actual = part["y_true"].to_numpy(dtype=float)
        coverage = float(np.mean((actual >= lower) & (actual <= upper)) * 100)
        saved = metrics[
            metrics["model"].astype(str).str.lower().eq("lightgbm") & metrics["horizon"].astype(int).eq(horizon)
        ]
        if len(saved) != 1:
            raise RuntimeError(f"Expected one saved C metric row for horizon {horizon}.")
        saved_row = saved.iloc[0]
        q05_loss = pinball_loss(actual, q05, 0.05)
        q50_loss = pinball_loss(actual, mid, 0.50)
        q95_loss = pinball_loss(actual, q95, 0.95)
        prob_rows.append(
            {
                "model": "lightgbm",
                "horizon": horizon,
                "n_scored": int(len(part)),
                "mae": float(saved_row["mae"]),
                "mape": float(saved_row["mape"]),
                "pinball_q05": q05_loss,
                "pinball_q50": q50_loss,
                "pinball_q95": q95_loss,
                "pinball_mean_3q": float(np.mean([q05_loss, q50_loss, q95_loss])),
                "median_pinball_q50": q50_loss,
                "picp_cqr_pct_saved_file": float(saved_row["picp_cqr"]),
                "picp_cqr_pct": coverage,
                "coverage_gap_from_90_pp": abs(coverage - 90.0),
                "interval_width_cqr": float(np.mean(upper - lower)),
                "interval_score_90": interval_score_90(actual, lower, upper),
                "saved_qhat": qhat,
            }
        )
        crossing_rows.append(
            {
                "model": "lightgbm",
                "horizon": horizon,
                "n_scored": int(len(part)),
                "raw_crossing_count": int(raw_cross.sum()),
                "raw_crossing_rate_pct": float(np.mean(raw_cross) * 100),
                "projected_order_violation_count": int(projected_cross.sum()),
                "projected_order_violation_rate_pct": float(np.mean(projected_cross) * 100),
                "calibrated_order_violation_count": int(calibrated_cross.sum()),
                "calibrated_order_violation_rate_pct": float(np.mean(calibrated_cross) * 100),
                "status": "CORRECTED_FROM_RAW_CROSSING" if int(raw_cross.sum()) else "PASS_NO_RAW_CROSSING",
            }
        )
        part["lower_cqr"] = lower
        part["upper_cqr"] = upper
        scored_parts.append(part)
    probabilistic = pd.DataFrame(prob_rows)
    crossing = pd.DataFrame(crossing_rows)
    if int(crossing["projected_order_violation_count"].sum()) != 0:
        raise RuntimeError("Projected intervals are still unordered.")
    scored = pd.concat(scored_parts, ignore_index=True)
    scored["crop_key"] = normalize_key(scored["crop"])
    market_column = "market_id" if "market_id" in scored.columns else "market"
    scored["market_key"] = normalize_key(scored[market_column])
    if "cultivation_exact_season_id" not in scored.columns:
        context_source = eligible.copy()
        for column in ("forecast_issue_week", "target_week"):
            context_source[column] = pd.to_datetime(context_source[column], format="mixed").dt.normalize()
        context_source["crop_key"] = normalize_key(context_source["crop_id"])
        context_source["market_key"] = normalize_key(context_source["market_id"])
        context_source["horizon"] = context_source["forecast_horizon_weeks"].astype(int)
        context = context_source[
            ["crop_key", "market_key", "forecast_issue_week", "target_week", "horizon", "cultivation_exact_season_id"]
        ].drop_duplicates(["crop_key", "market_key", "forecast_issue_week", "target_week", "horizon"])
        overlap = [name for name in context.columns if name in scored.columns and name not in {"crop_key", "market_key", "forecast_issue_week", "target_week", "horizon"}]
        scored = scored.drop(columns=overlap).merge(
            context,
            on=["crop_key", "market_key", "forecast_issue_week", "target_week", "horizon"],
            how="left",
            validate="many_to_one",
        )
    scored["season"] = scored["cultivation_exact_season_id"].astype("string").str.extract(r"^(Maha|Yala)\b", expand=False)
    if bool(scored["season"].isna().any()):
        raise RuntimeError("Conditional interval season context is missing. An unknown season bucket is not reported.")
    scored["year_quarter"] = scored["forecast_issue_week"].dt.to_period("Q").astype(str)
    scored["crop"] = scored["crop_key"]
    scored["market"] = scored["market_key"]
    conditional_rows = []
    skipped = 0
    for column, label in (("crop", "crop"), ("market", "market"), ("season", "season"), ("year_quarter", "year_quarter")):
        for (model, horizon, group_value), part in scored.groupby(["model", "horizon", column], dropna=False):
            if len(part) < 10:
                skipped += 1
                continue
            y = part["y_true"].to_numpy(dtype=float)
            lower = part["lower_cqr"].to_numpy(dtype=float)
            upper = part["upper_cqr"].to_numpy(dtype=float)
            coverage = float(np.mean((y >= lower) & (y <= upper)) * 100)
            conditional_rows.append(
                {
                    "dimension": label,
                    "group": str(group_value),
                    "model": str(model),
                    "horizon": int(horizon),
                    "n": int(len(part)),
                    "picp_cqr_pct": coverage,
                    "coverage_gap_from_90_pp": abs(coverage - 90.0),
                    "interval_width_cqr": float(np.mean(upper - lower)),
                    "interval_score_90": interval_score_90(y, lower, upper),
                }
            )
    conditional = pd.DataFrame(conditional_rows).sort_values(["dimension", "horizon", "model", "group"]).reset_index(drop=True)
    return {
        "probabilistic": probabilistic,
        "crossing": crossing,
        "conditional": conditional,
        "groups_skipped_under_10": skipped,
    }


def write_evaluation() -> dict:
    from htfe.training.saved_inference import _prepare

    prepared = _prepare()
    b_star = _with_keys(pd.read_csv(OUT / "predictions_b_star.csv"), "crop", "market_id", "horizon")
    experiment_c = prepared["predictions"]
    eligible = prepared["eligible"]
    metrics = pd.read_csv(OUT / "metrics_c.csv")
    comparison = pd.read_csv(OUT / "b_star_vs_c.csv")
    print("common-key comparison", flush=True)
    common = common_key_comparison(b_star, experiment_c)
    print("quantile crossing from frozen models", flush=True)
    crossing = quantile_crossing(experiment_c, eligible, prepared["spec"])
    print("conditional PICP, residuals, shock slice, bootstrap", flush=True)
    coverage = conditional_picp(experiment_c, prepared["qhats"])
    h4 = _test_lightgbm(experiment_c)
    h4 = h4[h4["feature_set"].astype(str).eq("C") & h4["horizon"].eq(4)].copy()
    by_crop = grouped_point_metrics(h4, "crop_key")
    by_market = grouped_point_metrics(h4, "market_key")
    by_season = grouped_point_metrics(h4, "season")
    shock, shock_note = shock_slice(h4)
    residual = residual_summary(h4)
    bootstrap = block_bootstrap(b_star, experiment_c)
    print("ordered intervals, pinball, and conditional coverage including quarter", flush=True)
    intervals = probabilistic_and_conditional(experiment_c, eligible, prepared["qhats"], metrics)
    note = protocol_note(comparison, metrics)
    print("shap on the frozen horizon-4 median, if the library imports", flush=True)
    shap_rows, shap_note = shap_table(eligible, prepared["spec"])
    EVAL.mkdir(parents=True, exist_ok=True)
    common.to_csv(EVAL / "common_key_b_star_vs_c.csv", index=False)
    crossing.to_csv(EVAL / "quantile_crossing_c.csv", index=False)
    coverage.to_csv(EVAL / "conditional_picp_c.csv", index=False)
    by_crop.to_csv(EVAL / "robustness_c_h4_by_crop.csv", index=False)
    by_market.to_csv(EVAL / "robustness_c_h4_by_market.csv", index=False)
    by_season.to_csv(EVAL / "robustness_c_h4_by_season.csv", index=False)
    shock.to_csv(EVAL / "shock_slice_c_h4.csv", index=False)
    (EVAL / "shock_slice_note.txt").write_text(shock_note, encoding="utf-8")
    residual.to_csv(EVAL / "residual_summary_c_h4.csv", index=False)
    bootstrap.to_csv(EVAL / "bstar_vs_c_block_bootstrap.csv", index=False)
    intervals["probabilistic"].to_csv(EVAL / "final_probabilistic_evaluation_c.csv", index=False)
    intervals["crossing"].to_csv(EVAL / "quantile_crossing_diagnostic_c.csv", index=False)
    intervals["conditional"].to_csv(EVAL / "conditional_interval_evaluation_c.csv", index=False)
    (EVAL / "conditional_interval_note.txt").write_text(
        "Conditional PICP uses the saved 2023 qhat and the ordered interval around the unchanged q50 forecast. "
        "Groups with fewer than 10 test rows are omitted, matching the notebook rule. "
        f"Groups omitted: {intervals['groups_skipped_under_10']}. "
        "Season is the official Maha or Yala label on the DOA source season. "
        "picp_cqr_pct_saved_file in the probabilistic table is the value already stored in metrics_c.csv. "
        "The ordered-interval PICP is reported beside it and does not replace that file. "
        "MAE and MAPE stay in their own columns. Coverage below 90% is reported as measured, not rounded up to 90%.",
        encoding="utf-8",
    )
    (EVAL / "bootstrap_note.txt").write_text(
        "The bootstrap resamples target weeks in blocks of 4. "
        "mean_ae_difference_c_minus_b_star is C absolute error minus B* absolute error on the matched rows. "
        "A negative value means C's absolute error is lower on this run. "
        "The interval is uncertainty in that paired difference. It is not a second accuracy table, "
        "and it does not replace the horizon MAE and MAPE columns.",
        encoding="utf-8",
    )
    write_json(OUT / "protocol_difference.json", note)
    if shap_rows is not None:
        shap_rows.to_csv(EVAL / "shap_lightgbm_c_h4.csv", index=False)
    (EVAL / "shap_note.txt").write_text(shap_note, encoding="utf-8")
    print(common.to_string(index=False), flush=True)
    print(shock_note, flush=True)
    print(shap_note, flush=True)
    return {
        "common_horizons": int(len(common)),
        "shock_note": shock_note,
        "shap_note": shap_note,
        "protocol_sentences": len(note["sentences"]),
        "conditional_groups": int(len(intervals["conditional"])),
        "groups_skipped_under_10": int(intervals["groups_skipped_under_10"]),
    }
