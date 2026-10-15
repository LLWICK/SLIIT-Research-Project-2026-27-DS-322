"""Read outputs/experiment_c for the studio. Does not train."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from lib.load import call_cached, title_case
from lib.style import section_title

OUT = Path(__file__).resolve().parents[2] / "outputs" / "experiment_c"
EVAL_ROOT = OUT.parent / "evaluation"


def _file_token(path: Path) -> tuple[str, int, int]:
    if not path.is_file():
        return (str(path), 0, 0)
    stat = path.stat()
    return (str(path), int(stat.st_mtime_ns), int(stat.st_size))


def _stamp() -> tuple[tuple[str, int, int], ...]:
    names = (
        "status.json",
        "audit_summary.json",
        "b_star_vs_c.csv",
        "dynamic_reforecast.csv",
        "dynamic_reforecast_note.json",
        "member3_packet.json",
    )
    return tuple(_file_token(OUT / name) for name in names)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _paint() -> int | None:
    try:
        value = st.session_state.get("_paint")
    except Exception:
        return None
    return value if isinstance(value, int) else None


def _reuse(fn: object) -> dict | None:
    paint = _paint()
    pack = getattr(fn, "_pack", None)
    if paint is None or getattr(fn, "_paint", None) != paint or not isinstance(pack, dict):
        return None
    return pack


def _keep(fn: object, pack: dict) -> dict:
    paint = _paint()
    if paint is not None:
        fn._paint = paint  # type: ignore[attr-defined]
        fn._pack = pack  # type: ignore[attr-defined]
    return pack


@st.cache_data(show_spinner=False)
def load_experiment_c(stamp: tuple[tuple[str, int, int], ...]) -> dict:
    del stamp
    status = _read_json(OUT / "status.json")
    audit = _read_json(OUT / "audit_summary.json")
    note = _read_json(OUT / "dynamic_reforecast_note.json")
    packet = _read_json(OUT / "member3_packet.json")
    comparison = pd.read_csv(OUT / "b_star_vs_c.csv") if (OUT / "b_star_vs_c.csv").is_file() else pd.DataFrame()
    walk = pd.read_csv(OUT / "dynamic_reforecast.csv") if (OUT / "dynamic_reforecast.csv").is_file() else pd.DataFrame()
    trained = bool(status.get("experiment_c_trained"))
    return {
        "trained": trained,
        "status": status.get("status"),
        "statements": list(status.get("statements") or []),
        "audit": audit,
        "comparison": comparison,
        "walk": walk,
        "note": note,
        "packet_forecasts": int(len(packet.get("forecasts") or [])),
        "coverage_note": packet.get("coverage_note"),
        "spatial_scope": status.get("spatial_scope") or audit.get("spatial_scope_for_scoring") or "national",
    }


def experiment_c() -> dict:
    saved = _reuse(experiment_c)
    if saved is not None:
        return saved
    pack = call_cached(load_experiment_c, (_stamp(),), "Opening the cultivation scores")
    return _keep(experiment_c, pack)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _parity_stamp() -> tuple[tuple[str, int, int], ...]:
    names = (
        "protocol_difference.json",
        "inference/saved_c_example.json",
        "inference/c_observed_ratio_sensitivity.csv",
        "inference/forecasts_12_c.csv",
        "inference/member3_forecast_c.json",
        "inference/manifest_check.json",
        "evaluation/common_key_b_star_vs_c.csv",
        "evaluation/quantile_crossing_c.csv",
        "evaluation/conditional_picp_c.csv",
        "evaluation/residual_summary_c_h4.csv",
        "evaluation/shock_slice_c_h4.csv",
        "evaluation/shock_slice_note.txt",
        "evaluation/robustness_c_h4_by_crop.csv",
        "evaluation/robustness_c_h4_by_market.csv",
        "evaluation/robustness_c_h4_by_season.csv",
        "evaluation/bstar_vs_c_block_bootstrap.csv",
        "evaluation/bootstrap_note.txt",
        "evaluation/shap_lightgbm_c_h4.csv",
        "evaluation/shap_note.txt",
        "evaluation/final_probabilistic_evaluation_c.csv",
        "evaluation/quantile_crossing_diagnostic_c.csv",
        "evaluation/conditional_interval_evaluation_c.csv",
        "evaluation/conditional_interval_note.txt",
        "inference/sensitivity_note.txt",
    )
    tokens = [_file_token(OUT / name) for name in names]
    extra = (
        "common_key_benchmark.csv",
        "seasonal_naive_benchmark.csv",
        "model_vs_naive_block_bootstrap.csv",
        "best_model_by_horizon.csv",
        "benchmark_note.txt",
        "inference/saved_b_example.json",
    )
    tokens.extend(_file_token(EVAL_ROOT / name) for name in extra)
    return tuple(tokens)


@st.cache_data(show_spinner=False)
def load_parity(stamp: tuple[tuple[str, int, int], ...]) -> dict:
    del stamp
    return {
        "protocol": _read_json(OUT / "protocol_difference.json"),
        "example": _read_json(OUT / "inference" / "saved_c_example.json"),
        "sensitivity": _read_csv(OUT / "inference" / "c_observed_ratio_sensitivity.csv"),
        "sensitivity_note": _read_text(OUT / "inference" / "sensitivity_note.txt"),
        "forecasts": _read_csv(OUT / "inference" / "forecasts_12_c.csv"),
        "member3": _read_json(OUT / "inference" / "member3_forecast_c.json"),
        "manifest": _read_json(OUT / "inference" / "manifest_check.json"),
        "common": _read_csv(OUT / "evaluation" / "common_key_b_star_vs_c.csv"),
        "crossing": _read_csv(OUT / "evaluation" / "quantile_crossing_c.csv"),
        "conditional": _read_csv(OUT / "evaluation" / "conditional_picp_c.csv"),
        "residual": _read_csv(OUT / "evaluation" / "residual_summary_c_h4.csv"),
        "shock": _read_csv(OUT / "evaluation" / "shock_slice_c_h4.csv"),
        "shock_note": _read_text(OUT / "evaluation" / "shock_slice_note.txt"),
        "by_crop": _read_csv(OUT / "evaluation" / "robustness_c_h4_by_crop.csv"),
        "by_market": _read_csv(OUT / "evaluation" / "robustness_c_h4_by_market.csv"),
        "by_season": _read_csv(OUT / "evaluation" / "robustness_c_h4_by_season.csv"),
        "bootstrap": _read_csv(OUT / "evaluation" / "bstar_vs_c_block_bootstrap.csv"),
        "bootstrap_note": _read_text(OUT / "evaluation" / "bootstrap_note.txt"),
        "shap": _read_csv(OUT / "evaluation" / "shap_lightgbm_c_h4.csv"),
        "shap_note": _read_text(OUT / "evaluation" / "shap_note.txt"),
        "benchmark": _read_csv(EVAL_ROOT / "common_key_benchmark.csv"),
        "seasonal": _read_csv(EVAL_ROOT / "seasonal_naive_benchmark.csv"),
        "naive_bootstrap": _read_csv(EVAL_ROOT / "model_vs_naive_block_bootstrap.csv"),
        "benchmark_note": _read_text(EVAL_ROOT / "benchmark_note.txt"),
        "best": _read_csv(EVAL_ROOT / "best_model_by_horizon.csv"),
        "probabilistic": _read_csv(OUT / "evaluation" / "final_probabilistic_evaluation_c.csv"),
        "crossing_diagnostic": _read_csv(OUT / "evaluation" / "quantile_crossing_diagnostic_c.csv"),
        "conditional_interval": _read_csv(OUT / "evaluation" / "conditional_interval_evaluation_c.csv"),
        "conditional_note": _read_text(OUT / "evaluation" / "conditional_interval_note.txt"),
        "b_example": _read_json(EVAL_ROOT / "inference" / "saved_b_example.json"),
    }


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _true_flag(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def parity() -> dict:
    saved = _reuse(parity)
    if saved is not None:
        return saved
    pack = call_cached(load_parity, (_parity_stamp(),), "Opening the score tables")
    return _keep(parity, pack)


def render_cultivation_audit(*, compact: bool = False) -> None:
    from lib.cultivation_status import load_cultivation_status

    file_status = load_cultivation_status()
    run = experiment_c()
    audit = run.get("audit") or {}
    if compact:
        if run.get("trained"):
            signal = audit.get("asof_rows_with_national_signal")
            total = audit.get("asof_rows")
            st.caption(
                "National cultivation progress is scored only in Experiment C"
                + (f" ({signal:,} of {total:,} crop-market weeks)." if signal is not None and total is not None else ".")
                + " District rows stay in the audit. This page is the runtime model."
            )
        else:
            st.caption(file_status["headline"])
        return

    with st.container(border=True):
        section_title("Cultivation file", "National rows are scored. District rows stay in the audit.")
        if not file_status.get("doa_present"):
            st.caption(file_status.get("headline") or "The cultivation file is not on disk.")
        else:
            for sentence in file_status["sentences"]:
                st.markdown(sentence)
        counts = file_status.get("counts") or []
        if counts:
            st.dataframe(pd.DataFrame(counts), width="stretch", hide_index=True)
        if audit:
            st.markdown(
                f"As-of coverage from this repository: **{audit.get('asof_rows_with_national_signal', 0):,}** "
                f"of **{audit.get('asof_rows', 0):,}** crop-market weeks have a national report. "
                "National progress is not district supply progress."
            )
            yearly = pd.DataFrame(audit.get("coverage_by_issue_year") or [])
            if not yearly.empty:
                st.dataframe(yearly, width="stretch", hide_index=True)
        if run.get("trained"):
            st.caption("Matched-row B* and C scores are on Cultivation proof. They are not in the full-sample A/B table.")
        else:
            st.caption("outputs/experiment_c does not contain a trained run.")


def render_experiment_c_scores() -> None:
    run = experiment_c()
    with st.container(border=True):
        section_title("Experiment C", "Matched-row national cultivation test")
        st.caption(
            "B* and C use the same eligible rows. B* does not see current progress. "
            "C adds the national as-of ratio. These rows are not in the full-sample A/B table. "
            "District rows are not features. The dashboard is the runtime model."
        )
        if not run.get("trained"):
            st.caption("Experiment C has not been trained in this run. No score from a notebook is shown here.")
            return
        comparison = run["comparison"]
        if comparison.empty:
            st.caption("outputs/experiment_c/b_star_vs_c.csv is missing.")
            return
        show = comparison.copy()
        if "model" in show.columns:
            show = show.rename(columns={"model": "Model"})
        show = show.rename(
            columns={
                "horizon": "Horizon",
                "n_scored_c": "Rows",
                "mae_b_star": "MAE B*",
                "mae_c": "MAE C",
                "mape_b_star": "MAPE B* %",
                "mape_c": "MAPE C %",
                "mae_change_pct": "MAE change %",
                "picp_cqr_b_star": "PICP B* %",
                "picp_cqr_c": "PICP C %",
                "interval_width_cqr_c": "Width C",
            }
        )
        columns = [
            column
            for column in (
                "Model",
                "Horizon",
                "Rows",
                "MAE B*",
                "MAE C",
                "MAPE B* %",
                "MAPE C %",
                "MAE change %",
                "PICP B* %",
                "PICP C %",
                "Width C",
            )
            if column in show.columns
        ]
        st.dataframe(show[columns], width="stretch", hide_index=True)
        st.caption(
            "MAE is LKR per kg. MAPE is a percent. "
            "A positive MAE change means C is less accurate than B* on that horizon. "
            "PICP is the calibrated interval. A share below 90% is below the intended level. "
            "LightGBM is the primary comparison."
        )
        lightgbm = comparison[comparison["model"].eq("lightgbm")] if "model" in comparison.columns else comparison
        for row in lightgbm.itertuples(index=False):
            direction = "lower" if float(row.mae_change_pct) < 0 else "higher"
            level = "at or above 90%" if float(row.picp_cqr_c) >= 90 else "below 90%"
            st.markdown(
                f"LightGBM horizon {int(row.horizon)}: B* MAE {float(row.mae_b_star):.2f}, "
                f"C MAE {float(row.mae_c):.2f}, MAPE {float(row.mape_b_star):.2f}% to {float(row.mape_c):.2f}%, "
                f"n={int(row.n_scored_c)}. C MAE is {direction} by {abs(float(row.mae_change_pct)):.2f}%. "
                f"Calibrated PICP {float(row.picp_cqr_c):.2f}%, {level}."
            )
        if run.get("coverage_note"):
            st.caption(run["coverage_note"])


def render_reforecast() -> None:
    run = experiment_c()
    with st.container(border=True):
        section_title("Dynamic reforecast", "One target week, earlier issue dates, saved C model")
        if not run.get("trained"):
            st.caption("The saved C model is not on disk, so this panel has no reforecast rows.")
            return
        note = run.get("note") or {}
        st.markdown(note.get("note") or "No reforecast note was written.")
        walk = run.get("walk")
        if walk is None or walk.empty:
            st.caption("No real rows were scored for the reforecast.")
            return
        label = (
            f"{note.get('crop', '')} · {note.get('market', '')} · target week {note.get('target_week', '')}"
        )
        st.caption(label + " · national progress · not a fixed example")
        show = walk.rename(
            columns={
                "forecast_issue_week": "Issue date",
                "horizon": "Horizon",
                "cultivation_report_date": "Report date",
                "progress": "Progress",
                "lower": "Lower",
                "median": "Median",
                "upper": "Upper",
                "actual_price": "Actual",
            }
        )
        st.dataframe(
            show[["Issue date", "Horizon", "Report date", "Progress", "Lower", "Median", "Upper", "Actual"]],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Lower, median, and upper come from the saved horizon model plus the 2023 interval adjustment. "
            "Progress is achieved hectares divided by target hectares and is left above 1 when the source ratio is above 1."
        )


def render_protocol_note() -> None:
    manifest_path = OUT.parent / "experiment_a" / "manifest.json"
    manifest = _read_json(manifest_path)
    split = manifest.get("split") or {}
    with st.container(border=True):
        section_title("Tree calendar", "The fit that produced these cards")
        if manifest.get("protocol") == "notebook_2022_early_stop" or split.get("selector_validation"):
            st.markdown(
                "This run fits on eligible targets through 2021-12-31, early-stops on 2022 target weeks, "
                "refits through 2022-12-31, and uses 2023 only to set qhat. "
                "LightGBM and XGBoost use the notebook's fixed hyperparameters. Optuna is not run. "
                "The seed is 42. Quantiles are 0.05, 0.50, and 0.95."
            )
            return
        note = parity().get("protocol") or {}
        sentences = list(note.get("sentences") or [])
        if not sentences:
            st.caption("The protocol note has not been written yet.")
            return
        for sentence in sentences:
            st.markdown(sentence)


def render_member3_handoff() -> None:
    packet = parity().get("member3") or {}
    with st.container(border=True):
        section_title("Given to Member 3", "Experiment C forecast packet from this run")
        if not packet:
            st.caption("The Member 3 forecast file is not on disk.")
            return
        measured = packet.get("measured_picp_pct_by_horizon") or {}
        if not measured:
            st.caption("The Member 3 file has no measured PICP values.")
        else:
            rows = []
            below = []
            for key, value in measured.items():
                number = float(value)
                note = "below the intended level of 90" if number < 90 else ""
                rows.append({"horizon weeks": int(key), "measured PICP %": f"{number:.2f}", "note": note})
                if number < 90:
                    below.append(f"Horizon {int(key)} is {number:.2f}")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            if below:
                st.markdown(
                    "The intended level is 90. " + ". ".join(below) + ". That is below the intended level."
                )
            note = packet.get("coverage_level_note")
            if note:
                st.caption(note)
        forecasts = packet.get("forecasts") or []
        if not forecasts:
            st.caption("The Member 3 file has no forecast rows.")
            return
        show = pd.DataFrame(forecasts)
        display = pd.DataFrame(
            {
                "crop": show["crop"].map(title_case) if "crop" in show.columns else [],
                "market": show["market"].map(title_case) if "market" in show.columns else [],
                "horizon weeks": show["horizon_weeks"] if "horizon_weeks" in show.columns else [],
                "issue week": show["forecast_issue_week"] if "forecast_issue_week" in show.columns else [],
                "forecast week": show["forecast_week"] if "forecast_week" in show.columns else [],
                "lower": show["lower_price"] if "lower_price" in show.columns else [],
                "predicted": show["predicted_price"] if "predicted_price" in show.columns else [],
                "upper": show["upper_price"] if "upper_price" in show.columns else [],
                "progress": show["cultivation_progress"].map(lambda value: f"{float(value):.3f}")
                if "cultivation_progress" in show.columns
                else [],
                "report date": show["cultivation_report_date"] if "cultivation_report_date" in show.columns else [],
            }
        )
        st.dataframe(display, width="stretch", hide_index=True)
        st.caption(
            "These rows are the forecast fields in the file given to Member 3. "
            "They are not the dashboard runtime price. Progress is national achieved hectares divided by target hectares."
        )


def render_saved_model_use() -> None:
    pack = parity()
    example = pack.get("example") or {}
    with st.container(border=True):
        section_title("Saved model, no refit", "One horizon-4 forecast reproduced from the frozen C model")
        if not example:
            st.caption("The reproduction check has not been written yet.")
            return
        st.markdown(
            f"{example.get('crop')} at {example.get('market')}, issued {example.get('forecast_issue_week')}, "
            f"target week {example.get('target_week')}. "
            f"Frozen median {example.get('predicted_price')} LKR/kg. "
            f"Absolute difference from the saved prediction file: {example.get('reproduction_absolute_difference')}."
        )
        st.caption(example.get("interval_rule") or "The median comes from the saved model. No fit step was run.")
        sensitivity = pack.get("sensitivity")
        if sensitivity is not None and not sensitivity.empty:
            st.markdown(pack.get("sensitivity_note") or "Sensitivity uses observed national ratios on this same vector.")
            st.dataframe(sensitivity, width="stretch", hide_index=True)
            st.caption(
                "The three observed-ratio rows are a mechanism check. They are not held-out accuracy. "
                "A ratio above 1 would be kept, and the target-exceeded flag would turn on only above 1."
            )
    with st.container(border=True):
        section_title("Twelve one-week forecasts", "3 crops × 4 score markets, frozen horizon-1 model")
        forecasts = pack.get("forecasts")
        if forecasts is None or forecasts.empty:
            st.caption("The twelve-forecast file has not been written yet.")
        else:
            show = forecasts[
                [
                    "crop",
                    "market",
                    "forecast_issue_week",
                    "forecast_week",
                    "cultivation_progress",
                    "lower_price",
                    "predicted_price",
                    "upper_price",
                    "actual_price_historical_demo",
                ]
            ]
            st.dataframe(show, width="stretch", hide_index=True)
            st.caption(
                "These rows include a historical actual for the reproduction check. "
                "The packet given to Member 3, including measured PICP, is in Given to Member 3 above."
            )
        manifest = pack.get("manifest") or {}
        missing = manifest.get("missing_required") or []
        if manifest:
            if missing:
                st.caption("Missing required artifacts: " + ", ".join(missing))
            else:
                st.caption(
                    f"Manifest check: {manifest.get('required_count', 0)} required artifacts are on disk. "
                    "The check does not import the notebooks and does not fit a model."
                )


def render_c_diagnostics() -> None:
    pack = parity()
    common = pack.get("common")
    if common is None or common.empty:
        return
    with st.container(border=True):
        section_title("Common keys", "B* and C on the same test rows")
        st.dataframe(common, width="stretch", hide_index=True)
        lower = common.loc[common["mae_c"] < common["mae_b_star"], "horizon"].tolist()
        st.caption(
            "MAE is LKR per kg. MAPE is a percent. Pinball is the median pinball loss, also in LKR per kg. "
            "These columns are not interchangeable. "
            "A higher C MAE means C is less accurate than B* on that horizon. "
            "Horizons where C MAE is lower on this file: "
            + (", ".join(str(int(item)) for item in lower) if lower else "none")
            + "."
        )
    with st.container(border=True):
        section_title("Quantile crossing", "Frozen models scored again, without fitting")
        crossing = pack.get("crossing")
        if crossing is not None and not crossing.empty:
            st.dataframe(crossing, width="stretch", hide_index=True)
            st.caption(
                "Raw crossing counts come from calling the saved quantile models. "
                "The prediction file stores ordered 5% and 95% tails, so a stored tail swap of zero is expected. "
                "A stored median outside those tails is a real row in the file."
            )
    with st.container(border=True):
        section_title("Conditional coverage", "Calibrated PICP by crop, market, and season")
        conditional = pack.get("conditional")
        if conditional is not None and not conditional.empty:
            horizon_4 = conditional[conditional["horizon"].eq(4)]
            st.dataframe(horizon_4, width="stretch", hide_index=True)
            st.caption(
                "Horizon 4 is shown here. PICP is the share of test prices inside the interval after the 2023 adjustment. "
                "The nominal level is 90%. Groups below 90% are in the table. "
                "This is not the full-sample Experiment B result. The full-sample horizon-4 calibrated PICP is on the Results page."
            )
            with st.expander("All horizons"):
                st.dataframe(conditional, width="stretch", hide_index=True)
    with st.container(border=True):
        section_title("Horizon 4 residuals and slices", "Bias, crop, market, season, and the 2024 Q1 window")
        residual = pack.get("residual")
        if residual is not None and not residual.empty:
            st.dataframe(residual, width="stretch", hide_index=True)
            st.caption("Residual is actual price minus predicted price. A negative mean means the median forecast is high on average.")
        for title, frame in (
            ("By crop", pack.get("by_crop")),
            ("By market", pack.get("by_market")),
            ("By season", pack.get("by_season")),
        ):
            st.markdown(f"**{title}**")
            if frame is not None and not frame.empty:
                st.dataframe(frame, width="stretch", hide_index=True)
        shock_note = pack.get("shock_note") or ""
        if shock_note:
            st.markdown(shock_note)
        shock = pack.get("shock")
        if shock is not None and not shock.empty:
            st.dataframe(shock, width="stretch", hide_index=True)
    with st.container(border=True):
        section_title("Paired bootstrap", "Uncertainty in the C minus B* absolute error")
        bootstrap = pack.get("bootstrap")
        if bootstrap is not None and not bootstrap.empty:
            st.dataframe(bootstrap, width="stretch", hide_index=True)
        note = pack.get("bootstrap_note") or ""
        if note:
            st.caption(note)
        if bootstrap is not None and not bootstrap.empty:
            above_flag = _true_flag(bootstrap["ci_entirely_above_zero"])
            below_flag = _true_flag(bootstrap["ci_entirely_below_zero"])
            above = bootstrap.loc[above_flag, "horizon"].tolist()
            below = bootstrap.loc[below_flag, "horizon"].tolist()
            crossed = bootstrap.loc[~above_flag & ~below_flag, "horizon"].tolist()
            st.caption(
                "Intervals entirely above zero: horizons "
                + (", ".join(str(int(item)) for item in above) if above else "none")
                + ". Intervals entirely below zero: horizons "
                + (", ".join(str(int(item)) for item in below) if below else "none")
                + ". Intervals that include zero: horizons "
                + (", ".join(str(int(item)) for item in crossed) if crossed else "none")
                + ". A positive difference means C's absolute error is higher. "
                "This does not replace the MAE and MAPE columns."
            )
    shap = pack.get("shap")
    if shap is not None and not shap.empty:
        with st.container(border=True):
            section_title("SHAP", "Frozen horizon-4 median, supporting only")
            st.dataframe(shap.head(12), width="stretch", hide_index=True)
            st.caption(pack.get("shap_note") or "SHAP is not the B* versus C test.")
    probabilistic = pack.get("probabilistic")
    if probabilistic is not None and not probabilistic.empty:
        with st.container(border=True):
            section_title("Interval quality, this run", "Pinball, PICP, width, and interval score")
            show = probabilistic[
                [
                    "horizon",
                    "n_scored",
                    "mae",
                    "mape",
                    "pinball_mean_3q",
                    "picp_cqr_pct_saved_file",
                    "picp_cqr_pct",
                    "interval_width_cqr",
                    "interval_score_90",
                ]
            ]
            st.dataframe(show, width="stretch", hide_index=True)
            st.caption(
                "MAE is LKR per kg and MAPE is a percent. "
                "picp_cqr_pct_saved_file is the calibrated coverage already stored for this matched-row run. "
                "picp_cqr_pct applies the same saved qhat after ordering the tails around q50. "
                "A value below 90 is below the intended level. These rows do not replace the full-sample Experiment B card."
            )
    conditional = pack.get("conditional_interval")
    if conditional is not None and not conditional.empty:
        with st.container(border=True):
            section_title("Conditional coverage, this run", "Crop, market, season, and quarter")
            horizon_4 = conditional[conditional["horizon"].eq(4)]
            st.dataframe(horizon_4, width="stretch", hide_index=True)
            st.caption(pack.get("conditional_note") or "Groups with fewer than 10 rows are omitted.")
    diagnostic = pack.get("crossing_diagnostic")
    if diagnostic is not None and not diagnostic.empty:
        with st.expander("Quantile order on the saved C predictions"):
            st.dataframe(diagnostic, width="stretch", hide_index=True)


def render_common_key_benchmark() -> None:
    pack = parity()
    benchmark = pack.get("benchmark")
    if benchmark is None or benchmark.empty:
        return
    columns = [
        "model",
        "horizon",
        "n_scored",
        "mae",
        "mape",
        "smape",
        "wape",
        "mean_bias",
        "r2",
        "mase_52w",
        "mae_skill_vs_naive_pct",
    ]
    show_cols = [column for column in columns if column in benchmark.columns]
    if not show_cols:
        st.caption("The common-key benchmark table is missing its score columns.")
        return
    with st.container(border=True):
        section_title("Common-key benchmark", "Same crop, market, issue week, and horizon")
        st.dataframe(benchmark[show_cols], width="stretch", hide_index=True)
        st.caption(
            pack.get("benchmark_note")
            or "This table is the identical-key subset of the saved Experiment B predictions. It does not replace the full-sample cards above."
        )
        seasonal = pack.get("seasonal")
        naive = pack.get("naive_bootstrap")
        if seasonal is not None and not seasonal.empty:
            with st.expander("Seasonal naive on the rows that have a 52-week lag"):
                st.dataframe(seasonal, width="stretch", hide_index=True)
        if naive is not None and not naive.empty:
            with st.expander("Block bootstrap of model error minus last-value error"):
                st.dataframe(naive, width="stretch", hide_index=True)
                st.caption(
                    "A negative mean difference means the fitted model has lower absolute error than the last observed price. "
                    "This is not the C minus B* bootstrap."
                )


def render_saved_b() -> None:
    example = parity().get("b_example") or {}
    with st.container(border=True):
        section_title("Saved Experiment B, no refit", "One frozen forecast")
        if not example:
            st.caption("The Experiment B reproduction check has not been written yet.")
            return
        st.markdown(
            f"{example.get('crop')} at {example.get('market')}, horizon {example.get('horizon_weeks')}, "
            f"issued {example.get('forecast_issue_week')}, target week {example.get('target_week')}. "
            f"Frozen median {example.get('predicted_price')} LKR/kg. "
            f"Absolute difference from the saved prediction file: {example.get('reproduction_absolute_difference')}."
        )
        st.caption(example.get("note") or "The median comes from the saved model. No fit step was run.")
