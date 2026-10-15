"""Saved notebook-style figures for the existing studio pages."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from lib.load import call_cached

FIGURES = Path(__file__).resolve().parents[2] / "outputs" / "figures"

PAGES = {
    "Data": (
        ("wholesale_missing_share_by_year.png", "Missing share of wholesale prices by year."),
        ("price_carrot_four_markets.png", "Carrot wholesale price in the four score markets."),
        ("price_carrot_dambulla.png", "Carrot at Dambulla. The grey band is 2020–2021."),
        ("price_leeks_dambulla.png", "Leeks at Dambulla."),
        ("price_tomato_dambulla.png", "Tomato at Dambulla."),
        ("price_season_week_of_year.png", "Seasonal level: mean price by week of year at Dambulla."),
        ("price_week_to_week_change.png", "Week-to-week price movement at Dambulla."),
        ("carrot_dambulla_wholesale_over_retail.png", "Dambulla carrot wholesale divided by retail."),
    ),
    "Features": (
        ("experiment_c_progress_distribution.png", "National as-of progress ratios used by Experiment C. Ratios above 1 are kept."),
        ("shap_lightgbm_c_h4_top_features.png", "LightGBM C, horizon 4, mean absolute SHAP from this run. SHAP does not replace the B* versus C test."),
        ("shap_cultivation_progress_h4.png", "SHAP for the cultivation-progress column on the same horizon-4 model."),
    ),
    "Training": (),
    "Results": (
        ("h4_residuals.png", "Experiment B LightGBM horizon-4 residuals on the 2024–2025 test."),
        ("experiment_a_mae.png", "Experiment A test MAE from this run."),
    ),
    "Comparison": (
        ("experiment_b_vs_a_mae.png", "Full-sample test MAE, Experiment A and Experiment B."),
        ("common_key_benchmark_mae.png", "Fair benchmark on the test rows shared by LightGBM, XGBoost, and SARIMAX."),
        ("final_model_benchmark_mae.png", "Same common-key MAE, including the last-value naive forecast."),
    ),
    "Uncertainty": (
        ("picp_vs_horizon.png", "Experiment B LightGBM calibrated PICP. The dashed line is the 90% intended level."),
        ("width_vs_horizon.png", "Experiment B LightGBM calibrated interval width."),
        ("carrot_dambulla_h4_band.png", "Carrot, Dambulla, horizon 4, ordered interval plus the 2023 qhat."),
        ("final_c_picp_by_horizon.png", "Experiment C LightGBM calibrated PICP on the matched rows."),
        ("final_c_interval_width_by_horizon.png", "Experiment C LightGBM calibrated interval width."),
    ),
    "Novelty": (
        ("experiment_c_bstar_vs_c_mae.png", "Matched-row MAE, LightGBM B* and Experiment C."),
        ("experiment_c_h4_by_crop.png", "Where horizon-4 MAE changes by crop."),
        ("experiment_c_h4_by_market.png", "Where horizon-4 MAE changes by market."),
        ("final_c_h4_predicted_vs_actual.png", "Experiment C LightGBM horizon 4, predicted against actual, matched rows."),
        ("final_c_h4_residual_vs_predicted.png", "Experiment C horizon-4 residuals against the predicted price."),
        ("final_c_h4_residual_histogram.png", "Experiment C horizon-4 residual distribution."),
        ("final_c_h4_residual_over_time.png", "Experiment C horizon-4 mean residual through the test period."),
        ("shap_lightgbm_c_h4_top_features.png", "SHAP support for the horizon-4 C model. The MAE table is the cultivation test."),
    ),
}


def _figure_token(page: str) -> tuple[tuple[str, int, int], ...]:
    rows = []
    for name, _caption in PAGES.get(page, ()):
        path = FIGURES / name
        if path.is_file():
            stat = path.stat()
            rows.append((name, int(stat.st_mtime_ns), int(stat.st_size)))
        else:
            rows.append((name, 0, 0))
    return tuple(rows)


@st.cache_data(show_spinner=False)
def _figure_payload(
    page: str,
    token: tuple[tuple[str, int, int], ...],
    skip: tuple[str, ...] = (),
) -> list[tuple[bytes, str]]:
    del token
    hidden = set(skip)
    payload = []
    for name, caption in PAGES.get(page, ()):
        if name in hidden:
            continue
        path = FIGURES / name
        if path.is_file():
            payload.append((path.read_bytes(), caption))
    return payload


def show_figures(page: str, skip: tuple[str, ...] = ()) -> None:
    if not PAGES.get(page):
        return
    payload = call_cached(
        _figure_payload,
        (page, _figure_token(page), tuple(skip)),
        "Opening the figures",
    )
    if not payload:
        return
    for raw, caption in payload:
        st.image(raw, caption=caption, width="stretch")
