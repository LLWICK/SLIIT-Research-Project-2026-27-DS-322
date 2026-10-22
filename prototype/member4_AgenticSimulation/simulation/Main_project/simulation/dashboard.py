"""Streamlit view of the carrot and leeks agent simulation.

Run from the project root:

    .\\.venv\\Scripts\\python.exe -m streamlit run simulation/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from simulation.agents.population import build_population
from simulation.calibration.baseline_path import N_REPLICATIONS, SLOPE_RATIO, fit_observed, simulate_baseline
from simulation.calibration.compare import compare_with_harti, series_moments
from simulation.config.districts import ADVISORY_CROPS, district_land_ha, district_targets
from simulation.config.settings import BaselineSettings, default_settings
from simulation.experiments.adoption import ADOPTION_RATES, crash_rate, price_volatility
from simulation.market.msrs import MEMBER3_CONFIG, saturation_score
from simulation.scenarios.advisory import run_advisory

CROP_LABEL = {"carrot": "Carrot", "leeks": "Leeks", "tomato": "Tomato"}
CROP_COLOR = {"Carrot": "#E69F00", "Leeks": "#009E73", "Tomato": "#CC79A7"}
SLOPE_CHOICES = (0.05, 0.5, 0.8, 1.0, 1.5)


def _label(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.copy()
    labeled["crop"] = labeled["crop_code"].map(CROP_LABEL)
    return labeled


def _scale() -> alt.Scale:
    present_domain = list(CROP_COLOR)
    return alt.Scale(domain=present_domain, range=[CROP_COLOR[name] for name in present_domain])


def _settings(use_harti_path: bool) -> tuple[BaselineSettings, dict[str, float], dict[str, float]]:
    if not use_harti_path:
        return default_settings(), {}, {}
    settings, processes, _observed = fit_observed()
    persistence = {code: item.persistence for code, item in processes.items() if code in ADVISORY_CROPS}
    shock_std = {code: item.shock_std for code, item in processes.items() if code in ADVISORY_CROPS}
    return settings, persistence, shock_std


@st.cache_data(show_spinner="Reading Colombo wholesale seasons…", ttl=300)
def load_observed() -> pd.DataFrame:
    _settings, _processes, observed = fit_observed()
    return observed


@st.cache_data(show_spinner="Fitting the season-price path…", ttl=300)
def load_processes() -> pd.DataFrame:
    _settings, processes, _observed = fit_observed()
    return pd.DataFrame(
        [
            {
                "crop_code": code,
                "center": item.center,
                "persistence": item.persistence,
                "shock_std": item.shock_std,
            }
            for code, item in processes.items()
        ]
    )


@st.cache_data(show_spinner="Comparing one-step price rules…", ttl=300)
def load_calibration() -> pd.DataFrame:
    rows = []
    for result in compare_with_harti():
        moments = result.moments
        pairs = moments.n_pairs
        rows.append(
            {
                "crop_code": result.crop_code,
                "seasons": moments.n_seasons,
                "pairs": pairs,
                "crashes": moments.crashes,
                "crash_rate": 0.0 if pairs == 0 else moments.crashes / pairs,
                "lag": moments.lag_corr,
                "deseasoned_lag": moments.lag_corr_deseasoned,
                "best_ratio": result.best_ratio,
                "flat_rmse": result.reference_rmse,
                "persist_rmse": result.persist_rmse,
                "cobweb_rmse": result.cobweb_rmse,
                "closest": result.closest,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Running the no-advisory price check…", ttl=300)
def load_price_check() -> pd.DataFrame:
    settings, processes, observed = fit_observed()
    n_seasons = int(observed.loc[observed["crop_code"] == "carrot", "ordinal"].nunique())
    simulated = simulate_baseline(settings, processes, n_seasons=n_seasons)
    rows = []
    for code, process in processes.items():
        crop_rows = observed[observed["crop_code"] == code]
        moments = series_moments(crop_rows, settings.crash_rule.drop_fraction)
        pairs = moments.n_pairs
        runs = simulated[simulated["crop_code"] == code]
        rows.append(
            {
                "crop_code": code,
                "scope": "district planting" if code in ADVISORY_CROPS else "price only",
                "persistence": process.persistence,
                "shock_std": process.shock_std,
                "center": process.center,
                "harti_crash_rate": 0.0 if pairs == 0 else moments.crashes / pairs,
                "simulated_crash_rate": float(runs["crash_rate"].mean()),
                "harti_lag": moments.lag_corr,
                "simulated_lag": float(runs["lag"].mean()),
                "harti_deseasoned_lag": moments.lag_corr_deseasoned,
                "runs": int(len(runs)),
                "seasons": n_seasons,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Running one season path…", ttl=300)
def load_run(
    adoption_rate: float,
    seed: int,
    n_seasons: int,
    n_farmers: int,
    slope_ratio: float,
    use_harti_path: bool,
) -> pd.DataFrame:
    settings, persistence, shock_std = _settings(use_harti_path)
    return run_advisory(
        adoption_rate=adoption_rate,
        seed=seed,
        n_seasons=n_seasons,
        slope_ratio=slope_ratio,
        n_farmers=n_farmers,
        settings=settings,
        persistence=persistence,
        shock_std=shock_std,
        shock_seed=seed,
    )


@st.cache_data(show_spinner="Running the adoption comparison…", ttl=300)
def load_adoption(
    n_seasons: int,
    replications: int,
    n_farmers: int,
    slope_ratio: float,
    use_harti_path: bool,
) -> pd.DataFrame:
    settings, persistence, shock_std = _settings(use_harti_path)
    rows: list[dict[str, object]] = []
    for rate in ADOPTION_RATES:
        for seed in range(replications):
            frame = run_advisory(
                adoption_rate=rate,
                seed=seed,
                n_seasons=n_seasons,
                slope_ratio=slope_ratio,
                n_farmers=n_farmers,
                settings=settings,
                persistence=persistence,
                shock_std=shock_std,
                shock_seed=seed,
            )
            for crop_code, crop_rows in frame.groupby("crop_code", sort=True):
                ordered = crop_rows.sort_values("season_index")
                rows.append(
                    {
                        "adoption_rate": rate,
                        "seed": seed,
                        "crop_code": str(crop_code),
                        "crash_rate": crash_rate(ordered),
                        "price_volatility": price_volatility(ordered),
                        "mean_area_ha": float(ordered["area_ha"].mean()),
                    }
                )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=300)
def load_opening_shock(n_farmers: int, slope_ratio: float, use_harti_path: bool) -> pd.DataFrame:
    """Carrot starts 20% above the price center. No random shock."""

    settings, _persistence, _shock_std = _settings(use_harti_path)
    carrot = settings.crop("carrot")
    initial = {
        "carrot": carrot.reference_price * 1.2,
        "leeks": settings.crop("leeks").reference_price,
    }
    rows = []
    for rate in ADOPTION_RATES:
        frame = run_advisory(
            adoption_rate=rate,
            seed=0,
            n_seasons=2,
            slope_ratio=slope_ratio,
            n_farmers=n_farmers,
            settings=settings,
            initial_prices=initial,
        )
        carrot_rows = frame[frame["crop_code"] == "carrot"].sort_values("season_index")
        rows.append(
            {
                "adoption_rate": rate,
                "season_0_ha": float(carrot_rows["area_ha"].iloc[0]),
                "season_1_ha": float(carrot_rows["area_ha"].iloc[1]),
            }
        )
    return pd.DataFrame(rows)


def _line(frame: pd.DataFrame, y_field: str, y_title: str) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=alt.X("season_index:Q", title="Season", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y(f"{y_field}:Q", title=y_title),
            color=alt.Color("crop:N", title="", scale=_scale()),
            tooltip=[
                alt.Tooltip("season_label:N", title="Season"),
                alt.Tooltip("crop:N", title="Crop"),
                alt.Tooltip(f"{y_field}:Q", title=y_title, format=".2f"),
            ],
        )
    )


def season_charts(frame: pd.DataFrame, settings: BaselineSettings) -> tuple[alt.Chart, alt.Chart]:
    labeled = _label(frame)
    rules = pd.DataFrame(
        [
            {"crop": CROP_LABEL[crop.crop_code], "price": crop.reference_price, "area_ha": crop.reference_area_ha}
            for crop in settings.crops
            if crop.crop_code in ADVISORY_CROPS
        ]
    )
    price = _line(labeled, "price", "Price, Rs/kg")
    area = _line(labeled, "area_ha", "Planted area, ha")
    crashes = labeled[labeled["is_crash"]]
    if not crashes.empty:
        price = price + alt.Chart(crashes).mark_point(color="#D55E00", size=70).encode(
            x="season_index:Q",
            y="price:Q",
            tooltip=[
                alt.Tooltip("season_label:N", title="Crash season"),
                alt.Tooltip("crop:N", title="Crop"),
                alt.Tooltip("price:Q", title="Price, Rs/kg", format=".2f"),
            ],
        )
    price_rules = alt.Chart(rules).mark_rule(strokeDash=[4, 4]).encode(
        y="price:Q",
        color=alt.Color("crop:N", scale=_scale(), legend=None),
    )
    area_rules = alt.Chart(rules).mark_rule(strokeDash=[4, 4]).encode(
        y="area_ha:Q",
        color=alt.Color("crop:N", scale=_scale(), legend=None),
    )
    return (price + price_rules).properties(height=320), (area + area_rules).properties(height=320)


def adoption_chart(summary: pd.DataFrame, y_field: str, y_title: str, harti_rate: float | None) -> alt.Chart:
    labeled = _label(summary)
    band = (
        alt.Chart(labeled)
        .mark_area(opacity=0.2)
        .encode(
            x=alt.X("adoption_rate:Q", title="Adopters", axis=alt.Axis(format="%")),
            y=alt.Y(f"{y_field}_min:Q", title=y_title),
            y2=f"{y_field}_max:Q",
            color=alt.Color("crop:N", title="", scale=_scale()),
        )
    )
    line = (
        alt.Chart(labeled)
        .mark_line(point=True)
        .encode(
            x="adoption_rate:Q",
            y=alt.Y(f"{y_field}_mean:Q", title=y_title),
            color=alt.Color("crop:N", title="", scale=_scale()),
            tooltip=[
                alt.Tooltip("adoption_rate:Q", title="Adopters", format=".0%"),
                alt.Tooltip("crop:N", title="Crop"),
                alt.Tooltip(f"{y_field}_mean:Q", title="Mean", format=".3f"),
                alt.Tooltip(f"{y_field}_min:Q", title="Lowest run", format=".3f"),
                alt.Tooltip(f"{y_field}_max:Q", title="Highest run", format=".3f"),
            ],
        )
    )
    chart = band + line
    if harti_rate is not None and y_field == "crash_rate":
        rule = alt.Chart(pd.DataFrame({"rate": [harti_rate]})).mark_rule(strokeDash=[6, 4], color="#0072B2").encode(y="rate:Q")
        chart = chart + rule
    return chart.properties(height=320)


def summarize_adoption(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["adoption_rate", "crop_code"], sort=True)
        .agg(
            crash_rate_mean=("crash_rate", "mean"),
            crash_rate_min=("crash_rate", "min"),
            crash_rate_max=("crash_rate", "max"),
            price_volatility_mean=("price_volatility", "mean"),
            mean_area_ha_mean=("mean_area_ha", "mean"),
            mean_area_ha_min=("mean_area_ha", "min"),
            mean_area_ha_max=("mean_area_ha", "max"),
        )
        .reset_index()
    )


def district_frame() -> pd.DataFrame:
    land = district_land_ha()
    rows = []
    for target in district_targets():
        rows.append(
            {
                "crop_code": target.crop_code,
                "district": target.district,
                "target_ha": target.target_ha,
                "district_land_ha": land[target.district],
            }
        )
    return pd.DataFrame(rows)


def score_curve() -> pd.DataFrame:
    rows = []
    for step in range(0, 201):
        ratio = step / 100
        committed = 0.0 if ratio == 0 else ratio * 100.0
        rows.append({"ratio": ratio, "score": saturation_score(committed, 100.0)})
    return pd.DataFrame(rows)


def population_frame(n_farmers: int, adoption_rate: float, seed: int) -> pd.DataFrame:
    farmers = build_population(n_farmers, adoption_rate, seed)
    rows: dict[str, dict[str, float]] = {}
    for farmer in farmers:
        row = rows.setdefault(farmer.district, {"farmers": 0, "adopters": 0, "land_ha": 0.0})
        row["farmers"] += 1
        row["adopters"] += int(farmer.adopter)
        row["land_ha"] += farmer.land_ha
    frame = pd.DataFrame(
        [
            {"district": district, "farmers": values["farmers"], "adopters": values["adopters"], "land_ha": values["land_ha"]}
            for district, values in rows.items()
        ]
    )
    return frame.sort_values("land_ha", ascending=False).reset_index(drop=True)


def _display_table(frame: pd.DataFrame) -> pd.DataFrame:
    show = _label(frame)
    columns = ["season_index", "season_label", "crop", "area_ha", "price", "is_crash"]
    show = show[columns].copy()
    show["area_ha"] = show["area_ha"].map(lambda value: round(float(value), 1))
    show["price"] = show["price"].map(lambda value: round(float(value), 2))
    return show.rename(
        columns={
            "season_index": "Season number",
            "season_label": "Season",
            "crop": "Crop",
            "area_ha": "Area, ha",
            "price": "Price, Rs/kg",
            "is_crash": "Crash",
        }
    )


def render_sidebar() -> dict[str, object]:
    st.sidebar.header("Run")
    adoption = st.sidebar.slider("Adopters", min_value=0, max_value=100, value=30, step=10)
    seed = int(st.sidebar.number_input("Seed", min_value=0, max_value=200, value=0, step=1))
    n_seasons = int(st.sidebar.slider("Seasons", min_value=8, max_value=24, value=16, step=1))
    n_districts = len(district_land_ha())
    n_farmers = int(st.sidebar.slider("Farmers", min_value=n_districts, max_value=200, value=100, step=1))
    replications = int(st.sidebar.select_slider("Runs in the adoption comparison", options=[5, 10, 20, 40], value=10))
    slope_ratio = float(
        st.sidebar.selectbox(
            "Slope ratio",
            options=list(SLOPE_CHOICES),
            index=0,
            format_func=lambda value: f"{value:.2f}",
        )
    )
    use_harti_path = st.sidebar.radio(
        "Price path",
        options=["Fitted to Colombo wholesale", "Clearing price only"],
        index=0,
    ) == "Fitted to Colombo wholesale"
    st.sidebar.caption(
        "Slope ratio 0.05 is the bottom of the grid used against Colombo wholesale. "
        "A larger ratio makes a price gap grow. The fitted path keeps part of last season's price and adds a shock estimated from those seasons."
    )
    return {
        "adoption_rate": adoption / 100,
        "seed": seed,
        "n_seasons": n_seasons,
        "n_farmers": n_farmers,
        "replications": replications,
        "slope_ratio": slope_ratio,
        "use_harti_path": use_harti_path,
    }


def render_path(choices: dict[str, object]) -> None:
    st.subheader("One run")
    st.caption(
        "Adopters are chosen once and kept for every season. "
        "A crash, drawn as a point, is a season at least 25% below the previous season. The first season is never a crash. "
        "Dashed lines are the price center and the checked national targets."
    )
    try:
        frame = load_run(
            float(choices["adoption_rate"]),
            int(choices["seed"]),
            int(choices["n_seasons"]),
            int(choices["n_farmers"]),
            float(choices["slope_ratio"]),
            bool(choices["use_harti_path"]),
        )
        settings, _persistence, _shock_std = _settings(bool(choices["use_harti_path"]))
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"This run could not be built. {exc}")
        return
    carrot = frame[frame["crop_code"] == "carrot"].sort_values("season_index")
    leeks = frame[frame["crop_code"] == "leeks"].sort_values("season_index")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Carrot crash rate", f"{crash_rate(carrot):.2f}")
    metric_cols[1].metric("Leeks crash rate", f"{crash_rate(leeks):.2f}")
    metric_cols[2].metric("Carrot mean area", f"{carrot['area_ha'].mean():.0f} ha")
    metric_cols[3].metric("Leeks mean area", f"{leeks['area_ha'].mean():.0f} ha")
    price_chart, area_chart = season_charts(frame, settings)
    left, right = st.columns(2)
    left.altair_chart(price_chart, width="stretch")
    right.altair_chart(area_chart, width="stretch")
    show = _display_table(frame)
    st.dataframe(show, hide_index=True, width="stretch")
    st.download_button(
        "Download this run",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="simulation_run.csv",
        mime="text/csv",
    )


def render_adoption(choices: dict[str, object]) -> None:
    st.subheader("Adoption comparison")
    st.caption(
        "Every adoption rate sees the same seeds and the same shocks. "
        "Compare a rate with 0% adopters. The blue dashed line, when the warehouse is available, is the Colombo wholesale crash rate. "
        "It is the historical rate, not the control in this model."
    )
    try:
        results = load_adoption(
            int(choices["n_seasons"]),
            int(choices["replications"]),
            int(choices["n_farmers"]),
            float(choices["slope_ratio"]),
            bool(choices["use_harti_path"]),
        )
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"The adoption comparison could not be built. {exc}")
        return
    summary = summarize_adoption(results)
    harti_rate = None
    if bool(choices["use_harti_path"]):
        try:
            calibration = load_calibration()
            advisory = calibration[calibration["crop_code"].isin(ADVISORY_CROPS)]
            rates = {str(row.crop_code): float(row.crash_rate) for row in advisory.itertuples(index=False)}
            if "carrot" in rates and "leeks" in rates and abs(rates["carrot"] - rates["leeks"]) < 1e-9:
                harti_rate = rates["carrot"]
        except (FileNotFoundError, ValueError):
            harti_rate = None
    left, right = st.columns(2)
    left.altair_chart(adoption_chart(summary, "crash_rate", "Crash rate", harti_rate), width="stretch")
    right.altair_chart(adoption_chart(summary, "mean_area_ha", "Mean planted area, ha", None), width="stretch")
    if not bool(choices["use_harti_path"]):
        st.info("Clearing price only has no shock. Seeds then produce the same path, and the crash rate stays near zero while planting stays near the targets.")
    show = summary.copy()
    show["crop"] = show["crop_code"].map(CROP_LABEL)
    show["adopters"] = show["adoption_rate"].map(lambda value: f"{value:.0%}")
    show = show[
        ["adopters", "crop", "crash_rate_mean", "crash_rate_min", "crash_rate_max", "price_volatility_mean", "mean_area_ha_mean"]
    ].rename(
        columns={
            "adopters": "Adopters",
            "crop": "Crop",
            "crash_rate_mean": "Mean crash rate",
            "crash_rate_min": "Lowest crash rate",
            "crash_rate_max": "Highest crash rate",
            "price_volatility_mean": "Mean price volatility",
            "mean_area_ha_mean": "Mean area, ha",
        }
    )
    for column in ("Mean crash rate", "Lowest crash rate", "Highest crash rate", "Mean price volatility"):
        show[column] = show[column].map(lambda value: round(float(value), 3))
    show["Mean area, ha"] = show["Mean area, ha"].map(lambda value: round(float(value), 1))
    st.dataframe(show, hide_index=True, width="stretch")
    st.download_button(
        "Download adoption runs",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="adoption_runs.csv",
        mime="text/csv",
    )
    with st.expander("One 20% carrot price shock, without the random shock"):
        st.caption("Season 0 has no score yet, so every adoption rate plants the same carrot area. Season 1 is the first season that can use the score.")
        shock = load_opening_shock(int(choices["n_farmers"]), float(choices["slope_ratio"]), bool(choices["use_harti_path"]))
        shock_show = shock.copy()
        shock_show["Adopters"] = shock_show["adoption_rate"].map(lambda value: f"{value:.0%}")
        shock_show["Season 0, ha"] = shock_show["season_0_ha"].map(lambda value: round(float(value), 1))
        shock_show["Season 1, ha"] = shock_show["season_1_ha"].map(lambda value: round(float(value), 1))
        st.dataframe(shock_show[["Adopters", "Season 0, ha", "Season 1, ha"]], hide_index=True, width="stretch")
        shock_long = shock.melt(id_vars="adoption_rate", value_vars=["season_0_ha", "season_1_ha"], var_name="season", value_name="area_ha")
        shock_long["season"] = shock_long["season"].map({"season_0_ha": "Season 0", "season_1_ha": "Season 1"})
        shock_chart = (
            alt.Chart(shock_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("adoption_rate:Q", title="Adopters", axis=alt.Axis(format="%")),
                y=alt.Y("area_ha:Q", title="Carrot area, ha"),
                color=alt.Color("season:N", title=""),
            )
            .properties(height=280)
        )
        st.altair_chart(shock_chart, width="stretch")


def render_check() -> None:
    st.subheader("Colombo wholesale check")
    st.caption(
        "This page does not use the sidebar. "
        f"The no-advisory runs use slope ratio {SLOPE_RATIO:.2f} and {N_REPLICATIONS} seeds. "
        "Persistence and the shock scale are estimated from these same seasons. "
        "Tomato has no checked district table, so its simulated path is price only."
    )
    try:
        observed = load_observed()
        check = load_price_check()
        calibration = load_calibration()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"The local warehouse could not be read. {exc}")
        return
    labeled = _label(observed)
    history = (
        alt.Chart(labeled)
        .mark_line(point=True)
        .encode(
            x=alt.X("ordinal:Q", title="Season order from 2015"),
            y=alt.Y("price:Q", title="Colombo wholesale, Rs/kg"),
            color=alt.Color("crop:N", title="", scale=_scale()),
            tooltip=[
                alt.Tooltip("season_label:N", title="Season"),
                alt.Tooltip("crop:N", title="Crop"),
                alt.Tooltip("price:Q", title="Mean weekly price", format=".2f"),
                alt.Tooltip("n_weeks:Q", title="Priced weeks"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(history, width="stretch")
    st.caption("Each point is the mean of priced weeks in one Maha or Yala season. Seasons with fewer than 15 priced weeks are left out.")
    check_show = _label(check)
    check_show = check_show[
        [
            "crop",
            "scope",
            "persistence",
            "shock_std",
            "center",
            "harti_crash_rate",
            "simulated_crash_rate",
            "harti_lag",
            "simulated_lag",
            "harti_deseasoned_lag",
            "runs",
        ]
    ].rename(
        columns={
            "crop": "Crop",
            "scope": "Simulated scope",
            "persistence": "Persistence",
            "shock_std": "Shock std, Rs/kg",
            "center": "Center, Rs/kg",
            "harti_crash_rate": "HARTI crash rate",
            "simulated_crash_rate": "Simulated crash rate",
            "harti_lag": "HARTI lag",
            "simulated_lag": "Simulated lag",
            "harti_deseasoned_lag": "HARTI deseasoned lag",
            "runs": "Runs",
        }
    )
    for column in ("Persistence", "HARTI crash rate", "Simulated crash rate", "HARTI lag", "Simulated lag", "HARTI deseasoned lag"):
        check_show[column] = check_show[column].map(lambda value: None if pd.isna(value) else round(float(value), 2))
    check_show["Shock std, Rs/kg"] = check_show["Shock std, Rs/kg"].map(lambda value: round(float(value), 1))
    check_show["Center, Rs/kg"] = check_show["Center, Rs/kg"].map(lambda value: round(float(value), 1))
    st.dataframe(check_show, hide_index=True, width="stretch")
    st.caption("A simulated crash rate below the HARTI rate is a property of this price rule. It is not evidence about the advisory.")
    calib = _label(calibration)
    calib = calib[
        ["crop", "seasons", "pairs", "crashes", "crash_rate", "lag", "deseasoned_lag", "best_ratio", "flat_rmse", "persist_rmse", "cobweb_rmse", "closest"]
    ].rename(
        columns={
            "crop": "Crop",
            "seasons": "Seasons",
            "pairs": "Consecutive pairs",
            "crashes": "Crashes",
            "crash_rate": "Crash rate",
            "lag": "Lag",
            "deseasoned_lag": "Deseasoned lag",
            "best_ratio": "Best slope ratio",
            "flat_rmse": "Flat RMSE",
            "persist_rmse": "Persist RMSE",
            "cobweb_rmse": "Cobweb RMSE",
            "closest": "Closest rule",
        }
    )
    st.markdown("**One-step rules on the same seasons**")
    st.caption("Flat repeats the mean season price. Persist repeats last season. Cobweb uses the slope ratio with the smallest one-step price error. RMSE is Rs/kg.")
    st.dataframe(calib, hide_index=True, width="stretch")


def render_districts(choices: dict[str, object]) -> None:
    st.subheader("District targets and the saturation score")
    warning = MEMBER3_CONFIG.warning_pressure_ratio
    critical = MEMBER3_CONFIG.critical_pressure_ratio
    st.caption(
        f"The score is Member 3's pressure function: 0 at {warning:.0%} of a district target and 1 at {critical:.0%}. "
        "The capacity is the checked target below. Member 3's demo capacities are not used. "
        "Tomato is absent because no tomato district table passed the sum check."
    )
    targets = _label(district_frame())
    chart = (
        alt.Chart(targets)
        .mark_bar()
        .encode(
            x=alt.X("target_ha:Q", title="Checked target, ha"),
            y=alt.Y(
                "district:N",
                title="",
                sort=targets.sort_values("district_land_ha", ascending=False)["district"].drop_duplicates().tolist(),
            ),
            color=alt.Color("crop:N", title="", scale=_scale()),
            tooltip=[
                alt.Tooltip("district:N", title="District"),
                alt.Tooltip("crop:N", title="Crop"),
                alt.Tooltip("target_ha:Q", title="Target, ha", format=".0f"),
            ],
        )
        .properties(height=420)
    )
    left, right = st.columns(2)
    left.altair_chart(chart, width="stretch")
    curve = score_curve()
    score = (
        alt.Chart(curve)
        .mark_line()
        .encode(x=alt.X("ratio:Q", title="Planted area / district target"), y=alt.Y("score:Q", title="Saturation score"))
        .properties(height=420)
    )
    marks = pd.DataFrame({"ratio": [warning, 1.0, critical]})
    score = score + alt.Chart(marks).mark_rule(strokeDash=[4, 4], color="#0072B2").encode(x="ratio:Q")
    right.altair_chart(score, width="stretch")
    right.caption(f"Dashed lines at {warning:.2f}, 1.00, and {critical:.2f}. At the target the score is 0.5, so both crops are scaled by the same amount and the mix does not move.")
    show = targets[["district", "crop", "target_ha"]].rename(columns={"district": "District", "crop": "Crop", "target_ha": "Target, ha"})
    st.dataframe(show, hide_index=True, width="stretch")
    st.markdown("**Farmers in this run**")
    st.caption("Land is the sum of the checked carrot and leeks targets. The same seed keeps the same adopter list when the adoption rate and the farmer count stay the same.")
    people = population_frame(int(choices["n_farmers"]), float(choices["adoption_rate"]), int(choices["seed"]))
    people["land_ha"] = people["land_ha"].map(lambda value: round(float(value), 1))
    people = people.rename(columns={"district": "District", "farmers": "Farmers", "adopters": "Adopters", "land_ha": "Land, ha"})
    st.dataframe(people, hide_index=True, width="stretch")
    totals = st.columns(3)
    totals[0].metric("Districts", f"{len(people)}")
    totals[1].metric("Farmers", f"{int(people['Farmers'].sum())}")
    totals[2].metric("Adopters", f"{int(people['Adopters'].sum())}")


def main() -> None:
    st.set_page_config(page_title="Agent simulation", layout="wide")
    st.title("Agent simulation")
    st.caption("Carrot and leeks. Fixed adopters. Member 3 saturation score. Prices are Colombo wholesale rupees per kilogram.")
    choices = render_sidebar()
    path_tab, adoption_tab, check_tab, district_tab = st.tabs(
        ["Season path", "Adoption", "Colombo wholesale", "Districts and score"]
    )
    with path_tab:
        render_path(choices)
    with adoption_tab:
        render_adoption(choices)
    with check_tab:
        render_check()
    with district_tab:
        render_districts(choices)


main()
