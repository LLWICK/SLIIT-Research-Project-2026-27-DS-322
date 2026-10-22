"""Read-only dashboard for curated.crop_market_week."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import psycopg
import streamlit as st

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"

MARKET_COLORS = {
    "Colombo": "#0072B2",
    "Dambulla": "#E69F00",
    "Meegoda": "#009E73",
    "Nuwara Eliya": "#CC79A7",
}
SUPPLY_COLORS = {
    "Nuwara Eliya": "#0072B2",
    "Badulla": "#009E73",
    "Dambulla": "#E69F00",
}
EXTENT_COLORS = {"Target": "#0072B2", "Achievement": "#E69F00"}
GROUP_LABEL = {
    "national": "National",
    "provincial": "Provincial",
    "inter_provincial": "Inter-provincial",
    "mahaweli": "Mahaweli",
}
LEVEL_LABEL = {"national": "National", "subtotal": "Subtotal", "area": "Area"}
GROUP_RANK = {"national": 0, "provincial": 1, "inter_provincial": 2, "mahaweli": 3}
LEVEL_RANK = {"national": 0, "subtotal": 1, "area": 2}

WEEK_SQL = """
    SELECT
        price_type, market_code, market_name, crop_code, crop_name,
        year, week, week_start, week_end, season_label,
        harti_price, harti_is_missing, cbsl_price, cbsl_is_missing,
        extent_report_date, extent_as_of_date, extent_season_label,
        extent_target_ha, extent_achievement_ha,
        extent_target_missing, extent_achievement_missing,
        nuwara_eliya_t2m_c, nuwara_eliya_precip_mm, nuwara_eliya_days,
        badulla_t2m_c, badulla_precip_mm, badulla_days,
        dambulla_t2m_c, dambulla_precip_mm, dambulla_days
    FROM curated.crop_market_week
"""

EXTENT_SQL = """
    SELECT
        report_date, season_label, as_of_date, crop_code,
        region_name, region_level, region_group,
        target_ha, achievement_ha, target_missing, achievement_missing,
        source_file
    FROM curated.cultivation_extent
"""


def reader_url() -> str:
    if not ENV_PATH.exists():
        raise FileNotFoundError("The project .env file is missing.")
    values: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    url = values.get("NEON_READER_URL")
    if not url:
        raise RuntimeError("NEON_READER_URL is missing from .env.")
    return url


def _fetch(cur, sql: str) -> pd.DataFrame:
    cur.execute(sql)
    columns = [item.name for item in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=columns)


@st.cache_data(show_spinner="Loading the warehouse…", ttl=300)
def load_warehouse() -> tuple[pd.DataFrame, pd.DataFrame]:
    with psycopg.connect(reader_url(), connect_timeout=30) as con:
        with con.cursor() as cur:
            weeks = _fetch(cur, WEEK_SQL)
            extent = _fetch(cur, EXTENT_SQL)
    for column in ("week_start", "week_end", "extent_report_date", "extent_as_of_date"):
        weeks[column] = pd.to_datetime(weeks[column])
    for column in ("report_date", "as_of_date"):
        extent[column] = pd.to_datetime(extent[column])
    weeks["year"] = weeks["year"].astype(int)
    weeks["week"] = weeks["week"].astype(int)
    return weeks, extent


def date_format(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "%Y"
    span = (frame["week_start"].max() - frame["week_start"].min()).days
    if span < 120:
        return "%d %b %Y"
    if span < 800:
        return "%b %Y"
    return "%Y"


def line_chart(frame: pd.DataFrame, y_field: str, y_title: str, color_field: str, colors: dict[str, str]) -> alt.Chart:
    present = [name for name in colors if name in set(frame[color_field])]
    return (
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=alt.X("week_start:T", title="Week starting Friday", axis=alt.Axis(format=date_format(frame))),
            y=alt.Y(f"{y_field}:Q", title=y_title),
            color=alt.Color(
                f"{color_field}:N",
                title="",
                scale=alt.Scale(domain=present, range=[colors[name] for name in present]),
            ),
            tooltip=[
                alt.Tooltip("week_start:T", title="Friday", format="%d %b %Y"),
                alt.Tooltip("week_end:T", title="Thursday", format="%d %b %Y"),
                alt.Tooltip("year:Q", title="HARTI year", format=".0f"),
                alt.Tooltip("week:Q", title="Week", format=".0f"),
                alt.Tooltip("season:N", title="Season"),
                alt.Tooltip(f"{color_field}:N", title="Series"),
                alt.Tooltip(f"{y_field}:Q", title=y_title, format=".2f"),
            ],
        )
        .properties(height=340)
        .interactive()
    )


def coverage_notes(weeks: pd.DataFrame) -> list[str]:
    notes = []
    priced = weeks[weeks["harti_price"].notna()]
    if not priced.empty:
        for price_type, start in priced.groupby("price_type")["year"].min().items():
            notes.append(f"{price_type.title()} prices in the warehouse start in {int(start)}.")
    week_53 = weeks[weeks["week"] == 53]
    if not week_53.empty and week_53["harti_price"].isna().all():
        years = ", ".join(str(year) for year in sorted(week_53["year"].unique()))
        notes.append(f"Week 53 ({years}) has no workbook price.")
    return notes


def cbsl_note(weeks: pd.DataFrame, crop_code: str, market_codes: list[str], year_from: int, year_to: int) -> str | None:
    available = weeks[weeks["cbsl_price"].notna()]
    if available.empty:
        return "This warehouse has no CBSL prices."
    crops = set(available["crop_code"])
    markets = set(available["market_code"])
    years = sorted(int(year) for year in available["year"].unique())
    if crop_code not in crops:
        names = ", ".join(sorted(available["crop_name"].unique()))
        return f"CBSL in this warehouse covers {names} only."
    if not any(code in markets for code in market_codes):
        names = ", ".join(sorted(available["market_name"].unique()))
        return f"CBSL in this warehouse covers {names} only."
    if year_to < min(years) or year_from > max(years):
        span = f"{years[0]}" if years[0] == years[-1] else f"{years[0]}–{years[-1]}"
        return f"CBSL prices in this warehouse are in {span}. Move the year filter to include that year."
    return None


def weather_frame(selected: pd.DataFrame) -> pd.DataFrame:
    one_market = selected["market_code"].iloc[0]
    base = (
        selected.loc[selected["market_code"] == one_market, ["week_start", "week_end", "year", "week", "season"]]
        .drop_duplicates("week_start")
        .sort_values("week_start")
    )
    columns = {
        "nuwara_eliya_precip_mm": ("Nuwara Eliya", "precip_mm"),
        "badulla_precip_mm": ("Badulla", "precip_mm"),
        "dambulla_precip_mm": ("Dambulla", "precip_mm"),
        "nuwara_eliya_t2m_c": ("Nuwara Eliya", "t2m_c"),
        "badulla_t2m_c": ("Badulla", "t2m_c"),
        "dambulla_t2m_c": ("Dambulla", "t2m_c"),
        "nuwara_eliya_days": ("Nuwara Eliya", "days"),
        "badulla_days": ("Badulla", "days"),
        "dambulla_days": ("Dambulla", "days"),
    }
    source = selected.loc[selected["market_code"] == one_market].drop_duplicates("week_start")
    pieces = []
    for column, (place, measure) in columns.items():
        piece = base.merge(source[["week_start", column]], on="week_start", how="left")
        piece = piece.rename(columns={column: "value"})
        piece["place"] = place
        piece["measure"] = measure
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def extent_on_weeks(selected: pd.DataFrame) -> pd.DataFrame:
    one_market = selected["market_code"].iloc[0]
    base = (
        selected.loc[
            selected["market_code"] == one_market,
            ["week_start", "week_end", "year", "week", "season", "extent_target_ha", "extent_achievement_ha", "extent_report_date"],
        ]
        .drop_duplicates("week_start")
        .sort_values("week_start")
    )
    long = base.melt(
        id_vars=["week_start", "week_end", "year", "week", "season", "extent_report_date"],
        value_vars=["extent_target_ha", "extent_achievement_ha"],
        var_name="measure",
        value_name="hectares",
    )
    long["measure"] = long["measure"].map(
        {"extent_target_ha": "Target", "extent_achievement_ha": "Achievement"}
    )
    return long


def snapshot_table(extent: pd.DataFrame, crop_code: str, report_date: pd.Timestamp) -> pd.DataFrame:
    rows = extent[(extent["crop_code"] == crop_code) & (extent["report_date"] == report_date)].copy()
    rows["group_rank"] = rows["region_group"].map(GROUP_RANK).fillna(9)
    rows["level_rank"] = rows["region_level"].map(LEVEL_RANK).fillna(9)
    rows = rows.sort_values(["group_rank", "level_rank", "region_name"])
    target = []
    achievement = []
    for row in rows.itertuples(index=False):
        target.append(pd.NA if row.target_missing or pd.isna(row.target_ha) else row.target_ha)
        achievement.append(
            pd.NA if row.achievement_missing or pd.isna(row.achievement_ha) else row.achievement_ha
        )
    return pd.DataFrame(
        {
            "Region": rows["region_name"].to_numpy(),
            "Level": rows["region_level"].map(LEVEL_LABEL).fillna(rows["region_level"]).to_numpy(),
            "Group": rows["region_group"].map(GROUP_LABEL).fillna(rows["region_group"]).to_numpy(),
            "Target (ha)": target,
            "Achievement (ha)": achievement,
        }
    )


def main() -> None:
    try:
        weeks, extent = load_warehouse()
    except Exception as exc:
        st.error("Could not read the warehouse. Check NEON_READER_URL in .env. If Neon is asleep, wait and use Reload.")
        st.caption(type(exc).__name__)
        st.stop()

    st.title("Cobweb data layer")
    st.caption(
        "Read-only view of `curated.crop_market_week`. "
        f"{len(weeks):,} rows. The reader login cannot change the warehouse."
    )

    with st.expander("How a row is built"):
        st.markdown(
            """
- A week runs Friday through Thursday. It belongs to the year that contains at least four of those days. Week 1 of 2025 is 3–9 January.
- `harti_price` is the workbook value. `cbsl_price` is the average of CBSL days inside that same week. They stay in separate columns.
- Extent is the latest Sri Lanka total whose report date and as-of date are both on or before that Thursday.
- Weather is Nuwara Eliya, Badulla, and Dambulla, where the crops are grown. A missing day is left out of the mean and the sum. It is not stored as zero.
- A dash in an extent report is an empty hectare cell, not zero.
            """
        )

    crop_lookup = (
        weeks[["crop_code", "crop_name"]].drop_duplicates().sort_values("crop_name")
    )
    crop_names = dict(zip(crop_lookup["crop_code"], crop_lookup["crop_name"], strict=True))
    market_lookup = (
        weeks[["market_code", "market_name"]].drop_duplicates().sort_values("market_name")
    )
    market_names = dict(zip(market_lookup["market_code"], market_lookup["market_name"], strict=True))
    year_min = int(weeks["year"].min())
    year_max = int(weeks["year"].max())

    st.sidebar.header("Filter")
    crop_code = st.sidebar.selectbox("Crop", list(crop_names), format_func=crop_names.get)
    price_type = st.sidebar.radio("Price type", ["wholesale", "retail"], format_func=str.title, horizontal=True)
    market_codes = st.sidebar.multiselect(
        "Markets",
        list(market_names),
        default=list(market_names),
        format_func=market_names.get,
    )
    if year_min == year_max:
        year_from = year_to = year_min
        st.sidebar.caption(f"HARTI year {year_min}")
    else:
        year_from, year_to = st.sidebar.slider("HARTI year", year_min, year_max, (year_min, year_max))
    st.sidebar.caption("The year is the HARTI week year. Week 1 of 2024 starts on 29 December 2023.")
    if st.sidebar.button("Reload"):
        load_warehouse.clear()
        st.rerun()

    if not market_codes:
        st.info("Choose at least one market.")
        st.stop()

    selected = weeks[
        (weeks["crop_code"] == crop_code)
        & (weeks["price_type"] == price_type)
        & (weeks["market_code"].isin(market_codes))
        & (weeks["year"].between(year_from, year_to))
    ].copy()
    selected["season"] = selected["season_label"].fillna("Spans two seasons")
    crop_name = crop_names[crop_code]

    priced = int(selected["harti_price"].notna().sum())
    blank = int(selected["harti_is_missing"].fillna(False).astype(bool).sum())
    cbsl_priced = int(selected["cbsl_price"].notna().sum())
    week_count = int(selected["week_start"].nunique())
    metric_cols = st.columns(4)
    metric_cols[0].metric("Weeks", f"{week_count:,}")
    metric_cols[1].metric("HARTI prices", f"{priced:,}")
    metric_cols[2].metric("Blank HARTI cells", f"{blank:,}")
    metric_cols[3].metric("CBSL prices", f"{cbsl_priced:,}")

    st.subheader(f"HARTI {price_type} price — {crop_name}")
    if priced == 0:
        year_span = f"{year_from}" if year_from == year_to else f"{year_from}–{year_to}"
        st.info(f"No HARTI {price_type} price is stored for {crop_name} in {year_span}.")
    else:
        st.altair_chart(
            line_chart(selected, "harti_price", "Weekly price", "market_name", MARKET_COLORS),
            width="stretch",
        )
    for note in coverage_notes(weeks):
        st.caption(note)
    st.caption("A gap is a blank cell or a week the workbook does not contain. Blank cells are not filled in.")

    st.subheader(f"CBSL average inside the same week — {crop_name}")
    missing_reason = cbsl_note(weeks, crop_code, market_codes, year_from, year_to)
    cbsl_rows = selected[selected["cbsl_price"].notna() | selected["cbsl_is_missing"].fillna(False)]
    if missing_reason and cbsl_priced == 0:
        st.info(missing_reason)
    else:
        st.altair_chart(
            line_chart(cbsl_rows, "cbsl_price", "Rs/kg", "market_name", MARKET_COLORS),
            width="stretch",
        )
        st.caption("Each point averages the CBSL daily prices from Friday through Thursday. Days with no price are left out. Pettah is stored as Colombo.")

    extent_long = extent_on_weeks(selected)
    weather = weather_frame(selected)
    left, right = st.columns(2)

    with left:
        st.subheader("National extent known that week")
        if extent_long["hectares"].notna().any():
            present = [name for name in EXTENT_COLORS if name in set(extent_long["measure"])]
            chart = (
                alt.Chart(extent_long)
                .mark_line(interpolate="step-after")
                .encode(
                    x=alt.X("week_start:T", title="Week starting Friday", axis=alt.Axis(format=date_format(extent_long))),
                    y=alt.Y("hectares:Q", title="Hectares"),
                    color=alt.Color(
                        "measure:N",
                        title="",
                        scale=alt.Scale(domain=present, range=[EXTENT_COLORS[name] for name in present]),
                    ),
                    tooltip=[
                        alt.Tooltip("week_start:T", title="Friday", format="%d %b %Y"),
                        alt.Tooltip("measure:N", title=""),
                        alt.Tooltip("hectares:Q", title="Hectares", format=".0f"),
                        alt.Tooltip("extent_report_date:T", title="Report date", format="%d %b %Y"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, width="stretch")
            st.caption("Sri Lanka total. The same total is attached to every market. A later report does not appear on an earlier week.")
        else:
            st.info(f"No checked extent snapshot is attached to {crop_name} in this year range.")

    with right:
        st.subheader("Supply-district weather")
        rain = weather[weather["measure"] == "precip_mm"]
        temperature = weather[weather["measure"] == "t2m_c"]
        rain_tab, temp_tab = st.tabs(["Rain", "Temperature"])
        with rain_tab:
            if rain["value"].notna().any():
                st.altair_chart(
                    line_chart(rain, "value", "Precipitation (mm)", "place", SUPPLY_COLORS),
                    width="stretch",
                )
            else:
                st.info("No rainfall is stored for these weeks.")
        with temp_tab:
            if temperature["value"].notna().any():
                st.altair_chart(
                    line_chart(temperature, "value", "Mean temperature (°C)", "place", SUPPLY_COLORS),
                    width="stretch",
                )
            else:
                st.info("No temperature is stored for these weeks.")
        observed = weather.loc[weather["measure"] == "days", "value"].dropna()
        if not observed.empty and observed.min() < 7:
            st.caption(
                f"Some weeks have fewer than 7 observed days. The lowest in this filter is {int(observed.min())}. "
                "Missing days are left out of the mean and the sum."
            )
        last_weather = weather.loc[weather["value"].notna(), "week_end"]
        if not last_weather.empty:
            st.caption(f"Weather in this filter runs through the week ending {last_weather.max():%d %b %Y}.")
        st.caption("These are growing areas, not the Colombo and Meegoda markets.")

    st.subheader(f"Extent rows for {crop_name}")
    crop_extent = extent[extent["crop_code"] == crop_code]
    if crop_extent.empty:
        st.info(f"No {crop_name} page passed the sum check, so the extent table has no {crop_name} rows.")
    else:
        attached = selected["extent_report_date"].dropna()
        dates = sorted(crop_extent["report_date"].unique())
        default_date = attached.max() if not attached.empty and attached.max() in set(dates) else dates[-1]
        labels = {}
        for report_date, as_of_date, season_label, source_file in (
            crop_extent[["report_date", "as_of_date", "season_label", "source_file"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ):
            labels[report_date] = f"{report_date:%d %b %Y} · {season_label} · as at {as_of_date:%d %b %Y} · {source_file}"
        chosen_date = st.selectbox(
            "Snapshot",
            dates,
            index=dates.index(default_date),
            format_func=labels.get,
        )
        table = snapshot_table(crop_extent, crop_code, chosen_date)
        st.dataframe(table, hide_index=True, width="stretch")
        if table["Target (ha)"].isna().any() or table["Achievement (ha)"].isna().any():
            st.caption("An empty hectare cell is a dash in the report, not zero.")

    st.subheader("Rows behind the charts")
    display = selected.sort_values(["week_start", "market_name"]).rename(
        columns={
            "week_start": "Friday",
            "week_end": "Thursday",
            "year": "Year",
            "week": "Week",
            "season": "Season",
            "market_name": "Market",
            "harti_price": "HARTI price",
            "cbsl_price": "CBSL price",
            "extent_target_ha": "Target (ha)",
            "extent_achievement_ha": "Achievement (ha)",
        }
    )
    st.dataframe(
        display[
            [
                "Friday",
                "Thursday",
                "Year",
                "Week",
                "Season",
                "Market",
                "HARTI price",
                "CBSL price",
                "Target (ha)",
                "Achievement (ha)",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download these rows",
        selected.sort_values(["week_start", "market_code"]).to_csv(index=False).encode("utf-8"),
        file_name=f"{crop_code}_{price_type}_{year_from}_{year_to}.csv",
        mime="text/csv",
    )


st.set_page_config(page_title="Cobweb data layer", layout="wide")
main()
