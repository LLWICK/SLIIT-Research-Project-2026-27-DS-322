"""One read-only query surface for the other members.

A HARTI week runs from Friday through Thursday. It belongs to the calendar
year that contains at least four of those days, and W1 is the first such
week of that year. This matches the printed dates on the HARTI bulletins
(week 01 of 2025 is 3-9 January; week 01 of 2024 is 29 December-4 January).
The workbook stops at W52, so a 53rd week has no HARTI price.

Extent on a row is the latest national snapshot whose report date and as-of
date are both on or before that Thursday. CBSL stays in its own column.
"""

from __future__ import annotations

import duckdb

from data_layer.paths import DUCKDB_PATH

VIEW_NAME = "crop_market_week"

VIEW_SQL = """
CREATE OR REPLACE VIEW curated.crop_market_week AS
WITH fridays AS (
    SELECT
        calendar_date AS week_start,
        CAST(calendar_date + 6 AS date) AS week_end,
        CAST(extract(year FROM calendar_date + 3) AS integer) AS year
    FROM curated.season_calendar
    WHERE CAST(extract(isodow FROM calendar_date) AS integer) = 5
),
weeks AS (
    SELECT
        week_start,
        week_end,
        year,
        CAST(row_number() OVER (PARTITION BY year ORDER BY week_start) AS integer) AS week
    FROM fridays
),
week_season AS (
    SELECT
        weeks.week_start,
        CASE WHEN count(DISTINCT season_calendar.season_code) = 1 THEN min(season_calendar.season_code) END AS season_code,
        CASE WHEN count(DISTINCT season_calendar.season_label) = 1 THEN min(season_calendar.season_label) END AS season_label
    FROM weeks
    JOIN curated.season_calendar
        ON season_calendar.calendar_date BETWEEN weeks.week_start AND weeks.week_end
    GROUP BY weeks.week_start
),
price_types AS (
    SELECT * FROM (VALUES ('retail'), ('wholesale')) AS listed(price_type)
),
spine AS (
    SELECT price_type, market_code, market_name, crop_code, crop_name
    FROM price_types
    CROSS JOIN curated.dim_market
    CROSS JOIN curated.dim_crop
),
national_extent AS (
    SELECT
        crop_code,
        CAST(report_date AS date) AS report_date,
        CAST(as_of_date AS date) AS as_of_date,
        season_label,
        target_ha,
        achievement_ha,
        target_missing,
        achievement_missing
    FROM curated.cultivation_extent
    WHERE region_level = 'national'
),
extent_ranked AS (
    SELECT
        weeks.week_start,
        national_extent.crop_code,
        national_extent.report_date,
        national_extent.as_of_date,
        national_extent.season_label,
        national_extent.target_ha,
        national_extent.achievement_ha,
        national_extent.target_missing,
        national_extent.achievement_missing,
        row_number() OVER (
            PARTITION BY weeks.week_start, national_extent.crop_code
            ORDER BY national_extent.report_date DESC, national_extent.as_of_date DESC
        ) AS rn
    FROM weeks
    JOIN national_extent
        ON national_extent.report_date <= weeks.week_end
       AND national_extent.as_of_date <= weeks.week_end
),
extent_for_week AS (
    SELECT
        week_start,
        crop_code,
        report_date,
        as_of_date,
        season_label,
        target_ha,
        achievement_ha,
        target_missing,
        achievement_missing
    FROM extent_ranked
    WHERE rn = 1
),
cbsl_week AS (
    SELECT
        weeks.week_start,
        prices.price_type,
        prices.market_code,
        prices.crop_code,
        avg(prices.price) FILTER (WHERE NOT prices.is_missing) AS cbsl_price,
        CASE
            WHEN count(prices.price) FILTER (WHERE NOT prices.is_missing) > 0 THEN false
            WHEN count(*) > 0 THEN true
        END AS cbsl_is_missing
    FROM curated.cbsl_price_daily AS prices
    JOIN weeks
        ON prices.report_date BETWEEN weeks.week_start AND weeks.week_end
    GROUP BY 1, 2, 3, 4
),
weather_point_week AS (
    SELECT
        weeks.week_start,
        weather.point_code,
        avg(weather.t2m_c) FILTER (WHERE NOT weather.is_missing) AS t2m_c,
        sum(weather.precip_mm) FILTER (WHERE NOT weather.is_missing) AS precip_mm,
        count(*) FILTER (WHERE NOT weather.is_missing) AS observed_days
    FROM curated.weather_daily AS weather
    JOIN weeks
        ON weather.obs_date BETWEEN weeks.week_start AND weeks.week_end
    GROUP BY 1, 2
),
weather_week AS (
    SELECT
        week_start,
        max(CASE WHEN point_code = 'nuwara_eliya' THEN t2m_c END) AS nuwara_eliya_t2m_c,
        max(CASE WHEN point_code = 'nuwara_eliya' THEN precip_mm END) AS nuwara_eliya_precip_mm,
        max(CASE WHEN point_code = 'nuwara_eliya' THEN observed_days END) AS nuwara_eliya_days,
        max(CASE WHEN point_code = 'badulla' THEN t2m_c END) AS badulla_t2m_c,
        max(CASE WHEN point_code = 'badulla' THEN precip_mm END) AS badulla_precip_mm,
        max(CASE WHEN point_code = 'badulla' THEN observed_days END) AS badulla_days,
        max(CASE WHEN point_code = 'dambulla' THEN t2m_c END) AS dambulla_t2m_c,
        max(CASE WHEN point_code = 'dambulla' THEN precip_mm END) AS dambulla_precip_mm,
        max(CASE WHEN point_code = 'dambulla' THEN observed_days END) AS dambulla_days
    FROM weather_point_week
    GROUP BY week_start
),
calendar_rows AS (
    SELECT
        'harti_week' AS row_kind,
        spine.price_type,
        spine.market_code,
        spine.market_name,
        spine.crop_code,
        spine.crop_name,
        weeks.year,
        weeks.week,
        weeks.week_start,
        weeks.week_end,
        week_season.season_code,
        week_season.season_label,
        harti.price AS harti_price,
        harti.is_missing AS harti_is_missing,
        cbsl_week.cbsl_price,
        cbsl_week.cbsl_is_missing,
        extent_for_week.report_date AS extent_report_date,
        extent_for_week.as_of_date AS extent_as_of_date,
        extent_for_week.season_label AS extent_season_label,
        extent_for_week.target_ha AS extent_target_ha,
        extent_for_week.achievement_ha AS extent_achievement_ha,
        extent_for_week.target_missing AS extent_target_missing,
        extent_for_week.achievement_missing AS extent_achievement_missing,
        weather_week.nuwara_eliya_t2m_c,
        weather_week.nuwara_eliya_precip_mm,
        CAST(weather_week.nuwara_eliya_days AS bigint) AS nuwara_eliya_days,
        weather_week.badulla_t2m_c,
        weather_week.badulla_precip_mm,
        CAST(weather_week.badulla_days AS bigint) AS badulla_days,
        weather_week.dambulla_t2m_c,
        weather_week.dambulla_precip_mm,
        CAST(weather_week.dambulla_days AS bigint) AS dambulla_days
    FROM weeks
    CROSS JOIN spine
    LEFT JOIN cbsl_week
        ON cbsl_week.week_start = weeks.week_start
       AND cbsl_week.price_type = spine.price_type
       AND cbsl_week.market_code = spine.market_code
       AND cbsl_week.crop_code = spine.crop_code
    LEFT JOIN extent_for_week
        ON extent_for_week.week_start = weeks.week_start
       AND extent_for_week.crop_code = spine.crop_code
    LEFT JOIN weather_week
        ON weather_week.week_start = weeks.week_start
    LEFT JOIN week_season
        ON week_season.week_start = weeks.week_start
    LEFT JOIN curated.harti_price_weekly AS harti
        ON CAST(harti.year AS integer) = weeks.year
       AND CAST(harti.week AS integer) = weeks.week
       AND harti.price_type = spine.price_type
       AND harti.market_code = spine.market_code
       AND harti.crop_code = spine.crop_code
)
SELECT * FROM calendar_rows
"""


def load_view() -> dict[str, int]:
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        con.execute(VIEW_SQL)
        counts = _check(con)
    finally:
        con.close()
    return counts


def _check(con) -> dict[str, int]:
    leaked = con.execute(
        f"""
        SELECT count(*)
        FROM curated.{VIEW_NAME}
        WHERE extent_report_date > week_end
           OR extent_as_of_date > week_end
        """
    ).fetchone()[0]
    if leaked:
        raise ValueError(f"{leaked} rows use an extent report from after the Thursday")

    not_friday = con.execute(
        f"""
        SELECT count(*)
        FROM curated.{VIEW_NAME}
        WHERE CAST(extract(isodow FROM week_start) AS integer) <> 5
           OR week_end <> week_start + 6
        """
    ).fetchone()[0]
    if not_friday:
        raise ValueError("Week bounds are not Friday through Thursday")

    shown = con.execute(
        f"""
        SELECT cast(week_start AS varchar), cast(week_end AS varchar), round(harti_price, 2)
        FROM curated.{VIEW_NAME}
        WHERE year = 2025 AND week = 1
          AND market_code = 'colombo' AND crop_code = 'carrot' AND price_type = 'wholesale'
        """
    ).fetchone()
    if shown != ("2025-01-03", "2025-01-09", 244.44):
        raise ValueError(f"2025 W1 should be 3-9 January with wholesale carrot 244.44, got {shown}")

    week_two = con.execute(
        f"""
        SELECT round(harti_price, 2)
        FROM curated.{VIEW_NAME}
        WHERE year = 2025 AND week = 2
          AND market_code = 'colombo' AND crop_code = 'carrot' AND price_type = 'wholesale'
        """
    ).fetchone()
    if week_two != (179.0,):
        raise ValueError(f"2025 W2 wholesale carrot should be 179, got {week_two}")

    week_53 = con.execute(
        f"""
        SELECT count(*)
        FROM curated.{VIEW_NAME}
        WHERE year = 2024 AND week = 53 AND harti_price IS NOT NULL
        """
    ).fetchone()[0]
    if week_53:
        raise ValueError("The workbook has no W53, so 2024 week 53 must not carry a HARTI price")

    tomato = con.execute(
        f"""
        SELECT count(*)
        FROM curated.{VIEW_NAME}
        WHERE crop_code = 'tomato' AND extent_report_date IS NOT NULL
        """
    ).fetchone()[0]
    if tomato:
        raise ValueError("Tomato has no checked extent snapshot")

    early_october = con.execute(
        f"""
        SELECT count(*)
        FROM curated.{VIEW_NAME}
        WHERE crop_code = 'leeks'
          AND extent_report_date = DATE '2025-10-15'
          AND week_end < DATE '2025-10-15'
        """
    ).fetchone()[0]
    if early_october:
        raise ValueError("The October 2025 leeks report was attached to an earlier week")

    july_week = con.execute(
        f"""
        SELECT extent_target_ha, extent_achievement_ha
        FROM curated.{VIEW_NAME}
        WHERE crop_code = 'leeks'
          AND week_start <= DATE '2026-07-15'
          AND week_end >= DATE '2026-07-15'
        LIMIT 1
        """
    ).fetchone()
    if july_week != (1341.0, 761.0):
        raise ValueError(f"The week containing 15 July 2026 should see the July leeks snapshot, got {july_week}")

    same_column = con.execute(
        f"""
        SELECT count(*)
        FROM curated.{VIEW_NAME}
        WHERE harti_price IS NOT NULL AND cbsl_price IS NOT NULL AND harti_price = cbsl_price AND market_code = 'meegoda'
        """
    ).fetchone()[0]
    if same_column:
        raise ValueError("Meegoda has no CBSL series")

    filled = con.execute(
        f"""
        SELECT count(*)
        FROM curated.{VIEW_NAME}
        WHERE harti_price IS NOT NULL OR harti_is_missing
        """
    ).fetchone()[0]
    counts = {
        "rows": con.execute(f"SELECT count(*) FROM curated.{VIEW_NAME}").fetchone()[0],
        "harti_prices": filled,
    }
    return {name: int(value) for name, value in counts.items()}


def main() -> None:
    counts = load_view()
    print(f"warehouse: {DUCKDB_PATH}")
    print(f"view: curated.{VIEW_NAME}")
    for kind, count in counts.items():
        print(f"  {kind}: {count:,}")


if __name__ == "__main__":
    main()
