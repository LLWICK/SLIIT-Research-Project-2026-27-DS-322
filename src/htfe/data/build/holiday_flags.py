"""Weekly Sri Lanka holiday flags aligned to HARTI Mondays."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

HOLIDAY_COLUMNS = (
    "holiday_poya",
    "holiday_sinhala_tamil_new_year",
    "holiday_vesak",
    "holiday_christmas_new_year",
    "holiday_ramadan_end",
    "holiday_deepavali",
    "holiday_any_public",
)


def _calendar(year_min: int, year_max: int) -> dict[date, str]:
    try:
        import holidays
    except ImportError as exc:
        raise ImportError("Install the 'holidays' package to build Sri Lanka holiday flags") from exc

    book = holidays.country_holidays("LK", years=range(year_min, year_max + 1))
    named: dict[date, str] = {}
    for day, label in book.items():
        named[day] = str(label)
    # The April New Year window is wider than the single public holiday.
    for year in range(year_min, year_max + 1):
        for offset in range(13, 18):
            day = date(year, 4, offset)
            named.setdefault(day, "Sinhala and Tamil New Year window")
    return named


def _in_christmas_window(day: date) -> bool:
    return (day.month == 12 and day.day >= 24) or (day.month == 1 and day.day <= 2)


def flags_for_weeks(week_starts: pd.Series) -> pd.DataFrame:
    starts = pd.to_datetime(week_starts).dt.normalize()
    years = range(int(starts.dt.year.min()) - 1, int(starts.dt.year.max()) + 2)
    named = _calendar(min(years), max(years))
    rows = []
    for start in starts.drop_duplicates().sort_values():
        start_day = start.date()
        days = [start_day + timedelta(days=offset) for offset in range(7)]
        labels = [named.get(day, "") for day in days]
        blob = " | ".join(label.lower() for label in labels if label)
        row = {
            "week_start": start,
            "holiday_poya": int("poya" in blob),
            "holiday_sinhala_tamil_new_year": int("sinhala" in blob or "tamil new" in blob),
            "holiday_vesak": int("vesak" in blob),
            "holiday_christmas_new_year": int(any(_in_christmas_window(day) for day in days)),
            "holiday_ramadan_end": int("eid" in blob or "ramazan" in blob or "ramadan" in blob),
            "holiday_deepavali": int("deepavali" in blob or "diwali" in blob),
            "holiday_any_public": int(any(labels)),
        }
        # April window is stored under a generic name; force the new-year flag.
        if any(day.month == 4 and 13 <= day.day <= 17 for day in days):
            row["holiday_sinhala_tamil_new_year"] = 1
            row["holiday_any_public"] = 1
        rows.append(row)
    return pd.DataFrame(rows)
