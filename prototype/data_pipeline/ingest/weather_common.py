"""Shared pieces for the weather ingesters.

* ``LOCATIONS``: producing areas and markets used for weather features, with the
  price-dataset location spellings they map to.
* ``request_with_retries``: GET with backoff for flaky connections and 5xx/429.
* ``add_period_keys`` / ``aggregate_periods``: daily -> weekly / monthly tables.

Week systems
------------
``iso``         ISO-8601 week (Monday start, weeks 1..52/53, ``iso_year`` may
                differ from the calendar year around 1 January).
``harti_week``  Matches the HARTI price workbook (verified ~97.3% against the
                published monthly sheet). W1 starts on the first Monday on or
                after 1 January; weeks are Mon–Sun. Years with 53 Mondays
                (e.g. 2024) carry a W53. Do **not** use day-of-year blocks
                (Jan 1–7 = W1); that convention was tested and rejected.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from data_pipeline.config import MIN_DELAY_SECONDS, REQUEST_TIMEOUT, USER_AGENT


@dataclass(frozen=True)
class Location:
    key: str
    name: str
    lat: float
    lon: float
    role: str  # "producer" or "market"
    district: str
    price_aliases: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""
    default: bool = True


# Coordinates checked against the Open-Meteo geocoder (GeoNames) on 2026-09-29;
# all default points are within ~2.5 km of the GeoNames place centroid.
LOCATIONS: tuple[Location, ...] = (
    Location("nuwara_eliya", "Nuwara Eliya", 6.9497, 80.7891, "producer", "Nuwara Eliya",
             ("N'Eliya", "Nuwara Eliya"), "GeoNames town centroid 6.9708, 80.7829 (1,868 m)"),
    Location("welimada", "Welimada", 6.9030, 80.9130, "producer", "Badulla", ()),
    Location("bandarawela", "Bandarawela", 6.8290, 80.9870, "producer", "Badulla", ("Bandarawela",)),
    Location("badulla", "Badulla", 6.9934, 81.0550, "producer", "Badulla", ("Badulla",)),
    Location("keppetipola", "Keppetipola", 6.8990, 80.8870, "producer", "Badulla", ("Keppetipola",)),
    Location("dambulla", "Dambulla", 7.8742, 80.6511, "producer", "Matale", ("Dambulla",),
             "Dambulla Dedicated Economic Centre"),
    Location("matale", "Matale", 7.4675, 80.6234, "producer", "Matale", ("Matale",)),
    Location("kandy", "Kandy", 7.2906, 80.6337, "producer", "Kandy", ("Kandy", "kandy")),
    Location("anuradhapura", "Anuradhapura", 8.3114, 80.4037, "producer", "Anuradhapura",
             ("Anuradhapura", "Anuradapura")),
    Location("jaffna", "Jaffna", 9.6615, 80.0255, "producer", "Jaffna", ("Jaffna", "Jafna")),
    Location("kalpitiya", "Kalpitiya (Puttalam)", 8.2330, 79.7600, "producer", "Puttalam", ("Puttalam",),
             "Kalpitiya peninsula; Puttalam town (price location) is 8.036, 79.828"),
    Location("tissamaharama", "Tissamaharama (Hambantota)", 6.2850, 81.2870, "producer", "Hambantota",
             ("Thissamaharama", "Hambanthota", "Hambantota"),
             "Hambantota town (price location) is 6.124, 81.119"),
    Location("monaragala", "Monaragala", 6.8728, 81.3507, "producer", "Monaragala", ("Monaragala",)),
    Location("kurunegala", "Kurunegala", 7.4863, 80.3647, "producer", "Kurunegala", ("Kurunegala",)),
    Location("colombo_pettah", "Colombo (Pettah)", 6.9375, 79.8508, "market", "Colombo", ("Colombo",),
             "Pettah / Manning Market area"),
    Location("narahenpita", "Narahenpita", 6.8900, 79.8770, "market", "Colombo", ("Colombo",),
             "Narahenpita Economic Centre"),
    # Optional: wholesale centres present in the price workbook but not in the core list.
    Location("meegoda", "Meegoda", 6.8479, 80.0484, "market", "Colombo", ("Meegoda",),
             "Meegoda Dedicated Economic Centre", default=False),
    Location("thambuttegama", "Thambuttegama", 8.1500, 80.3000, "market", "Anuradhapura",
             ("Tabuththegama", "Tabuttegama"), "Thambuttegama Dedicated Economic Centre", default=False),
    Location("embilipitiya", "Embilipitiya", 6.3439, 80.8489, "producer", "Ratnapura", ("Embilipitiya",),
             default=False),
    Location("hambantota_town", "Hambantota (town)", 6.1241, 81.1185, "market", "Hambantota",
             ("Hambanthota", "Hambantota"), default=False),
)


def select_locations(keys: str | None = None, include_extra: bool = False) -> list[Location]:
    """Return locations by comma-separated key list, or the default set."""
    by_key = {loc.key: loc for loc in LOCATIONS}
    if keys:
        wanted = [k.strip() for k in keys.split(",") if k.strip()]
        unknown = [k for k in wanted if k not in by_key]
        if unknown:
            raise SystemExit(f"Unknown location key(s): {unknown}. Known: {sorted(by_key)}")
        return [by_key[k] for k in wanted]
    return [loc for loc in LOCATIONS if loc.default or include_extra]


def locations_frame(locations: list[Location] | None = None) -> pd.DataFrame:
    locs = list(LOCATIONS) if locations is None else locations
    return pd.DataFrame(
        {
            "location": [l.key for l in locs],
            "name": [l.name for l in locs],
            "role": [l.role for l in locs],
            "district": [l.district for l in locs],
            "lat": [l.lat for l in locs],
            "lon": [l.lon for l in locs],
            "price_aliases": ["|".join(l.price_aliases) for l in locs],
            "default_set": [l.default for l in locs],
            "note": [l.note for l in locs],
        }
    )


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


class RateLimited(Exception):
    """Raised when the server answers 429 and the caller should decide what to do."""

    def __init__(self, response: requests.Response):
        super().__init__(f"HTTP 429: {response.text[:300]}")
        self.response = response


def request_with_retries(
    session: requests.Session,
    url: str,
    params: dict,
    *,
    timeout: float = REQUEST_TIMEOUT,
    retries: int = 5,
    backoff: float = 10.0,
    raise_on_429: bool = True,
) -> requests.Response:
    """GET with exponential backoff on connection errors, timeouts and 5xx.

    4xx responses are returned as-is (except 429, which raises ``RateLimited``
    when ``raise_on_429`` is set) so the caller can log the status and move on.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            wait = backoff * (2**attempt)
            print(f"    network error ({type(exc).__name__}); retry {attempt + 1}/{retries} in {wait:.0f}s")
            time.sleep(wait)
            continue
        if resp.status_code == 429 and raise_on_429:
            raise RateLimited(resp)
        if resp.status_code >= 500:
            wait = backoff * (2**attempt)
            print(f"    HTTP {resp.status_code}; retry {attempt + 1}/{retries} in {wait:.0f}s")
            time.sleep(wait)
            continue
        return resp
    if last_exc is not None:
        raise last_exc
    return resp


def polite_sleep(seconds: float | None = None) -> None:
    time.sleep(max(MIN_DELAY_SECONDS, seconds or 0.0))


FETCH_LOG_FIELDS = ["fetched_at_utc", "source", "location", "status", "bytes", "weight", "url", "message"]


def append_fetch_log(log_path: Path, **row) -> None:
    new = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row.setdefault("fetched_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FETCH_LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FETCH_LOG_FIELDS})


def read_fetch_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=FETCH_LOG_FIELDS)
    df = pd.read_csv(log_path)
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], utc=True, format="ISO8601")
    return df


def harti_first_monday(year: int | pd.Series) -> pd.Timestamp | pd.Series:
    """First Monday on or after 1 January of ``year`` (HARTI W1 start)."""
    if isinstance(year, pd.Series):
        jan1 = pd.to_datetime(dict(year=year.astype(int), month=1, day=1))
        return jan1 + pd.to_timedelta((7 - jan1.dt.weekday) % 7, unit="D")
    jan1 = pd.Timestamp(year=int(year), month=1, day=1)
    return jan1 + pd.Timedelta(days=(7 - jan1.weekday()) % 7)


def assign_harti_week(dates: pd.Series) -> pd.DataFrame:
    """Map each calendar date to HARTI (year, week, week_start Monday).

    Days before a year's first Monday belong to the previous year's last week.
    """
    d = pd.to_datetime(dates)
    monday = d - pd.to_timedelta(d.dt.weekday, unit="D")
    year = monday.dt.year.astype(int)
    first = harti_first_monday(year)
    before = monday < first
    year = year.where(~before, year - 1).astype(int)
    first = harti_first_monday(year)
    week = ((monday - first).dt.days // 7 + 1).astype(int)
    return pd.DataFrame({"year": year.to_numpy(), "harti_week": week.to_numpy(), "week_start": monday.dt.normalize()})


def add_period_keys(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar year/month, HARTI week keys, and ISO week keys."""
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    out["calendar_year"] = d.dt.year
    out["month"] = d.dt.month
    harti = assign_harti_week(d)
    # ``year`` means HARTI year when used with harti_week / cal_week.
    out["year"] = harti["year"].to_numpy()
    out["harti_week"] = harti["harti_week"].to_numpy()
    out["week_start"] = harti["week_start"].to_numpy()
    # Legacy alias: values are HARTI weeks (not day-of-year blocks).
    out["cal_week"] = out["harti_week"]
    iso = d.dt.isocalendar()
    out["iso_year"] = iso["year"].astype(int)
    out["iso_week"] = iso["week"].astype(int)
    return out


def aggregate_periods(
    daily: pd.DataFrame,
    rules: dict[str, str],
    period: str,
    extra: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Aggregate a tidy daily frame (date, location, lat, lon, vars...) to periods.

    ``period`` is one of ``harti_week`` (alias ``cal_week``), ``iso_week`` or
    ``month``. ``rules`` maps a daily column to ``sum``/``mean``/``max``/``min``;
    sums use the days present. ``extra`` adds named aggregations as
    ``{out_col: (source_col, func)}``. Every row carries ``n_days`` (days with
    data), ``days_in_period`` and ``complete`` (all days present), so partial
    trailing periods are visible.
    """
    keyed = add_period_keys(daily)
    if period in ("harti_week", "cal_week"):
        group = ["location", "lat", "lon", "year", "harti_week"]
        period = "harti_week"
    elif period == "iso_week":
        group = ["location", "lat", "lon", "iso_year", "iso_week"]
    elif period == "month":
        group = ["location", "lat", "lon", "calendar_year", "month"]
    else:
        raise ValueError(period)

    rules = {c: f for c, f in rules.items() if c in keyed.columns}
    named = {c: (c, f) for c, f in rules.items()}
    named.update({k: v for k, v in (extra or {}).items() if v[0] in keyed.columns})
    named["period_start"] = ("date", "min")
    named["period_end"] = ("date", "max")
    first_var = next(iter(rules))
    named["n_days"] = (first_var, "count")
    agg = keyed.groupby(group, sort=True).agg(**named).reset_index()

    for col, func in rules.items():
        if func == "sum":
            counts = keyed.groupby(group, sort=True)[col].count().to_numpy()
            agg.loc[counts == 0, col] = float("nan")

    if period == "harti_week":
        start = harti_first_monday(agg["year"]) + pd.to_timedelta((agg["harti_week"] - 1) * 7, unit="D")
        end = start + pd.Timedelta(days=6)
        agg.insert(agg.columns.get_loc("harti_week") + 1, "week_label", "W" + agg["harti_week"].astype(str))
        agg.insert(agg.columns.get_loc("harti_week") + 1, "cal_week", agg["harti_week"])
    elif period == "iso_week":
        start = pd.to_datetime(
            agg["iso_year"].astype(str) + "-W" + agg["iso_week"].astype(str).str.zfill(2) + "-1", format="%G-W%V-%u"
        )
        end = start + pd.Timedelta(days=6)
    else:
        agg = agg.rename(columns={"calendar_year": "year"})
        start = pd.to_datetime(agg["year"].astype(str) + "-" + agg["month"].astype(str) + "-01")
        end = start + pd.offsets.MonthEnd(0)
    agg["period_start"] = pd.to_datetime(start).dt.date.astype(str)
    agg["period_end"] = pd.to_datetime(end).dt.date.astype(str)
    agg["days_in_period"] = (pd.to_datetime(end) - pd.to_datetime(start)).dt.days + 1
    agg["complete"] = agg["n_days"] == agg["days_in_period"]
    return agg


def completeness_profile(daily: pd.DataFrame, value_cols: list[str], today: pd.Timestamp) -> pd.DataFrame:
    """Per-location coverage: first/last date, expected vs present days, missing
    values per variable, and latency (days between today and the last value)."""
    rows = []
    for loc, g in daily.groupby("location", sort=False):
        dates = pd.to_datetime(g["date"])
        start, end = dates.min(), dates.max()
        expected = (end - start).days + 1
        row = {
            "location": loc,
            "first_date": start.date().isoformat(),
            "last_date": end.date().isoformat(),
            "expected_days": expected,
            "rows": len(g),
            "missing_dates": expected - dates.nunique(),
            "duplicate_dates": int(dates.duplicated().sum()),
        }
        for col in value_cols:
            if col not in g.columns:
                continue
            present = g.loc[g[col].notna(), "date"]
            row[f"na_{col}"] = int(g[col].isna().sum())
            last = pd.to_datetime(present).max() if len(present) else pd.NaT
            row[f"latency_days_{col}"] = (today - last).days if pd.notna(last) else None
        rows.append(row)
    return pd.DataFrame(rows)
