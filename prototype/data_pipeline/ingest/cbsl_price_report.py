"""CBSL Daily Price Report: crawl the listing, download PDFs, parse to a tidy CSV.

Source: https://www.cbsl.gov.lk/en/statistics/economic-indicators/price-report

The listing is a Drupal view with 15 reports per page (``?page=0`` is newest).
It only goes back to 2016-08-01. Reports from 2016-03-11 (first issue) to
2016-07-29 were hosted on the pre-2017 site and now survive only in the
Internet Archive; ``crawl --wayback`` adds them to the manifest.

PDF file names are not consistent across years, so always use the manifest
URL instead of guessing:

    price_report_YYYYMMDDe.pdf    2016 - ~2021
    price_report_YYYYMMDD.pdf     many days in 2020-2021
    price_report_YYYYMMDD_e.pdf   ~2022 onwards

Run from the ``prototype`` folder::

    python -m data_pipeline.ingest.cbsl_price_report crawl
    python -m data_pipeline.ingest.cbsl_price_report crawl --wayback
    python -m data_pipeline.ingest.cbsl_price_report download --start 2016-01-01
    python -m data_pipeline.ingest.cbsl_price_report download --start 2016 --weekly
    python -m data_pipeline.ingest.cbsl_price_report parse
    python -m data_pipeline.ingest.cbsl_price_report status

Note: www.cbsl.gov.lk publishes IPv6 (CloudFront) addresses that hang from
some networks. Requests are forced over IPv4 unless ``--allow-ipv6`` is given.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import socket
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from data_pipeline.config import (
    MIN_DELAY_SECONDS,
    REQUEST_TIMEOUT,
    USER_AGENT,
    processed_dir,
    raw_dir,
)

SOURCE = "cbsl_price_report"
SITE = "https://www.cbsl.gov.lk"
LISTING_URL = f"{SITE}/en/statistics/economic-indicators/price-report"
PDF_BASE = f"{SITE}/sites/default/files/cbslweb_documents/statistics/pricerpt/"
OLD_SITE_PREFIX = "www.cbsl.gov.lk/pics_n_docs/01_home/docs/Price_report/"
WAYBACK_CDX = "http://web.archive.org/cdx/search/cdx"
WAYBACK_RAW = "http://web.archive.org/web/{ts}id_/{url}"
LISTING_CUTOFF = date(2016, 8, 1)

MANIFEST_FIELDS = [
    "date", "title", "url", "filename", "lang", "origin", "listing_page",
    "file_date", "date_mismatch", "first_seen",
]
STATUS_FIELDS = [
    "date", "url", "local_path", "status", "http_status", "bytes",
    "seconds", "attempts", "error", "updated_at",
]

TITLE_DATE_RE = re.compile(r"(\d{1,2})\s*(?:st|nd|rd|th)?\s+([A-Za-z]+)\s*,?\s+(\d{4})")
FILE_DATE_RE = re.compile(r"price_?report_?(\d{8})_?([a-z])?\.pdf$", re.I)

log = logging.getLogger(SOURCE)


# --------------------------------------------------------------------------
# paths and small helpers
# --------------------------------------------------------------------------
def base_dir() -> Path:
    return raw_dir(SOURCE)


def manifest_path() -> Path:
    return base_dir() / "manifest.csv"


def status_path() -> Path:
    return base_dir() / "download_status.csv"


def pdf_dir(year: int | str) -> Path:
    path = base_dir() / "pdf" / str(year)
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    path = base_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(command: str, verbose: bool = False) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = logs_dir() / f"{command}_{stamp}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return logfile


def force_ipv4() -> None:
    """Make urllib3 resolve only IPv4 addresses (see module docstring)."""
    import urllib3.util.connection as urllib3_cn

    urllib3_cn.allowed_gai_family = lambda: socket.AF_INET


def parse_date_arg(value: str | None, end: bool = False) -> date | None:
    """Accept YYYY, YYYY-MM or YYYY-MM-DD; ``end`` picks the last day."""
    if not value:
        return None
    parts = value.split("-")
    try:
        if len(parts) == 1:
            return date(int(parts[0]), 12, 31) if end else date(int(parts[0]), 1, 1)
        if len(parts) == 2:
            y, m = int(parts[0]), int(parts[1])
            if end:
                nxt = date(y + (m == 12), m % 12 + 1, 1)
                return date.fromordinal(nxt.toordinal() - 1)
            return date(y, m, 1)
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad date {value!r}: use YYYY, YYYY-MM or YYYY-MM-DD") from exc


def date_from_title(title: str) -> date | None:
    m = TITLE_DATE_RE.search(title)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)[:3]} {m.group(3)}", "%d %b %Y").date()
    except ValueError:
        return None


def date_from_filename(name: str) -> tuple[date | None, str]:
    m = FILE_DATE_RE.search(name)
    if not m:
        return None, ""
    try:
        d = datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        d = None
    return d, (m.group(2) or "").lower()


def is_valid_pdf(path: Path, deep: bool = False) -> bool:
    """Cheap check: size, %PDF header and %%EOF near the end. ``deep`` opens it."""
    try:
        size = path.stat().st_size
        if size < 2048:
            return False
        with path.open("rb") as fh:
            if fh.read(5) != b"%PDF-":
                return False
            fh.seek(max(0, size - 2048))
            if b"%%EOF" not in fh.read():
                return False
        if deep:
            import pymupdf

            with pymupdf.open(path) as doc:
                return doc.page_count > 0
        return True
    except Exception:  # noqa: BLE001 - any failure means "not valid"
        return False


class Throttle:
    """Guarantee at least ``delay`` seconds between request starts."""

    def __init__(self, delay: float):
        self.delay = max(delay, MIN_DELAY_SECONDS)
        self._last = 0.0

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.delay:
            time.sleep(self.delay - gap)
        self._last = time.monotonic()


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en"})
    return s


def get_with_retries(session: requests.Session, url: str, throttle: Throttle, *,
                     retries: int = 4, backoff: float = 2.0, stream: bool = False,
                     params: dict | None = None) -> tuple[requests.Response | None, int, str]:
    """GET with throttling and exponential backoff. 404/410 are not retried."""
    error = ""
    for attempt in range(1, retries + 1):
        throttle.wait()
        try:
            resp = session.get(url, params=params, timeout=(15, REQUEST_TIMEOUT), stream=stream)
            if resp.status_code in (404, 410):
                return resp, attempt, f"HTTP {resp.status_code}"
            if resp.status_code == 200:
                return resp, attempt, ""
            error = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]
        sleep_for = backoff ** attempt
        log.warning("attempt %d/%d failed for %s (%s); sleeping %.0fs", attempt, retries, url, error, sleep_for)
        time.sleep(sleep_for)
    return None, retries, error


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
@dataclass
class ManifestRow:
    date: str
    title: str
    url: str
    filename: str
    lang: str
    origin: str
    listing_page: str
    file_date: str
    date_mismatch: str
    first_seen: str

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in MANIFEST_FIELDS}


def read_manifest() -> list[dict]:
    path = manifest_path()
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_manifest(rows: Iterable[dict]) -> Path:
    rows = sorted(rows, key=lambda r: (r["date"], r["url"]), reverse=True)
    path = manifest_path()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)
    return path


def make_row(title: str, url: str, origin: str, page: str, today: str) -> ManifestRow | None:
    filename = url.rsplit("/", 1)[-1]
    fdate, lang = date_from_filename(filename)
    tdate = date_from_title(title)
    d = tdate or fdate
    if d is None:
        return None
    if lang in ("", "e"):
        lang_out = "en"
    else:
        lang_out = {"s": "si", "t": "ta"}.get(lang, lang)
    return ManifestRow(
        date=d.isoformat(), title=title, url=url, filename=filename, lang=lang_out,
        origin=origin, listing_page=page, file_date=fdate.isoformat() if fdate else "",
        date_mismatch=str(bool(tdate and fdate and tdate != fdate)), first_seen=today,
    )


def parse_listing_page(html: bytes, page: int, today: str) -> tuple[list[ManifestRow], int | None]:
    soup = BeautifulSoup(html, "lxml")
    view = soup.select_one("div.view-price-report") or soup
    rows = []
    for a in view.select("div.views-row a[href]"):
        href = urljoin(SITE, a["href"])
        title = a.get_text(" ", strip=True)
        if not href.lower().endswith(".pdf") or "pricerpt" not in href or not title:
            continue
        row = make_row(title, href, "listing", str(page), today)
        if row:
            rows.append(row)
    last_page = None
    last = soup.select_one("ul.pager li.pager-last a[href]")
    if last:
        m = re.search(r"page=(\d+)", last["href"])
        last_page = int(m.group(1)) if m else None
    return rows, last_page


def crawl_listing(session, throttle, existing: dict[str, dict], max_pages: int | None,
                  incremental: bool) -> list[ManifestRow]:
    today = date.today().isoformat()
    found: list[ManifestRow] = []
    page, last_page = 0, None
    while True:
        resp, _, err = get_with_retries(session, LISTING_URL, throttle, params={"page": page})
        if resp is None or resp.status_code != 200:
            log.error("listing page %d failed: %s", page, err)
            break
        rows, lp = parse_listing_page(resp.content, page, today)
        last_page = lp if lp is not None else last_page
        new = [r for r in rows if r.url not in existing]
        found.extend(rows)
        log.info("page %d/%s: %d reports (%d new) %s .. %s", page, last_page, len(rows), len(new),
                 rows[-1].date if rows else "-", rows[0].date if rows else "-")
        if not rows:
            break
        if incremental and not new:
            log.info("incremental: page %d has nothing new, stopping", page)
            break
        page += 1
        if max_pages is not None and page >= max_pages:
            break
        if last_page is not None and page > last_page:
            break
    return found


def crawl_wayback(session, throttle) -> list[ManifestRow]:
    """Pre-2016-08 reports from the old CBSL site, via the Internet Archive CDX index."""
    today = date.today().isoformat()
    params = {
        "url": OLD_SITE_PREFIX, "matchType": "prefix", "filter": ["statuscode:200", "mimetype:application/pdf"],
        "collapse": "urlkey", "fl": "timestamp,original", "output": "json",
    }
    resp, _, err = get_with_retries(session, WAYBACK_CDX, throttle, params=params)
    if resp is None or resp.status_code != 200:
        log.error("Wayback CDX query failed: %s", err)
        return []
    data = resp.json()
    rows = []
    for ts, original in data[1:]:
        fdate, _ = date_from_filename(original)
        if fdate is None or fdate >= LISTING_CUTOFF:
            continue
        url = WAYBACK_RAW.format(ts=ts, url=original)
        row = make_row(f"Price Report - {fdate:%d %B %Y}", url, "wayback", "", today)
        if row:
            rows.append(row)
    log.info("Wayback: %d archived reports before %s", len(rows), LISTING_CUTOFF)
    return rows


def cmd_crawl(args) -> int:
    logfile = setup_logging("crawl", args.verbose)
    session, throttle = make_session(), Throttle(args.delay)
    existing = {r["url"]: r for r in read_manifest()}
    log.info("crawl start; manifest has %d rows; log %s", len(existing), logfile)
    found: list[ManifestRow] = []
    if not args.skip_listing:
        found += crawl_listing(session, throttle, existing, args.max_pages, args.incremental)
    if args.wayback:
        found += crawl_wayback(session, throttle)
    merged = dict(existing)
    added = 0
    for row in found:
        if row.url in merged:
            keep = merged[row.url]
            keep["listing_page"] = row.listing_page or keep.get("listing_page", "")
            keep["title"] = row.title or keep.get("title", "")
        else:
            merged[row.url] = row.as_dict()
            added += 1
    path = write_manifest(merged.values())
    en = [r for r in merged.values() if r["lang"] == "en"]
    dates = sorted({r["date"] for r in en})
    log.info("manifest %s: %d rows (%d new), %d English report dates %s .. %s",
             path, len(merged), added, len(dates), dates[0] if dates else "-", dates[-1] if dates else "-")
    by_year: dict[str, int] = {}
    for d in dates:
        by_year[d[:4]] = by_year.get(d[:4], 0) + 1
    log.info("English report dates per year: %s", by_year)
    return 0


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------
def select_reports(rows: list[dict], start: date | None, end: date | None, weekly: bool,
                   weekday: int, lang: str = "en", include_wayback: bool = True) -> list[dict]:
    """One manifest row per report date (prefer live listing over Wayback)."""
    per_date: dict[str, dict] = {}
    rank = {"listing": 0, "probe": 1, "wayback": 2}
    for r in rows:
        if r["lang"] != lang or (not include_wayback and r["origin"] == "wayback"):
            continue
        d = date.fromisoformat(r["date"])
        if (start and d < start) or (end and d > end):
            continue
        cur = per_date.get(r["date"])
        if cur is None or rank.get(r["origin"], 9) < rank.get(cur["origin"], 9):
            per_date[r["date"]] = r
    chosen = sorted(per_date.values(), key=lambda r: r["date"])
    if not weekly:
        return chosen
    by_week: dict[tuple[int, int], list[dict]] = {}
    for r in chosen:
        iso = date.fromisoformat(r["date"]).isocalendar()
        by_week.setdefault((iso[0], iso[1]), []).append(r)
    picked = []
    for _, week_rows in sorted(by_week.items()):
        picked.append(min(week_rows, key=lambda r: (abs(date.fromisoformat(r["date"]).weekday() - weekday),
                                                     r["date"])))
    return picked


def local_pdf_path(row: dict) -> Path:
    name = row["filename"]
    if row["origin"] == "wayback":
        name = f"wayback_{name}"
    return pdf_dir(row["date"][:4]) / name


def read_status() -> dict[str, dict]:
    path = status_path()
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["url"]: r for r in csv.DictReader(fh)}


def write_status(status: dict[str, dict]) -> None:
    path = status_path()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=STATUS_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(status.values(), key=lambda r: r["date"]))
    tmp.replace(path)


def download_one(session, throttle, row: dict, retries: int) -> dict:
    dest = local_pdf_path(row)
    rec = {"date": row["date"], "url": row["url"], "local_path": str(dest), "http_status": "",
           "bytes": "", "seconds": "", "attempts": "", "error": "",
           "updated_at": datetime.now().isoformat(timespec="seconds")}
    t0 = time.monotonic()
    resp, attempts, err = get_with_retries(session, row["url"], throttle, retries=retries, stream=True)
    rec["attempts"] = attempts
    if resp is None:
        rec.update(status="failed", error=err, seconds=f"{time.monotonic() - t0:.2f}")
        return rec
    rec["http_status"] = resp.status_code
    if resp.status_code != 200:
        rec.update(status="missing" if resp.status_code in (404, 410) else "failed", error=err,
                   seconds=f"{time.monotonic() - t0:.2f}")
        return rec
    tmp = dest.with_suffix(".part")
    try:
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(64 * 1024):
                fh.write(chunk)
    except requests.RequestException as exc:
        tmp.unlink(missing_ok=True)
        rec.update(status="failed", error=f"stream: {exc}"[:300], seconds=f"{time.monotonic() - t0:.2f}")
        return rec
    if not is_valid_pdf(tmp):
        ctype = resp.headers.get("content-type", "")
        tmp.unlink(missing_ok=True)
        rec.update(status="invalid", error=f"not a complete PDF (content-type {ctype})",
                   seconds=f"{time.monotonic() - t0:.2f}")
        return rec
    tmp.replace(dest)
    rec.update(status="ok", bytes=dest.stat().st_size, seconds=f"{time.monotonic() - t0:.2f}")
    return rec


def cmd_download(args) -> int:
    logfile = setup_logging("download", args.verbose)
    rows = read_manifest()
    if not rows:
        log.error("manifest %s is empty: run `crawl` first", manifest_path())
        return 1
    start, end = parse_date_arg(args.start), parse_date_arg(args.end, end=True)
    todo = select_reports(rows, start, end, args.weekly, args.weekday, include_wayback=not args.no_wayback)
    if args.dates:
        wanted = set(args.dates)
        todo = [r for r in todo if r["date"] in wanted]
    status = read_status()
    pending, skipped = [], 0
    for r in todo:
        dest = local_pdf_path(r)
        if not args.force and dest.exists() and is_valid_pdf(dest):
            skipped += 1
            if r["url"] not in status or status[r["url"]].get("status") != "ok":
                status[r["url"]] = {"date": r["date"], "url": r["url"], "local_path": str(dest), "status": "ok",
                                    "bytes": dest.stat().st_size, "updated_at": datetime.now().isoformat(timespec="seconds")}
            continue
        if not args.retry_missing and status.get(r["url"], {}).get("status") == "missing":
            skipped += 1
            continue
        pending.append(r)
    if args.limit:
        pending = pending[: args.limit]
    log.info("download: %d selected (%s..%s, weekly=%s), %d already present/skipped, %d to fetch; log %s",
             len(todo), start, end, args.weekly, skipped, len(pending), logfile)
    session, throttle = make_session(), Throttle(args.delay)
    t_start = time.monotonic()
    n_ok = n_bytes = 0
    counts: dict[str, int] = {}
    for i, r in enumerate(pending, 1):
        rec = download_one(session, throttle, r, args.retries)
        status[r["url"]] = rec
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
        if rec["status"] == "ok":
            n_ok += 1
            n_bytes += int(rec["bytes"])
        else:
            log.warning("%s %s: %s", r["date"], rec["status"], rec["error"])
        if i % 25 == 0 or i == len(pending):
            elapsed = time.monotonic() - t_start
            rate = elapsed / i
            log.info("%d/%d done (%s); %.2f s/report, %.2f MB/report, ETA %.1f min",
                     i, len(pending), counts, rate, (n_bytes / max(n_ok, 1)) / 1e6,
                     rate * (len(pending) - i) / 60)
            write_status(status)
    write_status(status)
    log.info("download finished: %s in %.1f min", counts, (time.monotonic() - t_start) / 60)
    return 0 if not counts.get("failed") else 2


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------
def cmd_status(args) -> int:
    setup_logging("status", args.verbose)
    rows = [r for r in read_manifest() if r["lang"] == "en"]
    chosen = select_reports(rows, None, None, False, 2)
    by_year: dict[str, list[int]] = {}
    for r in chosen:
        y = r["date"][:4]
        have = local_pdf_path(r).exists()
        by_year.setdefault(y, [0, 0])
        by_year[y][0] += 1
        by_year[y][1] += int(have)
    log.info("year  in_manifest  downloaded")
    for y in sorted(by_year):
        log.info("%s  %11d  %10d", y, *by_year[y])
    tot = [sum(v[0] for v in by_year.values()), sum(v[1] for v in by_year.values())]
    log.info("all   %11d  %10d", *tot)
    return 0


# --------------------------------------------------------------------------
# parse (implementation lives in cbsl_price_parser)
# --------------------------------------------------------------------------
def cmd_parse(args) -> int:
    from data_pipeline.ingest import cbsl_price_parser as parser_mod

    logfile = setup_logging("parse", args.verbose)
    start, end = parse_date_arg(args.start), parse_date_arg(args.end, end=True)
    pdfs = sorted((base_dir() / "pdf").glob("*/*.pdf"))
    out = Path(args.out) if args.out else processed_dir() / "cbsl_daily_prices_long.csv"
    log.info("parse: %d PDFs found; output %s; log %s", len(pdfs), out, logfile)
    return parser_mod.parse_many(pdfs, out, start=start, end=end, workers=args.workers,
                                 keep_all_columns=args.all_columns,
                                 report_path=base_dir() / "parse_report.csv")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m data_pipeline.ingest.cbsl_price_report",
        description="Crawl, download and parse the CBSL Daily Price Report PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Run from the")[0],
    )
    p.add_argument("--allow-ipv6", action="store_true", help="do not force IPv4 (default: IPv4 only)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("crawl", help="build/refresh manifest.csv from the listing (and Wayback)")
    c.add_argument("--max-pages", type=int, help="stop after N listing pages (default: all)")
    c.add_argument("--incremental", action="store_true", help="stop at the first page with no new reports")
    c.add_argument("--wayback", action="store_true", help="also add 2016-03..2016-07 reports from the Internet Archive")
    c.add_argument("--skip-listing", action="store_true", help="only run the Wayback step")
    c.add_argument("--delay", type=float, default=1.2, help="seconds between requests (min %.1f)" % MIN_DELAY_SECONDS)
    c.set_defaults(func=cmd_crawl)

    d = sub.add_parser("download", help="download PDFs listed in the manifest (resumable)")
    d.add_argument("--start", help="first date: YYYY, YYYY-MM or YYYY-MM-DD")
    d.add_argument("--end", help="last date: YYYY, YYYY-MM or YYYY-MM-DD")
    d.add_argument("--dates", nargs="+", help="explicit report dates (YYYY-MM-DD) within the range")
    d.add_argument("--weekly", action="store_true", help="one report per ISO week (closest to --weekday)")
    d.add_argument("--weekday", type=int, default=2, help="0=Mon .. 4=Fri; default 2 (Wednesday)")
    d.add_argument("--limit", type=int, help="fetch at most N reports this run")
    d.add_argument("--delay", type=float, default=1.2, help="seconds between requests (min %.1f)" % MIN_DELAY_SECONDS)
    d.add_argument("--retries", type=int, default=4)
    d.add_argument("--force", action="store_true", help="re-download even if a valid PDF exists")
    d.add_argument("--retry-missing", action="store_true", help="retry URLs that previously returned 404")
    d.add_argument("--no-wayback", action="store_true", help="skip Internet Archive copies")
    d.set_defaults(func=cmd_download)

    q = sub.add_parser("parse", help="parse downloaded PDFs to a tidy long CSV")
    q.add_argument("--start", help="first date: YYYY, YYYY-MM or YYYY-MM-DD")
    q.add_argument("--end", help="last date: YYYY, YYYY-MM or YYYY-MM-DD")
    q.add_argument("--out", help="output CSV (default processed/cbsl_daily_prices_long.csv)")
    q.add_argument("--workers", type=int, default=4, help="parallel processes")
    q.add_argument("--all-columns", action="store_true",
                   help="also keep yesterday / comparison columns (default: today only)")
    q.set_defaults(func=cmd_parse)

    s = sub.add_parser("status", help="count manifest vs downloaded reports per year")
    s.set_defaults(func=cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.allow_ipv6:
        force_ipv4()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
