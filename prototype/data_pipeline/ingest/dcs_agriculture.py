"""Download agriculture and retail-price data from the Department of Census and Statistics (DCS).

Sources (all public, https://www.statistics.gov.lk):

* ``highland_timeseries`` - the "Highland Crops Time Series Data" query tool
  (``/HIES/HIES2006_07Website/``). Its AJAX backend ``HighlandCrops.asp`` returns,
  per crop, extent (ha) and production (mt) by district and season (Yala, Maha,
  Total) for 2001 onwards. One HTML table per crop is saved.
* ``highland_pages`` - static Highland Crops pages (seasonal other-field-crops
  table, potato, big onion) and the PDFs they embed (cassava, sugarcane).
* ``big_onion`` - Big Onion survey reports (Yala 2024, Yala 2025).
* ``paddy`` - paddy time-series workbooks (district x season, 1979-2025) and the
  latest season report PDF.
* ``retail_dashboard`` - the Weekly Retail Prices dashboard data file
  (``/DashBoard/Prices/Prices_Data.php``; Colombo district markets, 2017 onwards).
* ``retail_bulletins`` - weekly retail price bulletin PDFs (``DCSB-WRP-*.pdf``).
* ``abstract`` - selected Statistical Abstract tables (highland crop extent and
  production by district/season, crop production, retail and producer prices).
* ``indices`` - volume index of agricultural production and per-capita
  availability pages.

Files are written to ``RAW_DIR/dcs/<subsection>/`` with their original filenames
where the server provides one. Every download is recorded in
``RAW_DIR/dcs/manifest.csv``. Existing files are skipped unless ``--force``.

Run from the ``prototype`` folder::

    python -m data_pipeline.ingest.dcs_agriculture --help
    python -m data_pipeline.ingest.dcs_agriculture                      # everything
    python -m data_pipeline.ingest.dcs_agriculture --sections highland_timeseries retail_dashboard --force
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from data_pipeline.config import MIN_DELAY_SECONDS, REQUEST_TIMEOUT, USER_AGENT, raw_dir

BASE = "https://www.statistics.gov.lk"
AGRI = f"{BASE}/Agriculture/StaticalInformation"
HC_APP = "http://www.statistics.gov.lk/HIES/HIES2006_07Website/"
HC_ASP = HC_APP + "HighlandCrops.asp?getcode="
PRICES_DASHBOARD = f"{BASE}/DashBoard/Prices/"
PRICES_DATA = f"{BASE}/DashBoard/Prices/Prices_Data.php"
RETAIL_BULLETIN_INDEX = f"{BASE}/InflationAndPrices/StaticalInformation/RetailPrices"

# District codes used by the highland crops tool (value attribute of optDistrict).
HC_DISTRICT_CODES = [
    "01", "11", "12", "13", "21", "22", "23", "31", "32", "33", "41", "42", "43", "44",
    "45", "51", "52", "53", "61", "62", "71", "72", "81", "82", "91", "92", "99",
]

SECTIONS = [
    "highland_timeseries", "highland_pages", "big_onion", "paddy",
    "retail_dashboard", "retail_bulletins", "abstract", "indices",
]

PADDY_SERIES = [
    "PaddyExtent_Maha_Season", "PaddyExtent_Yala_Season",
    "Sown_Extent-Maha_Seasons-1979-2025", "Sown_Extent-Yala_Seasons-1979-2025",
    "Harvested_Extent-Maha_Seasons-1979-2025", "Harvested_Extent-Yala_Seasons-1979-2025",
    "Average_Yeild-Maha_Season-1979-2025", "Average_Yeild-Yala_Season-1979-2025",
    "Production-Maha_Season-1979-2025", "Production-Yala_Season-1979-2025",
    "Asweddumaized_Extent_in_acres", "Asweddumaized_Extent_in_hectares",
    "Number_of_Aswaddumized_paddy_parcels",
]

# Statistical Abstract tables worth keeping, matched on the link title.
ABSTRACT_TABLE_PATTERNS = {
    "chapter5": [
        r"cultivated extent by crops",
        r"highland crops",
        r"production of main crops",
        r"production of highland crops",
        r"volume index of agricultur",
    ],
    "chapter8": [
        r"retail prices of (selected )?commodities",
        r"producers?.? prices of selected commodities",
        r"producer prices for selected items of highland",
    ],
}

MANIFEST_FIELDS = [
    "url", "local_path", "bytes", "sha256", "downloaded_at", "http_status",
    "content_type", "section", "description",
]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


@dataclass
class Fetcher:
    """Throttled HTTP client that retries once after a pause."""

    delay: float = MIN_DELAY_SECONDS
    session: requests.Session = field(default_factory=requests.Session)
    failures: list[dict] = field(default_factory=list)
    _last: float = 0.0

    def __post_init__(self) -> None:
        self.session.headers["User-Agent"] = USER_AGENT

    def get(self, url: str, referer: str | None = None, retry_wait: float = 15.0) -> requests.Response | None:
        headers = {"Referer": referer} if referer else {}
        for attempt in (1, 2):
            wait = self.delay - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            error = ""
            try:
                resp = self.session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
                self._last = time.monotonic()
                if resp.status_code == 200:
                    return resp
                error = f"HTTP {resp.status_code}"
            except requests.RequestException as exc:
                self._last = time.monotonic()
                error = f"{type(exc).__name__}: {exc}"
            log(f"  ! attempt {attempt} failed for {url}: {error}")
            if attempt == 1:
                time.sleep(retry_wait)
        self.failures.append({"url": url, "error": error, "at": now_iso()})
        return None


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class Manifest:
    """CSV manifest keyed by local path, preserved across runs."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: dict[str, dict] = {}
        if path.exists():
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    self.rows[row["local_path"]] = row

    def add(self, **row) -> None:
        self.rows[row["local_path"]] = {k: row.get(k, "") for k in MANIFEST_FIELDS}

    def save(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for key in sorted(self.rows):
                writer.writerow(self.rows[key])


class Downloader:
    def __init__(self, root: Path, force: bool = False, dry_run: bool = False):
        self.root = root
        self.force = force
        self.dry_run = dry_run
        self.http = Fetcher()
        self.manifest = Manifest(root / "manifest.csv")

    def page(self, url: str, referer: str | None = None) -> str | None:
        resp = self.http.get(url, referer=referer)
        if resp is None:
            return None
        resp.encoding = resp.encoding or "utf-8"
        return resp.text

    def fetch(self, url: str, dest: Path, section: str, description: str,
              referer: str | None = None, min_bytes: int = 1, use_server_name: bool = False) -> Path | None:
        """Download ``url`` to ``dest`` (or the server-supplied filename in dest's folder)."""
        if dest.exists() and not self.force and dest.stat().st_size >= min_bytes:
            log(f"  = exists {dest.relative_to(self.root)}")
            self._record_existing(url, dest, section, description)
            return dest
        if self.dry_run:
            log(f"  ~ would fetch {url} -> {dest}")
            return None
        resp = self.http.get(url, referer=referer)
        if resp is None:
            return None
        if use_server_name:
            name = _filename_from_disposition(resp.headers.get("content-disposition", ""))
            if name:
                dest = dest.with_name(name)
        content = resp.content
        if len(content) < min_bytes:
            log(f"  ! {url} returned {len(content)} bytes (< {min_bytes}); recorded as failure")
            self.http.failures.append({"url": url, "error": f"only {len(content)} bytes", "at": now_iso()})
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        self.manifest.add(
            url=url, local_path=str(dest), bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(), downloaded_at=now_iso(),
            http_status=resp.status_code, content_type=resp.headers.get("content-type", ""),
            section=section, description=description,
        )
        log(f"  + {len(content):>9,d} B  {dest.relative_to(self.root)}")
        return dest

    def _record_existing(self, url: str, dest: Path, section: str, description: str) -> None:
        if str(dest) in self.manifest.rows:
            return
        data = dest.read_bytes()
        self.manifest.add(
            url=url, local_path=str(dest), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
            downloaded_at=datetime.fromtimestamp(dest.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            http_status="", content_type="", section=section, description=description,
        )

    def embedded_pdf(self, page_url: str, folder: Path, section: str, description: str) -> Path | None:
        """Fetch a DCS page that shows a PDF in an iframe and download that PDF."""
        html = self.page(page_url)
        if not html:
            return None
        frame = BeautifulSoup(html, "lxml").find("iframe", src=True)
        if not frame:
            log(f"  ! no iframe on {page_url}")
            self.http.failures.append({"url": page_url, "error": "no iframe/PDF on page", "at": now_iso()})
            return None
        pdf_url = urljoin(page_url, frame["src"].strip())
        name = unquote(Path(urlparse(pdf_url).path).name)
        return self.fetch(pdf_url, folder / name, section, description, referer=page_url, min_bytes=1000)

    def finish(self) -> None:
        if self.dry_run:
            return
        self.manifest.save()
        fail_path = self.root / "fetch_failures.csv"
        if self.http.failures:
            new = not fail_path.exists()
            with fail_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["url", "error", "at"])
                if new:
                    writer.writeheader()
                writer.writerows(self.http.failures)
        log(f"manifest: {self.manifest.path} ({len(self.manifest.rows)} rows); "
            f"failures this run: {len(self.http.failures)}")


def _filename_from_disposition(value: str) -> str | None:
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', value or "", re.I)
    return unquote(match.group(1)) if match else None


def _split_pairs(text: str) -> list[tuple[str, str]]:
    parts = text.split("~")
    return [(parts[i], parts[i + 1]) for i in range(0, len(parts) - 1, 2)]


# --------------------------------------------------------------------------- sections


def highland_timeseries(dl: Downloader, crop_filter: list[str] | None = None) -> None:
    """One extent+production table (all districts, all years) per crop from HighlandCrops.asp."""
    folder = dl.root / "highland_crops_timeseries"
    folder.mkdir(parents=True, exist_ok=True)
    catalogue: dict = {"retrieved_at": now_iso(), "app": HC_APP, "categories": []}
    resp = dl.http.get(HC_ASP + "category", referer=HC_APP)
    if resp is None:
        log("  ! highland crops tool unreachable")
        return
    for cat_id, cat_name in _split_pairs(resp.text):
        if cat_id == "-1":
            continue
        resp = dl.http.get(HC_ASP + f"crop&cropCat={cat_id}", referer=HC_APP)
        crops = [] if resp is None else [(c, n) for c, n in _split_pairs(resp.text) if c != "-1"]
        entry = {"category_id": cat_id, "category": cat_name, "crops": []}
        for crop_id, crop_name in crops:
            if crop_filter and crop_id not in crop_filter and crop_name.lower() not in crop_filter:
                continue
            resp = dl.http.get(HC_ASP + f"yearF&cropID={crop_id}", referer=HC_APP)
            years = [] if resp is None else [(c, y) for c, y in _split_pairs(resp.text) if c != "-1"]
            entry["crops"].append({"crop_id": crop_id, "crop": crop_name, "years": [y for _, y in years]})
            if not years:
                log(f"  ! no years for crop {crop_id} {crop_name}")
                continue
            code = (f"Table&dC={len(HC_DISTRICT_CODES)}&dL={''.join(HC_DISTRICT_CODES)}"
                    f"&P=1&E=1&C={crop_id}&F={years[0][0]}&T={years[-1][0]}")
            dest = folder / f"HighlandCrops_C{int(crop_id):03d}_{slug(crop_name)}.html"
            dl.fetch(HC_ASP + code, dest, "highland_timeseries",
                     f"Extent (ha) & production (mt) by district, Yala/Maha/Total, {years[0][1]}-{years[-1][1]}: "
                     f"{crop_name} [{cat_name}]", referer=HC_APP, min_bytes=500)
        catalogue["categories"].append(entry)
    if not dl.dry_run:
        path = folder / "catalogue.json"
        path.write_text(json.dumps(catalogue, indent=2), encoding="utf-8")
        dl.manifest.add(url=HC_ASP + "category", local_path=str(path), bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), downloaded_at=now_iso(),
                        http_status=200, content_type="application/json", section="highland_timeseries",
                        description="Category/crop/year catalogue of the highland crops tool (built from AJAX lists)")
        dl.fetch(HC_APP, folder / "HIES2006_07Website_index.html", "highland_timeseries",
                 "Landing page of the highland crops query tool (district codes)")


def highland_pages(dl: Downloader) -> None:
    folder = dl.root / "highland_crops_pages"
    pages = {
        "HighlandCrops.html": (f"{AGRI}/HighlandCrops", "Highland Crops index page"),
        "SeasonalCrops.html": (f"{AGRI}/SeasonalCrops",
                               "Extent (ha) & production (mt) of seasonal crops, national, 2006/07 Maha-2018/19 Maha"),
        "Potato.html": (f"{AGRI}/Highlandcrops/Potato",
                        "Potato extent/production by district 2007-2018 and monthly harvest"),
        "BigOnion.html": (f"{AGRI}/Highlandcrops/BigOnion",
                          "Big onion production, yield, extent, farmers by district (Yala) 2004-2023"),
    }
    for name, (url, desc) in pages.items():
        dl.fetch(url, folder / name, "highland_pages", desc, min_bytes=1000)
    for page, desc in [
        (f"{AGRI}/Highlandcrops/cassava", "Cassava extent and production, national total (PDF)"),
        (f"{AGRI}/Highlandcrops/sugarcanenationaltotal", "Sugarcane extent and production, national total (PDF)"),
    ]:
        dl.embedded_pdf(page, folder, "highland_pages", desc)


def big_onion(dl: Downloader) -> None:
    folder = dl.root / "big_onion_survey"
    for page, desc in [
        (f"{AGRI}/new/BigOnion_Survey_SL_Yala2025", "Big Onion Survey in Sri Lanka, Yala 2025 (PDF report)"),
        (f"{AGRI}/new/ExtentandproductionofBigOnion2024Yala", "Extent and production of Big Onion, 2024 Yala (PDF)"),
    ]:
        dl.embedded_pdf(page, folder, "big_onion", desc)


def paddy(dl: Downloader, season_pdfs: bool = False) -> None:
    folder = dl.root / "paddy_statistics"
    for series in PADDY_SERIES:
        dl.fetch(f"{AGRI}/PaddyStatistics/{series}", folder / f"{series}.xlsx", "paddy",
                 f"Paddy time series workbook: {series.replace('_', ' ')}", min_bytes=1000, use_server_name=True)
    dl.embedded_pdf(f"{AGRI}/PaddyStatistics/PaddyStatistics/2025YalaSeason", folder, "paddy",
                    "Paddy Statistics 2025 Yala Season report (crop cutting survey)")
    if not season_pdfs:
        return
    html = dl.page(f"{AGRI}/PaddyStatistics")
    if not html:
        return
    for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
        href = urljoin(f"{AGRI}/PaddyStatistics", a["href"])
        if "/MetricUnits/" in href:
            dl.embedded_pdf(href, folder / "season_tables_metric", "paddy",
                            f"Paddy district table (metric units): {href.rsplit('/', 1)[-1]}")


def retail_dashboard(dl: Downloader) -> None:
    folder = dl.root / "weekly_retail_prices"
    dl.fetch(PRICES_DASHBOARD, folder / "dashboard_index.html", "retail_dashboard",
             "Weekly Retail Prices dashboard page (market coverage notes)", min_bytes=1000)
    dl.fetch(PRICES_DATA, folder / "Prices_Data.php", "retail_dashboard",
             "Dashboard data file: JS arrays pip (products), prices (weekly avg Rs), uvbl (unavailability notes); "
             "main markets in Colombo district", referer=PRICES_DASHBOARD, min_bytes=100_000)


def retail_bulletins(dl: Downloader, mode: str = "recent", count: int = 8) -> None:
    if mode == "none":
        return
    folder = dl.root / "weekly_retail_prices" / "bulletins"
    html = dl.page(RETAIL_BULLETIN_INDEX)
    if not html:
        return
    pages = []
    for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
        href = urljoin(RETAIL_BULLETIN_INDEX, a["href"].strip())
        if re.search(r"DCSB-WRP-\d{4}-\d{2}-W\d$", href):
            pages.append((href, " ".join(a.get_text().split())))
    pages = sorted(set(pages), key=lambda p: p[0], reverse=True)
    log(f"  {len(pages)} weekly bulletins listed")
    if mode == "recent":
        pages = pages[:count]
    for href, label in pages:
        name = href.rsplit("/", 1)[-1] + ".pdf"
        pdf_url = f"{BASE}/Resource/en/InflationAndPrices/retail/{name}"
        if (folder / name).exists() and not dl.force:
            dl.fetch(pdf_url, folder / name, "retail_bulletins", f"Weekly retail price bulletin, {label}")
            continue
        dl.embedded_pdf(href, folder, "retail_bulletins", f"Weekly retail price bulletin, {label}")


def abstract(dl: Downloader, years: list[int]) -> None:
    for year in years:
        for chapter, patterns in ABSTRACT_TABLE_PATTERNS.items():
            chap_url = f"{BASE}/abstract{year}/{chapter}"
            html = dl.page(chap_url)
            if not html or len(html) < 1000:
                log(f"  ! abstract {year} {chapter} not available")
                dl.http.failures.append({"url": chap_url, "error": "page missing/empty", "at": now_iso()})
                continue
            for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
                title = " ".join(a.get_text().split())
                href = urljoin(chap_url, a["href"].strip())
                if not href.lower().endswith(".pdf"):
                    continue
                if any(re.search(p, title, re.I) for p in patterns):
                    rel = Path(unquote(urlparse(href).path)).relative_to(f"/abstract{year}")
                    dl.fetch(href, dl.root / "statistical_abstract" / str(year) / rel, "abstract",
                             f"Statistical Abstract {year}: {title}", referer=chap_url, min_bytes=1000)


def indices(dl: Downloader) -> None:
    folder = dl.root / "other_indices"
    for name, url, desc in [
        ("VolumeIndexofAgriculturalProduction.html", f"{AGRI}/VolumeIndexofAgriculturalProduction",
         "Volume index of agricultural production 2014-2025 (HTML table)"),
        ("PerCapitaAvailabilities.html", f"{AGRI}/PerCapitaAvailabilities",
         "Production, imports and per-capita availability of rice, sugar, pulses, potatoes, onions (HTML)"),
    ]:
        dl.fetch(url, folder / name, "indices", desc, min_bytes=1000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download DCS agriculture statistics and weekly retail prices into RAW_DIR/dcs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sections", nargs="+", choices=SECTIONS + ["all"], default=["all"],
                        help="which sub-sections to download")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument("--dry-run", action="store_true", help="list what would be fetched without saving")
    parser.add_argument("--crops", nargs="*", default=None,
                        help="limit highland_timeseries to these crop ids or names (e.g. 31 carrot)")
    parser.add_argument("--bulletins", choices=["none", "recent", "all"], default="recent",
                        help="which weekly retail bulletin PDFs to fetch")
    parser.add_argument("--bulletin-count", type=int, default=8, help="number of bulletins for --bulletins recent")
    parser.add_argument("--abstract-years", nargs="+", type=int, default=[2022, 2023, 2024, 2025],
                        help="Statistical Abstract editions to scan")
    parser.add_argument("--paddy-season-pdfs", action="store_true",
                        help="also fetch every per-season paddy district table PDF (about 43 files)")
    args = parser.parse_args(argv)

    sections = SECTIONS if "all" in args.sections else args.sections
    dl = Downloader(raw_dir("dcs"), force=args.force, dry_run=args.dry_run)
    crops = [c.lower() for c in args.crops] if args.crops else None
    try:
        for section in sections:
            log(f"== {section}")
            if section == "highland_timeseries":
                highland_timeseries(dl, crops)
            elif section == "highland_pages":
                highland_pages(dl)
            elif section == "big_onion":
                big_onion(dl)
            elif section == "paddy":
                paddy(dl, args.paddy_season_pdfs)
            elif section == "retail_dashboard":
                retail_dashboard(dl)
            elif section == "retail_bulletins":
                retail_bulletins(dl, args.bulletins, args.bulletin_count)
            elif section == "abstract":
                abstract(dl, args.abstract_years)
            elif section == "indices":
                indices(dl)
    finally:
        dl.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
