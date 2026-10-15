"""HARTI Daily Food Price Bulletin Extractor and Ingestion Pipeline.

Extracts agricultural commodity prices from the Hector Kobbekaduwa Agrarian Research
and Training Institute (HARTI) daily price bulletins.

Portal: https://www.harti.gov.lk/daily-price.php
URL pattern: https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/{YYYY}/{Month}/daily_{DD-MM-YYYY}.pdf

Note on Source Limitations:
1. Automated Web Retrieval: As of late 2026, recent bulletin links on the HARTI portal
   (e.g., October 2026 entries) frequently return HTTP 404 or omit the .pdf extension.
2. PDF Font Encoding: HARTI multi-page bulletins (11 pages) embed custom subset fonts
   (e.g., Iskoola Pota, Nirmala UI) with sparse /ToUnicode CMaps.
3. Fallback Mechanism: This extractor provides a robust live crawler with exponential
   backoff, accompanied by an explicit offline bulletin ingestion pipeline
   (`ingest_offline_bulletin` and `ingest_tabular_bulletin`) that guarantees
   cryptographic provenance and full preservation of Min/Max/Modal prices without fabrication.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pipeline.config import (
    CROP_NAME_MAPPINGS,
    HARTI_CONFIG,
    MARKET_CENTERS,
    RAW_HARTI_DIR,
    TARGET_CROPS,
)
from pipeline.extractors.base import BaseExtractor

logger = logging.getLogger("extractor.harti")


class HartiPriceExtractor(BaseExtractor):
    """Extractor and offline ingestion interface for HARTI daily food price bulletins."""

    def __init__(self, raw_output_dir: Path = RAW_HARTI_DIR) -> None:
        super().__init__(source_name="HARTI")
        self.raw_output_dir = Path(raw_output_dir)
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def parse_date_from_filename(filename: str) -> Optional[datetime]:
        """Extract date from various HARTI bulletin filename conventions.

        Supported patterns:
        - Modern 2026 portal format: 'Vegetable Pricenew ex1(2026.10.07).pdf' -> 2026-10-07
        - Alternative delimited format: 'Vegetables (2026-10-04).pdf' -> 2026-10-04
        - Canonical historical format: 'daily_31-12-2024.pdf' -> 2024-12-31
        """
        # Pattern 1: YYYY.MM.DD or YYYY-MM-DD
        m1 = re.search(r"(\d{4})[.-](\d{2})[.-](\d{2})", filename)
        if m1:
            try:
                y, m, d = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
                return datetime(y, m, d)
            except ValueError:
                pass

        # Pattern 2: DD-MM-YYYY or DD_MM_YYYY or DD.MM.YYYY
        m2 = re.search(r"(\d{2})[-_. ](\d{2})[-_. ](\d{4})", filename)
        if m2:
            try:
                d, m, y = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
                return datetime(y, m, d)
            except ValueError:
                pass

        return None

    def scrape_available_bulletin_links(self) -> list[dict[str, str]]:
        """Scrape the HARTI daily price portal for available daily PDF bulletin links.

        Returns:
            List of dictionaries with {'date': 'DD-MM-YYYY', 'iso_date': 'YYYY-MM-DD', 'relative_url': 'assets/...', 'full_url': '...'}
        """
        portal_url = HARTI_CONFIG["portal_url"]
        self.logger.info(f"Querying HARTI price portal: {portal_url}")
        try:
            response = self.session.get(portal_url, timeout=self.timeout)
            response.raise_for_status()
        except Exception as exc:
            self.logger.error(f"Failed to reach HARTI portal at {portal_url}: {exc}")
            return []

        html = response.text
        # Discover relative daily bulletin links
        link_matches = re.findall(
            r"href=[\"']?(assets/pdf/food_price/daily/eng/[^\"'>]+\.pdf)[\"']?",
            html,
            re.IGNORECASE,
        )

        discovered = []
        for rel_link in link_matches:
            dt = self.parse_date_from_filename(rel_link)
            date_str = dt.strftime("%d-%m-%Y") if dt else "unknown"
            iso_str = dt.strftime("%Y-%m-%d") if dt else "unknown"
            full_url = f"https://www.harti.gov.lk/{rel_link.lstrip('/')}"
            discovered.append(
                {
                    "date": date_str,
                    "iso_date": iso_str,
                    "relative_url": rel_link,
                    "full_url": full_url,
                }
            )

        self.logger.info(f"Discovered {len(discovered)} bulletin links on HARTI portal.")
        return discovered

    def format_date_tokens(self, date_input: str | datetime) -> tuple[str, str, str, str]:
        """Convert input date to (iso_date, dd-mm-yyyy, year, month_full)."""
        if isinstance(date_input, str):
            clean = date_input.strip()
            # Normalize dots or slashes to hyphens
            clean_hyphen = re.sub(r"[./]", "-", clean)
            parts = clean_hyphen.split("-")
            if len(parts) == 3 and len(parts[0]) == 4:
                dt = datetime.strptime(clean_hyphen, "%Y-%m-%d")
            elif len(parts) == 3 and len(parts[2]) == 4:
                dt = datetime.strptime(clean_hyphen, "%d-%m-%Y")
            elif len(clean) == 8 and clean.isdigit():
                dt = datetime.strptime(clean, "%Y%m%d")
            else:
                # Try fallback filename parser
                dt = self.parse_date_from_filename(date_input)
                if not dt:
                    raise ValueError(f"Unsupported date format: {date_input}")
        elif isinstance(date_input, datetime):
            dt = date_input
        else:
            raise ValueError(f"Unsupported date format: {date_input}")

        iso_date = dt.strftime("%Y-%m-%d")
        harti_token = dt.strftime("%d-%m-%Y")
        year = dt.strftime("%Y")
        month_full = dt.strftime("%B")
        return iso_date, harti_token, year, month_full

    def build_pdf_url(self, year: str, month: str, date_token: str) -> str:
        """Construct the official HARTI PDF URL according to canonical portal schema."""
        return HARTI_CONFIG["pdf_url_template"].format(
            year=year, month=month, date=date_token
        )

    def fetch_bulletin(
        self, date_input: str | datetime, save_raw: bool = True
    ) -> tuple[bytes, dict[str, Any]]:
        """Attempt to download official HARTI daily bulletin from web portal.

        Resolves either discovered dynamic portal link or canonical archive URL pattern.

        Raises:
            FileNotFoundError: If the remote file returns HTTP 404 or non-PDF payload.
        """
        iso_date, harti_token, year, month = self.format_date_tokens(date_input)

        # 1. Check if the portal published a modern bulletin under a specific filename
        target_url = None
        discovered_links = self.scrape_available_bulletin_links()
        for link in discovered_links:
            if link.get("iso_date") == iso_date or link.get("date") == harti_token:
                target_url = link.get("full_url")
                self.logger.info(f"Resolved dynamic portal bulletin URL for {iso_date}: {target_url}")
                break

        if not target_url:
            target_url = self.build_pdf_url(year, month, harti_token)
            self.logger.info(f"Attempting live fetch of HARTI bulletin for {iso_date} from canonical URL: {target_url}")

        encoded_url = urllib.parse.quote(target_url, safe=":/?#[]@!$&'()*+,;=")
        try:
            response = self.session.get(encoded_url, timeout=self.timeout)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to HARTI server at {target_url}. Error: {exc}. "
                "Consider using ingest_offline_bulletin() fallback."
            ) from exc

        if response.status_code == 404:
            raise FileNotFoundError(
                f"HARTI Daily Price Bulletin for {iso_date} not found at {target_url} (HTTP 404). "
                "Limitation Notice: Recent HARTI daily bulletin URLs frequently return 404 or have broken links on the portal. "
                "Use ingest_offline_bulletin() or ingest_tabular_records() fallback for offline bulletins."
            )
        response.raise_for_status()

        pdf_bytes = response.content
        if not pdf_bytes.startswith(b"%PDF"):
            raise FileNotFoundError(
                f"HARTI server returned non-PDF content (HTTP {response.status_code}) for {target_url}. "
                f"Magic bytes: {pdf_bytes[:20]!r}. "
                "Portal may require offline bulletin ingestion fallback."
            )

        manifest = {}
        if save_raw:
            pdf_path = self.raw_output_dir / f"harti_daily_{harti_token}.pdf"
            manifest_path = self.raw_output_dir / f"harti_daily_{harti_token}.manifest.json"

            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)

            manifest = self.record_manifest(
                manifest_path=manifest_path,
                source_url=target_url,
                parameters={"date": iso_date, "harti_token": harti_token},
                raw_payload_bytes=pdf_bytes,
                extra_metadata={
                    "http_status": response.status_code,
                    "content_type": response.headers.get("Content-Type", ""),
                    "local_raw_path": str(pdf_path),
                },
            )

        return pdf_bytes, manifest

    def ingest_offline_bulletin(
        self,
        file_path: Path | str,
        report_date: Optional[str] = None,
        save_raw: bool = True,
    ) -> tuple[bytes, dict[str, Any]]:
        """Ingest a captured or offline HARTI PDF bulletin, preserving full SHA-256 provenance.

        This serves as the tested, verifiable fallback when live HARTI portal crawling is impaired.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Offline HARTI bulletin not found at {path}")

        with open(path, "rb") as f:
            pdf_bytes = f.read()

        if not report_date:
            dt = self.parse_date_from_filename(path.name)
            if dt:
                report_date = dt.strftime("%Y-%m-%d")
            else:
                raise ValueError(
                    f"Could not infer report date from '{path.name}'. Please pass report_date explicitly."
                )

        iso_date, harti_token, year, month = self.format_date_tokens(report_date)
        sha256_hash = self.compute_sha256(pdf_bytes)

        manifest = {}
        if save_raw:
            dest_pdf = self.raw_output_dir / f"harti_offline_{harti_token}.pdf"
            manifest_path = self.raw_output_dir / f"harti_offline_{harti_token}.manifest.json"

            if path.resolve() != dest_pdf.resolve():
                with open(dest_pdf, "wb") as f:
                    f.write(pdf_bytes)

            manifest = self.record_manifest(
                manifest_path=manifest_path,
                source_url=f"offline://harti_bulletin/{path.name}",
                parameters={"date": iso_date, "harti_token": harti_token},
                raw_payload_bytes=pdf_bytes,
                extra_metadata={
                    "offline_ingestion": True,
                    "original_path": str(path.resolve()),
                    "local_raw_path": str(dest_pdf),
                },
            )

        return pdf_bytes, manifest

    def ingest_tabular_records(
        self,
        raw_records: list[dict[str, Any]],
        report_date: str,
        source_reference: str = "HARTI_OFFLINE_BULLETIN",
        provenance_sha256: str = "",
    ) -> list[dict[str, Any]]:
        """Normalize structured tabular records from a HARTI daily bulletin.

        HARTI bulletins provide Min, Max, and Modal prices for wholesale and retail markets.
        All 5 research crops (Carrot, Cabbage, Leek, Big onion, Tomato) are covered in HARTI.
        """
        iso_date, _, _, _ = self.format_date_tokens(report_date)
        standardized_records: list[dict[str, Any]] = []

        for r in raw_records:
            crop_raw = str(r.get("crop_name", "")).strip()
            norm_crop = CROP_NAME_MAPPINGS.get(crop_raw.lower(), crop_raw)

            market_raw = str(r.get("market_name", "")).strip()
            # Verify market center against known registry
            market_std = MARKET_CENTERS.get(market_raw, {}).get("market_name", market_raw)

            price_type = str(r.get("price_type", "wholesale")).strip().lower()
            if price_type not in ("wholesale", "retail", "producer"):
                price_type = "wholesale"

            min_p = float(r["min_price"]) if r.get("min_price") is not None else None
            max_p = float(r["max_price"]) if r.get("max_price") is not None else None
            modal_p = float(r["modal_price"]) if r.get("modal_price") is not None else None

            # Primary representative price is modal if available, else midpoint or min
            if modal_p is not None:
                rep_price = modal_p
            elif min_p is not None and max_p is not None:
                rep_price = round((min_p + max_p) / 2.0, 2)
            elif min_p is not None:
                rep_price = min_p
            else:
                rep_price = None

            record = {
                "price_date": iso_date,
                "crop_name": norm_crop,
                "source_commodity_name": crop_raw,
                "market_name": market_std,
                "original_market_name": market_raw,
                "price_type": price_type,
                "unit": r.get("unit", "Rs./kg"),
                "price_lkr_per_kg": rep_price,
                "min_price_lkr_per_kg": min_p,
                "max_price_lkr_per_kg": max_p,
                "modal_price_lkr_per_kg": modal_p,
                "raw_price_str": str(r.get("raw_price_str", modal_p or "")),
                "is_target_crop": norm_crop in TARGET_CROPS,
                "is_previous_day": False,
                "source_agency": "HARTI",
                "source_url": source_reference,
                "provenance_sha256": provenance_sha256 or self.compute_sha256(str(r)),
            }
            standardized_records.append(record)

        return standardized_records

    def find_dec_vegetable_stream(self, pdf_bytes: bytes) -> Optional[bytes]:
        """Locate and decompress the PDF stream containing the English DEC wholesale vegetable table.

        Returns:
            Decompressed stream bytes, or None if no matching stream is found (e.g. scanned or corrupt PDF).
        """
        if not pdf_bytes.startswith(b"%PDF"):
            return None

        stream_pattern = re.compile(rb"stream[\r\n]+(.*?)[\r\n]+endstream", re.DOTALL)
        for sm in stream_pattern.finditer(pdf_bytes):
            try:
                d = zlib.decompress(sm.group(1))
                tjs = re.findall(rb"\[(.*?)\]\s*TJ", d)
                if not tjs:
                    continue
                words = []
                for tj in tjs:
                    parts = re.findall(rb"(?<!\\)\(((?:\\.|[^()\\])*)\)", tj)
                    joined = "".join(p.decode("latin1", errors="ignore") for p in parts).strip()
                    if joined:
                        words.append(joined.lower())

                # Table signature: Contains 'variety' and 'peliyagoda'
                if any("variety" in w for w in words) and any("peliyagoda" in w for w in words):
                    return d
            except Exception:
                continue
        return None

    def parse_dec_vegetable_table(
        self,
        decompressed_stream: bytes,
        default_date: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Extract structured commodity rows from the English wholesale DEC table stream."""
        d_str = decompressed_stream.decode("latin1", errors="ignore")

        # Match 1 0 0 1 x y Tm ... [ ... ] TJ text elements
        pattern = re.compile(
            r"1\s+0\s+0\s+1\s+([-\d.]+)\s+([-\d.]+)\s+Tm(?:(?!ET).)*?\[(.*?)\]\s*TJ",
            re.DOTALL,
        )
        elements: list[tuple[float, float, str]] = []
        for m in pattern.finditer(d_str):
            x = float(m.group(1))
            y = float(m.group(2))
            tj = m.group(3)
            parts = re.findall(r"(?<!\\)\(((?:\\.|[^()\\])*)\)", tj)
            txt = "".join(parts).strip()
            txt = re.sub(r"\\([()])", r"\1", txt).strip()
            if txt:
                elements.append((x, y, txt))

        if not elements:
            return []

        # Cluster elements by vertical y-coordinate (tolerance 1.5 pt)
        rows: dict[float, list[tuple[float, str]]] = defaultdict(list)
        for x, y, text in elements:
            matched_y = None
            for ey in rows:
                if abs(ey - y) < 1.5:
                    matched_y = ey
                    break
            if matched_y is None:
                matched_y = y
            rows[matched_y].append((x, text))

        sorted_y = sorted(rows.keys(), reverse=True)

        # 1. Locate Market header row
        mkt_row_y = None
        for y in sorted_y:
            txts = [t[1].lower() for t in rows[y]]
            if any("variety" in t for t in txts) and any("peliyagoda" in t for t in txts):
                mkt_row_y = y
                break

        if mkt_row_y is None:
            self.logger.warning("Could not identify Market header row in HARTI DEC table stream.")
            return []

        mkt_elems = sorted(rows[mkt_row_y], key=lambda t: t[0])
        markets = [(x, txt) for x, txt in mkt_elems if "variety" not in txt.lower()]
        if not markets:
            return []

        # 2. Locate Date row (row with YYYY.MM.DD)
        date_row_y = None
        for y in sorted_y:
            if y != mkt_row_y:
                txts = [t[1] for t in rows[y]]
                if any(re.search(r"\d{4}\.\d{2}\.\d{2}", t) for t in txts):
                    date_row_y = y
                    break

        col_dates: dict[str, str] = {}
        if date_row_y:
            date_elems = sorted(rows[date_row_y], key=lambda t: t[0])
            for x_d, dt_str in date_elems:
                m = re.search(r"(\d{4}\.\d{2}\.\d{2})", dt_str)
                if m:
                    closest_mkt = min(markets, key=lambda m: abs(m[0] - x_d))[1]
                    col_dates[closest_mkt] = m.group(1).replace(".", "-")

        # 3. Determine column boundaries
        # Look for dash '-' positions in sample vegetable rows (e.g. Carrot or Beans)
        sample_dashes: list[float] = []
        for y in sorted_y:
            if y < mkt_row_y - 5:
                row_txts = [t[1].lower() for t in rows[y] if t[0] < 90.0]
                if any("carrot" in t or "beans" in t for t in row_txts):
                    dashes = sorted([t[0] for t in rows[y] if t[1] == "-"])
                    if len(dashes) == len(markets):
                        sample_dashes = dashes
                        break

        col_intervals: list[tuple[str, float, float]] = []
        if len(sample_dashes) == len(markets):
            midpoints = [(sample_dashes[i] + sample_dashes[i+1]) / 2.0 for i in range(len(sample_dashes) - 1)]
            edges = [90.0] + midpoints + [600.0]
            for i in range(len(markets)):
                col_intervals.append((markets[i][1], edges[i], edges[i+1]))
        else:
            # Fallback to market header midpoints
            for i in range(len(markets)):
                m_name, x_c = markets[i][1], markets[i][0]
                left = (markets[i-1][0] + x_c) / 2.0 if i > 0 else 85.0
                right = (x_c + markets[i+1][0]) / 2.0 if i < len(markets) - 1 else 595.0
                col_intervals.append((m_name, left, right))

        # 4. Extract commodity price rows
        extracted_rows: list[dict[str, Any]] = []
        non_commodity_names = {
            "up country vegetable",
            "low country vegetable",
            "hector kobbekaduwa agrarian research and training institute",
            "data management division",
            "market",
            "variety",
            "banana",
            "other fruits (rs/fruit)",
        }

        for y in sorted_y:
            if y >= mkt_row_y - 5:
                continue
            row_elems = sorted(rows[y], key=lambda t: t[0])
            if not row_elems:
                continue

            name_elems = [t[1] for t in row_elems if t[0] < 90.0]
            if not name_elems:
                continue
            crop_name = " ".join(name_elems).strip().rstrip("\\").strip()
            if not crop_name or crop_name.lower() in non_commodity_names:
                continue

            for m_name, lb, rb in col_intervals:
                col_tokens = [t[1] for t in row_elems if lb <= t[0] < rb]
                nums: list[float] = []
                for tok in col_tokens:
                    cleaned = tok.replace(",", "").strip()
                    if cleaned.replace(".", "", 1).isdigit():
                        try:
                            nums.append(float(cleaned))
                        except ValueError:
                            pass

                min_p: Optional[float] = None
                max_p: Optional[float] = None

                if len(nums) >= 2:
                    min_p = min(nums)
                    max_p = max(nums)
                elif len(nums) == 1:
                    min_p = nums[0]
                    max_p = nums[0]

                obs_date = col_dates.get(m_name, default_date or "unknown")
                raw_price_str = "-".join(str(n) for n in nums) if nums else "-"

                extracted_rows.append(
                    {
                        "crop_name": crop_name,
                        "market_name": m_name,
                        "price_date": obs_date,
                        "price_type": "wholesale",
                        "unit": "Rs./kg",
                        "min_price": min_p,
                        "max_price": max_p,
                        "modal_price": None,
                        "raw_price_str": raw_price_str,
                    }
                )

        return extracted_rows

    def extract_from_bytes(
        self,
        pdf_bytes: bytes,
        report_date: Optional[str] = None,
        source_url: str = "",
        sha256_hash: str = "",
        include_previous_day: bool = True,
    ) -> list[dict[str, Any]]:
        """Extract standardized vegetable price observations from HARTI bulletin PDF bytes.

        Args:
            pdf_bytes: Raw bytes of the HARTI daily bulletin PDF.
            report_date: Optional ISO date (YYYY-MM-DD). If omitted, inferred from PDF content or tokens.
            source_url: Provenance source URL.
            sha256_hash: Cryptographic SHA-256 hash of the input document.
            include_previous_day: Whether to retain records whose column date indicates previous day.

        Returns:
            List of standardized price observation dictionaries adhering to warehouse schema.
        """
        if not sha256_hash:
            sha256_hash = self.compute_sha256(pdf_bytes)

        if not pdf_bytes.startswith(b"%PDF"):
            raise ValueError(
                f"Payload does not start with valid PDF magic bytes (%PDF). "
                f"First 50 bytes: {pdf_bytes[:50]!r}"
            )

        dec_stream = self.find_dec_vegetable_stream(pdf_bytes)
        if dec_stream is None:
            self.logger.warning(
                "HARTI PDF font-encoding limitation: Could not detect standard English DEC wholesale table stream. "
                "The PDF may use custom subset glyphs without standard mappings or may be scanned. "
                "Preserved without fabricating records."
            )
            return []

        raw_rows = self.parse_dec_vegetable_table(dec_stream, default_date=report_date)
        if not raw_rows:
            self.logger.warning("No commodity rows extracted from HARTI DEC vegetable table stream.")
            return []

        self.logger.info(f"Parsed {len(raw_rows)} raw market commodity observations from HARTI PDF.")

        # Determine reference bulletin report date
        if not report_date:
            dates_found = [r["price_date"] for r in raw_rows if r["price_date"] != "unknown"]
            if dates_found:
                report_date = max(dates_found)
            else:
                report_date = datetime.now().strftime("%Y-%m-%d")

        norm_records = self.ingest_tabular_records(
            raw_records=raw_rows,
            report_date=report_date,
            source_reference=source_url or "HARTI_PDF_STREAM",
            provenance_sha256=sha256_hash,
        )

        final_records: list[dict[str, Any]] = []
        for r in norm_records:
            is_prev = r["price_date"] != report_date
            r["is_previous_day"] = is_prev
            if is_prev and not include_previous_day:
                continue
            final_records.append(r)

        return final_records

    def extract_from_file(
        self,
        file_path: Path | str,
        report_date: Optional[str] = None,
        include_previous_day: bool = True,
    ) -> list[dict[str, Any]]:
        """Extract price observation records from a local HARTI PDF bulletin file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"HARTI bulletin file not found at {path}")

        with open(path, "rb") as f:
            pdf_bytes = f.read()

        if not report_date:
            dt = self.parse_date_from_filename(path.name)
            if dt:
                report_date = dt.strftime("%Y-%m-%d")

        sha256_hash = self.compute_sha256(pdf_bytes)
        source_url = f"file:///{path.resolve().as_posix()}"

        return self.extract_from_bytes(
            pdf_bytes=pdf_bytes,
            report_date=report_date,
            source_url=source_url,
            sha256_hash=sha256_hash,
            include_previous_day=include_previous_day,
        )
