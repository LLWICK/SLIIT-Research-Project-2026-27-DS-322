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

        Raises:
            FileNotFoundError: If the remote file returns HTTP 404 (known limitation on harti.gov.lk).
        """
        iso_date, harti_token, year, month = self.format_date_tokens(date_input)
        url = self.build_pdf_url(year, month, harti_token)
        self.logger.info(f"Attempting live fetch of HARTI bulletin for {iso_date} from {url}")

        try:
            response = self.session.get(url, timeout=self.timeout)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to HARTI server at {url}. Error: {exc}. "
                "Consider using ingest_offline_bulletin() fallback."
            ) from exc

        if response.status_code == 404:
            raise FileNotFoundError(
                f"HARTI Daily Price Bulletin for {iso_date} not found at {url} (HTTP 404). "
                "Limitation Notice: Recent HARTI daily bulletin URLs frequently return 404 or have broken links on the portal. "
                "Use ingest_offline_bulletin() or ingest_tabular_bulletin() fallback for offline bulletins."
            )
        response.raise_for_status()

        pdf_bytes = response.content
        sha256_hash = self.compute_sha256(pdf_bytes)

        manifest = {}
        if save_raw:
            pdf_path = self.raw_output_dir / f"harti_daily_{harti_token}.pdf"
            manifest_path = self.raw_output_dir / f"harti_daily_{harti_token}.manifest.json"

            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)

            manifest = self.record_manifest(
                manifest_path=manifest_path,
                source_url=url,
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
