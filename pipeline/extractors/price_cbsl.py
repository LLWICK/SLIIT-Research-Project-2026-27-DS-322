"""CBSL Daily Price Report Extractor.

Extracts official agricultural commodity price data from the Central Bank of Sri Lanka (CBSL)
Daily Price Reports (PDF format).

Source: https://www.cbsl.gov.lk/en/statistics/economic-indicators/price-report
URL pattern: https://www.cbsl.gov.lk/sites/default/files/cbslweb_documents/statistics/pricerpt/price_report_{YYYYMMDD}_e.pdf
"""

from __future__ import annotations

import logging
import re
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from pipeline.config import (
    CBSL_CONFIG,
    CROP_NAME_MAPPINGS,
    RAW_CBSL_DIR,
    TARGET_CROPS,
)
from pipeline.extractors.base import BaseExtractor

logger = logging.getLogger("extractor.cbsl")


class CbslPriceExtractor(BaseExtractor):
    """Extractor for CBSL daily vegetable price reports published in PDF format."""

    def __init__(self, raw_output_dir: Path = RAW_CBSL_DIR) -> None:
        super().__init__(source_name="CBSL")
        self.raw_output_dir = Path(raw_output_dir)
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)

    def format_date_str(self, date_input: str | datetime) -> tuple[str, str]:
        """Convert input date to ISO (YYYY-MM-DD) and CBSL URL token (YYYYMMDD)."""
        if isinstance(date_input, str):
            clean = date_input.strip()
            if "-" in clean:
                dt = datetime.strptime(clean, "%Y-%m-%d")
            else:
                dt = datetime.strptime(clean, "%Y%m%d")
        elif isinstance(date_input, datetime):
            dt = date_input
        else:
            raise ValueError(f"Unsupported date format: {date_input}")

        iso_date = dt.strftime("%Y-%m-%d")
        cbsl_token = dt.strftime("%Y%m%d")
        return iso_date, cbsl_token

    def build_pdf_url(self, date_token: str) -> str:
        """Construct the official CBSL PDF URL for a given YYYYMMDD date."""
        return CBSL_CONFIG["pdf_url_template"].format(date=date_token)

    def fetch_report(
        self, date_input: str | datetime, save_raw: bool = True
    ) -> tuple[bytes, dict[str, Any]]:
        """Download CBSL daily price report PDF and record cryptographic provenance.

        Args:
            date_input: Date as 'YYYY-MM-DD' or 'YYYYMMDD' or datetime object.
            save_raw: Whether to save the PDF and manifest to RAW_CBSL_DIR.

        Returns:
            Tuple of (pdf_bytes, manifest_dict).
        """
        iso_date, cbsl_token = self.format_date_str(date_input)
        url = self.build_pdf_url(cbsl_token)
        self.logger.info(f"Fetching CBSL price report for {iso_date} from {url}")

        response = self.session.get(url, timeout=self.timeout)
        if response.status_code == 404:
            raise FileNotFoundError(
                f"CBSL Daily Price Report for {iso_date} not found at {url} (HTTP 404). "
                "Note: CBSL does not publish reports on weekends and public holidays."
            )
        response.raise_for_status()

        pdf_bytes = response.content
        if not pdf_bytes.startswith(b"%PDF"):
            raise ValueError(
                f"Response from {url} is not a valid PDF document (magic bytes: {pdf_bytes[:10]!r})"
            )

        manifest = {}
        if save_raw:
            pdf_path = self.raw_output_dir / f"price_report_{cbsl_token}_e.pdf"
            manifest_path = self.raw_output_dir / f"price_report_{cbsl_token}_e.manifest.json"

            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)

            manifest = self.record_manifest(
                manifest_path=manifest_path,
                source_url=url,
                parameters={"date": iso_date, "cbsl_token": cbsl_token},
                raw_payload_bytes=pdf_bytes,
                extra_metadata={
                    "http_status": response.status_code,
                    "content_type": response.headers.get("Content-Type", ""),
                    "local_raw_path": str(pdf_path),
                },
            )

        return pdf_bytes, manifest

    def extract_text_tokens_from_pdf(self, pdf_bytes: bytes) -> list[str]:
        """Extract text tokens from uncompressed PDF streams using standard library."""
        decompressed_chunks: list[bytes] = []

        # Find all PDF stream segments and decompress FlateDecode streams
        stream_matches = re.findall(b"stream[\r\n]+(.*?)[\r\n]+endstream", pdf_bytes, re.DOTALL)
        for chunk in stream_matches:
            try:
                decompressed_chunks.append(zlib.decompress(chunk))
            except Exception:
                # Some streams may not be Flate/zlib or may be images/metadata
                pass

        full_decompressed = b"\n".join(decompressed_chunks)
        tokens: list[str] = []

        # Extract all text tokens inside TJ (kerning array) and Tj (simple string) operators
        # In PDF syntax, parentheses can be escaped: \( and \)
        tj_blocks = re.findall(rb"\[(.*?)\]\s*TJ", full_decompressed, re.DOTALL)
        for tj in tj_blocks:
            parts = re.findall(rb"(?<!\\)\(((?:\\.|[^()\\])*)\)", tj)
            token_text = "".join(p.decode("latin1", errors="ignore") for p in parts)
            token_text = re.sub(r"\\([()])", r"\1", token_text).strip()
            if token_text:
                tokens.append(token_text)

        # Also extract standalone Tj strings
        tj_singles = re.findall(rb"(?<!\\)\(((?:\\.|[^()\\])*)\)\s*Tj", full_decompressed)
        for s in tj_singles:
            token_text = s.decode("latin1", errors="ignore")
            token_text = re.sub(r"\\([()])", r"\1", token_text).strip()
            if token_text and token_text not in tokens:
                tokens.append(token_text)

        return tokens

    def detect_report_date_from_tokens(self, tokens: list[str]) -> Optional[str]:
        """Attempt to extract report date (e.g. '14 October 2026') from text tokens."""
        for t in tokens:
            m = re.search(
                r"(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
                t,
                re.IGNORECASE,
            )
            if m:
                try:
                    dt = datetime.strptime(m.group(1), "%d %B %Y")
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass
        return None

    def parse_commodity_table(
        self, tokens: list[str]
    ) -> list[dict[str, Any]]:
        """Parse structured 10-price-column commodity rows from CBSL token stream.

        CBSL price report layout structure for each item:
        [Item Name, 'Rs./kg',
         w_pettah_prev, w_pettah_today,
         w_dambulla_prev, w_dambulla_today,
         r_pettah_prev, r_pettah_today,
         r_dambulla_prev, r_dambulla_today,
         r_narahenpita_prev, r_narahenpita_today]
        """
        rows: list[dict[str, Any]] = []
        n = len(tokens)

        # Locate the main table start header
        start_idx = 0
        for idx, t in enumerate(tokens):
            if "Wholesale and Retail Prices" in t:
                start_idx = idx
                break

        i = start_idx
        while i < n:
            token = tokens[i]
            # Commodity rows are signaled by the unit 'Rs./kg' following the item name
            if i + 1 < n and tokens[i + 1] == "Rs./kg":
                item_name = token.strip()
                unit = tokens[i + 1]
                price_tokens = tokens[i + 2 : i + 12]

                # Validate that price tokens resemble prices (number or 'n.a.')
                valid_price_tokens = True
                for p in price_tokens:
                    p_clean = p.replace(",", "").strip()
                    if not (p_clean.replace(".", "", 1).isdigit() or p_clean.lower() in ("n.a.", "na", "-")):
                        valid_price_tokens = False
                        break

                if len(price_tokens) == 10 and valid_price_tokens:
                    rows.append(
                        {
                            "item_name": item_name,
                            "unit": unit,
                            "w_pettah_prev": price_tokens[0],
                            "w_pettah_today": price_tokens[1],
                            "w_dambulla_prev": price_tokens[2],
                            "w_dambulla_today": price_tokens[3],
                            "r_pettah_prev": price_tokens[4],
                            "r_pettah_today": price_tokens[5],
                            "r_dambulla_prev": price_tokens[6],
                            "r_dambulla_today": price_tokens[7],
                            "r_narahenpita_prev": price_tokens[8],
                            "r_narahenpita_today": price_tokens[9],
                        }
                    )
                    i += 12
                    continue
            i += 1

        return rows

    def clean_numeric_price(self, val_str: str) -> Optional[float]:
        """Convert price string to float, preserving 'n.a.' or missing values as None.

        Never fabricates a price or assumes 0.0 for missing observations.
        """
        if not val_str:
            return None
        cleaned = val_str.strip().lower().replace(",", "")
        if cleaned in ("n.a.", "na", "-", "", "none", "null"):
            return None
        try:
            return float(cleaned)
        except ValueError:
            self.logger.warning(f"Unrecognized price format '{val_str}', preserving as None")
            return None

    def extract_from_bytes(
        self,
        pdf_bytes: bytes,
        report_date: Optional[str] = None,
        source_url: str = "",
        sha256_hash: str = "",
        include_previous_day: bool = False,
    ) -> list[dict[str, Any]]:
        """Extract standardized price observation records from raw PDF bytes.

        Args:
            pdf_bytes: Raw bytes of the CBSL daily price report.
            report_date: Optional ISO date (YYYY-MM-DD). If omitted, extracted from PDF text.
            source_url: Provenance source URL.
            sha256_hash: Cryptographic hash of the source payload.
            include_previous_day: If True, also emits records for previous business day.

        Returns:
            List of price observation dictionaries.
        """
        if not sha256_hash:
            sha256_hash = self.compute_sha256(pdf_bytes)

        tokens = self.extract_text_tokens_from_pdf(pdf_bytes)

        if not report_date:
            report_date = self.detect_report_date_from_tokens(tokens)
            if not report_date:
                raise ValueError("Could not detect report date from PDF content. Please specify report_date.")

        parsed_rows = self.parse_commodity_table(tokens)
        self.logger.info(
            f"Parsed {len(parsed_rows)} valid commodity rows from CBSL PDF ({report_date})"
        )

        dt_today = datetime.strptime(report_date, "%Y-%m-%d")
        # Approximate previous business day (e.g. Friday if Monday, else -1 day)
        prev_delta = 3 if dt_today.weekday() == 0 else 1
        dt_prev = dt_today - timedelta(days=prev_delta)
        prev_date_str = dt_prev.strftime("%Y-%m-%d")

        records: list[dict[str, Any]] = []
        crops_found = set()

        for row in parsed_rows:
            raw_item = row["item_name"]
            norm_key = raw_item.lower().strip()

            # Map to research target crop if matched
            target_crop = CROP_NAME_MAPPINGS.get(norm_key)

            # Field mappings: (market_name, price_type, is_prev, row_key)
            field_mappings = [
                ("Pettah", "wholesale", False, "w_pettah_today"),
                ("Pettah", "wholesale", True, "w_pettah_prev"),
                ("Dambulla", "wholesale", False, "w_dambulla_today"),
                ("Dambulla", "wholesale", True, "w_dambulla_prev"),
                ("Pettah", "retail", False, "r_pettah_today"),
                ("Pettah", "retail", True, "r_pettah_prev"),
                ("Dambulla", "retail", False, "r_dambulla_today"),
                ("Dambulla", "retail", True, "r_dambulla_prev"),
                ("Narahenpita", "retail", False, "r_narahenpita_today"),
                ("Narahenpita", "retail", True, "r_narahenpita_prev"),
            ]

            if target_crop:
                crops_found.add(target_crop)

            for market_name, price_type, is_prev, col_key in field_mappings:
                if is_prev and not include_previous_day:
                    continue

                raw_val = row[col_key]
                price_val = self.clean_numeric_price(raw_val)
                obs_date = prev_date_str if is_prev else report_date

                record = {
                    "price_date": obs_date,
                    "crop_name": target_crop if target_crop else raw_item,
                    "source_commodity_name": raw_item,
                    "market_name": market_name,
                    "original_market_name": market_name,
                    "price_type": price_type,
                    "unit": row["unit"],
                    "price_lkr_per_kg": price_val,
                    "min_price_lkr_per_kg": None,
                    "max_price_lkr_per_kg": None,
                    "modal_price_lkr_per_kg": price_val,
                    "raw_price_str": raw_val,
                    "is_target_crop": target_crop is not None,
                    "is_previous_day": is_prev,
                    "source_agency": "CBSL",
                    "source_url": source_url,
                    "provenance_sha256": sha256_hash,
                }
                records.append(record)

        # Verification check: Leek is not published in CBSL daily reports
        missing_research_crops = set(TARGET_CROPS) - crops_found
        if "Leek" in missing_research_crops:
            self.logger.info(
                "Verified expected CBSL source limitation: 'Leek' is not published in CBSL daily reports. "
                "Preserved as absent without fabricating records."
            )

        return records

    def extract_from_file(
        self,
        file_path: Path | str,
        report_date: Optional[str] = None,
        include_previous_day: bool = False,
    ) -> list[dict[str, Any]]:
        """Extract price observations from a local CBSL PDF file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"CBSL report file not found at {path}")

        with open(path, "rb") as f:
            pdf_bytes = f.read()

        # Deduce report date from filename if not provided
        if not report_date:
            match = re.search(r"price_report_(\d{8})", path.name)
            if match:
                date_token = match.group(1)
                report_date = f"{date_token[:4]}-{date_token[4:6]}-{date_token[6:8]}"

        sha256_hash = self.compute_sha256(pdf_bytes)
        return self.extract_from_bytes(
            pdf_bytes=pdf_bytes,
            report_date=report_date,
            source_url=f"file:///{path.resolve().as_posix()}",
            sha256_hash=sha256_hash,
            include_previous_day=include_previous_day,
        )
