"""
Department of Agriculture (DOA) Socio Economics and Planning Centre (SEPC) Extractor.
SLIIT Research Project DS-322 (2026/27) - Member 1

Handles extraction from DOA SEPC downloads portal and structured cultivation data.
Official source: https://doa.gov.lk/sepc-downloads_en/
"""

import csv
import datetime
from html.parser import HTMLParser
import io
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union


class UnsupportedSourceFormatError(RuntimeError):
    """Raised when encountering formats that cannot be reliably parsed synchronously (e.g. Google Drive multi-page PDFs)."""
    pass


class DOAPublicationRecord:
    """Catalog entry for publications listed on the DOA SEPC downloads portal."""

    def __init__(
        self,
        category: str,
        title: str,
        year: Optional[int],
        season: Optional[str],
        language: str,
        download_url: str,
        source_page_url: str = "https://doa.gov.lk/sepc-downloads_en/",
        extraction_timestamp: Optional[str] = None,
    ):
        self.category = category
        self.title = title
        self.year = year
        self.season = season
        self.language = language
        self.download_url = download_url
        self.source_page_url = source_page_url
        self.extraction_timestamp = extraction_timestamp or datetime.datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": "DOA_SEPC",
            "category": self.category,
            "title": self.title,
            "year": self.year,
            "season": self.season,
            "language": self.language,
            "download_url": self.download_url,
            "source_page_url": self.source_page_url,
            "extraction_timestamp": self.extraction_timestamp,
        }

    def __repr__(self) -> str:
        return f"<DOAPublicationRecord '{self.title}' ({self.language}) -> {self.download_url}>"


class DOACultivationRecord:
    """Structured cultivation and production forecast record from DOA SEPC reports."""

    def __init__(
        self,
        crop: str,
        district: str,
        season: str,
        year: int,
        target_extent_ha: Optional[float] = None,
        cultivated_extent_ha: Optional[float] = None,
        expected_production_mt: Optional[float] = None,
        raw_source: str = "",
        extraction_timestamp: Optional[str] = None,
    ):
        self.crop = crop
        self.district = district
        self.season = season
        self.year = year
        self.target_extent_ha = target_extent_ha
        self.cultivated_extent_ha = cultivated_extent_ha
        self.expected_production_mt = expected_production_mt
        self.raw_source = raw_source
        self.extraction_timestamp = extraction_timestamp or datetime.datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": "DOA_SEPC",
            "crop": self.crop,
            "district": self.district,
            "season": self.season,
            "year": self.year,
            "target_extent_ha": self.target_extent_ha,
            "cultivated_extent_ha": self.cultivated_extent_ha,
            "expected_production_mt": self.expected_production_mt,
            "raw_source": self.raw_source,
            "extraction_timestamp": self.extraction_timestamp,
        }

    def __repr__(self) -> str:
        return (
            f"<DOACultivationRecord crop={self.crop} district={self.district} "
            f"season={self.season} {self.year} cultivated_ha={self.cultivated_extent_ha}>"
        )


class _DOAPublicationHTMLParser(HTMLParser):
    """Parses DOA SEPC downloads page HTML to extract publication titles and download links."""

    def __init__(self):
        super().__init__()
        self.publications: List[Dict[str, Any]] = []
        self._current_category = "General"
        self._current_title = ""
        self._current_text: List[str] = []
        self._in_anchor = False
        self._current_href = ""
        self._current_anchor_text: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        tag_lower = tag.lower()
        attr_dict = {k.lower(): (v or "") for k, v in attrs}

        if tag_lower in ("h1", "h2", "h3", "h4", "span"):
            if "vc_tta-title-text" in attr_dict.get("class", ""):
                self._current_text = []

        elif tag_lower == "a":
            self._in_anchor = True
            self._current_href = attr_dict.get("href", "")
            self._current_anchor_text = []

        elif tag_lower in ("p", "strong", "em"):
            self._current_text = []

    def handle_endtag(self, tag: str):
        tag_lower = tag.lower()

        if tag_lower in ("h1", "h2", "h3", "h4", "span"):
            text = " ".join("".join(self._current_text).split()).strip()
            if text in (
                "Agstat Booklet",
                "Cost of Cultivation of Agricultural Crops",
                "Crop Enterprise Budget",
                "Vegetable Varietal Distribution in Sri Lanka",
                "Crop Forecast Monthly Report",
            ):
                self._current_category = text

        elif tag_lower in ("p", "strong", "em"):
            text = " ".join("".join(self._current_text).split()).strip()
            if text and ("Agstat" in text or "Yala" in text or "Maha" in text or "Cost of" in text or "Crop" in text):
                self._current_title = text

        elif tag_lower == "a":
            self._in_anchor = False
            anchor_label = " ".join("".join(self._current_anchor_text).split()).strip()
            href = self._current_href

            if href and ("drive.google.com" in href or ".pdf" in href.lower()):
                lang = "English"
                if anchor_label in ("සි", "Sinhala", "SI"):
                    lang = "Sinhala"
                elif anchor_label in ("த", "Tamil", "TA"):
                    lang = "Tamil"

                self.publications.append({
                    "category": self._current_category,
                    "title": self._current_title or self._current_category,
                    "download_url": href,
                    "language": lang,
                })

    def handle_data(self, data: str):
        if self._in_anchor:
            self._current_anchor_text.append(data)
        self._current_text.append(data)


class DOAExtractor:
    """Extractor for DOA SEPC publication catalog and structured cultivation statistics."""

    def __init__(self, base_url: str = "https://doa.gov.lk/sepc-downloads_en/"):
        self.base_url = base_url

    def extract_catalog_from_html(
        self,
        html_content: str,
        source_url: str = "https://doa.gov.lk/sepc-downloads_en/",
    ) -> List[DOAPublicationRecord]:
        """
        Parses DOA SEPC downloads HTML to produce catalog of available reports.
        """
        parser = _DOAPublicationHTMLParser()
        parser.feed(html_content)

        catalog: List[DOAPublicationRecord] = []
        for pub in parser.publications:
            title = pub["title"]
            year, season = self._extract_year_and_season(title)

            rec = DOAPublicationRecord(
                category=pub["category"],
                title=title,
                year=year,
                season=season,
                language=pub["language"],
                download_url=pub["download_url"],
                source_page_url=source_url,
            )
            catalog.append(rec)

        return catalog

    def extract_cultivation_from_csv(
        self,
        source: Union[str, Path],
    ) -> List[DOACultivationRecord]:
        """
        Ingests structured cultivation data from a CSV file or CSV formatted string.

        Expected columns (case-insensitive):
            crop, district, season, year, target_extent_ha, cultivated_extent_ha, expected_production_mt
        """
        if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and Path(source).is_file()):
            content = Path(source).read_text(encoding="utf-8", errors="replace")
            source_label = str(source)
        else:
            content = str(source)
            source_label = "in_memory_csv"

        # Skip comment lines (e.g. # [SYNTHETIC TEST FIXTURE])
        lines = [line for line in content.splitlines() if not line.strip().startswith("#")]
        if not lines:
            return []

        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        records: List[DOACultivationRecord] = []

        for row in reader:
            normalized_row = {k.strip().lower(): (v.strip() if v else "") for k, v in row.items() if k}

            crop = normalized_row.get("crop", "Unknown")
            district = normalized_row.get("district", "Unknown")
            season = normalized_row.get("season", "Unknown")

            try:
                year = int(normalized_row.get("year", "0"))
            except ValueError:
                year = 0

            target_ha = self._parse_optional_float(normalized_row.get("target_extent_ha"))
            cultivated_ha = self._parse_optional_float(normalized_row.get("cultivated_extent_ha"))
            production_mt = self._parse_optional_float(normalized_row.get("expected_production_mt"))
            prov_source = normalized_row.get("provenance_source") or source_label

            record = DOACultivationRecord(
                crop=crop,
                district=district,
                season=season,
                year=year,
                target_extent_ha=target_ha,
                cultivated_extent_ha=cultivated_ha,
                expected_production_mt=production_mt,
                raw_source=prov_source,
            )
            records.append(record)

        return records

    def extract_pdf_publication(self, pub_record: DOAPublicationRecord) -> None:
        """
        Documented handler for PDF booklets hosted on Google Drive.
        Explicitly raises UnsupportedSourceFormatError as direct PDF parsing requires
        local file download and specialized PDF OCR / layout parsers.
        """
        raise UnsupportedSourceFormatError(
            f"DOA publication '{pub_record.title}' is distributed as a compiled multi-page PDF "
            f"hosted on Google Drive ({pub_record.download_url}). Direct streaming extraction is not "
            f"supported without downloading the PDF locally and running a layout/table extraction tool "
            f"(such as pdfplumber or tabula-py)."
        )

    def _extract_year_and_season(self, text: str) -> Tuple[Optional[int], Optional[str]]:
        """Extracts year and agricultural season from publication title text."""
        season = None
        if re.search(r"\bYala\b", text, re.IGNORECASE):
            season = "Yala"
        elif re.search(r"\bMaha\b", text, re.IGNORECASE):
            season = "Maha"

        year_match = re.search(r"\b(20\d{2})\b", text)
        year = int(year_match.group(1)) if year_match else None

        return year, season

    def _parse_optional_float(self, value: Optional[str]) -> Optional[float]:
        """Safely parses float string or returns None."""
        if not value:
            return None
        cleaned = value.replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
