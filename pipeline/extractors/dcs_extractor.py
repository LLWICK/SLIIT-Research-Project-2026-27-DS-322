"""
Department of Census and Statistics (DCS) Agricultural Statistics Extractor.
SLIIT Research Project DS-322 (2026/27) - Member 1

Extracts agricultural yield, production, and extent statistics from official DCS HTML sources.
Official source base: https://www.statistics.gov.lk/Agriculture/StaticalInformation/
"""

import datetime
import html
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union


class DCSExtractionRecord:
    """Represents a single raw extracted data point from a DCS agricultural table."""

    def __init__(
        self,
        crop: str,
        metric: str,
        raw_metric: str,
        raw_unit: str,
        district: str,
        period_label: str,
        raw_value: str,
        table_id: str = "",
        table_title: str = "",
        source_url: str = "",
        extraction_timestamp: Optional[str] = None,
        footnote: Optional[str] = None,
    ):
        self.crop = crop
        self.metric = metric
        self.raw_metric = raw_metric
        self.raw_unit = raw_unit
        self.district = district
        self.period_label = period_label
        self.raw_value = raw_value
        self.table_id = table_id
        self.table_title = table_title
        self.source_url = source_url
        self.extraction_timestamp = extraction_timestamp or datetime.datetime.utcnow().isoformat()
        self.footnote = footnote

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": "DCS",
            "crop": self.crop,
            "metric": self.metric,
            "raw_metric": self.raw_metric,
            "raw_unit": self.raw_unit,
            "district": self.district,
            "period_label": self.period_label,
            "raw_value": self.raw_value,
            "table_id": self.table_id,
            "table_title": self.table_title,
            "source_url": self.source_url,
            "extraction_timestamp": self.extraction_timestamp,
            "footnote": self.footnote,
        }

    def __repr__(self) -> str:
        return (
            f"<DCSExtractionRecord crop={self.crop} metric={self.metric} "
            f"district={self.district} period={self.period_label} raw_value={self.raw_value}>"
        )


class _HTMLTableDataExtractor(HTMLParser):
    """Internal HTML parser to extract structured tables and associated metadata from DCS HTML."""

    def __init__(self):
        super().__init__()
        self.tables_data: List[Dict[str, Any]] = []
        self._rows_stack: List[List[List[str]]] = []
        self._current_row: List[str] = []
        self._current_cell: List[str] = []
        self._in_cell = False
        self._text_buffer: List[str] = []
        self._last_headings: List[str] = []
        self._last_unit: Optional[str] = None
        self._last_footnote: Optional[str] = None
        self._table_metadata_stack: List[Dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        tag_lower = tag.lower()
        attr_dict = {k.lower(): (v or "") for k, v in attrs}

        if tag_lower in ("h1", "h2", "h3", "h4", "h5", "b", "strong", "span", "p"):
            self._text_buffer = []

        if tag_lower == "table":
            meta = {
                "headings": list(self._last_headings),
                "unit": self._last_unit,
                "footnote": self._last_footnote,
                "class": attr_dict.get("class", ""),
            }
            self._table_metadata_stack.append(meta)
            self._rows_stack.append([])

        elif tag_lower == "tr":
            self._current_row = []

        elif tag_lower in ("td", "th"):
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag: str):
        tag_lower = tag.lower()

        if tag_lower in ("h1", "h2", "h3", "h4", "h5", "b", "strong", "span", "p"):
            text = " ".join("".join(self._text_buffer).split()).strip()
            if text:
                # Check for unit pattern
                unit_match = re.search(r"Unit\s*:\s*([^.<]+)|Metric Tons|Hectares|Kg/Ha", text, re.IGNORECASE)
                if unit_match:
                    self._last_unit = text
                # Check for footnote pattern
                if "*" in text and ("estimate" in text.lower() or "provisional" in text.lower()):
                    self._last_footnote = text
                # Check for table title/heading pattern (e.g. CO3 - 1, Table F07 - 1, etc.)
                if re.search(r"(CO\d+|Table\s+[A-Za-z0-9-]+|Production|Extent|Yield)", text, re.IGNORECASE):
                    if len(text) > 3 and not unit_match:
                        self._last_headings.append(text)
                        if len(self._last_headings) > 4:
                            self._last_headings.pop(0)

        elif tag_lower in ("td", "th"):
            self._in_cell = False
            cell_text = " ".join("".join(self._current_cell).split()).strip()
            # Unescape HTML entities e.g. &nbsp;
            cell_text = html.unescape(cell_text)
            self._current_row.append(cell_text)

        elif tag_lower == "tr":
            # Only add non-empty rows to the active table at the top of the stack
            if self._rows_stack and any(cell for cell in self._current_row):
                self._rows_stack[-1].append(self._current_row)

        elif tag_lower == "table":
            meta = self._table_metadata_stack.pop() if self._table_metadata_stack else {}
            table_rows = self._rows_stack.pop() if self._rows_stack else []

            # Consider tables that contain data rows
            if len(table_rows) >= 2:
                # Verify if this table contains data cells (not just layout/navigation)
                has_numeric = any(
                    any(re.search(r"\d", c) for c in row[1:])
                    for row in table_rows[1:]
                )
                if has_numeric:
                    self.tables_data.append({
                        "rows": [list(r) for r in table_rows],
                        "metadata": meta,
                    })

    def handle_data(self, data: str):
        if self._in_cell:
            self._current_cell.append(data)
        self._text_buffer.append(data)


class DCSExtractor:
    """Extractor for agricultural data from the Department of Census and Statistics (DCS)."""

    def __init__(self, base_url: str = "https://www.statistics.gov.lk/Agriculture/StaticalInformation/"):
        self.base_url = base_url

    def extract_from_html(
        self,
        html_content: str,
        source_url: str = "",
        crop_hint: str = "",
    ) -> List[DCSExtractionRecord]:
        """
        Parses HTML string and extracts agricultural production/yield/extent records.

        Args:
            html_content: Raw HTML text of the DCS page.
            source_url: Optional source URL for provenance tracking.
            crop_hint: Optional fallback crop name if not detectable from table title.

        Returns:
            List of DCSExtractionRecord instances.
        """
        parser = _HTMLTableDataExtractor()
        parser.feed(html_content)

        extracted_records: List[DCSExtractionRecord] = []

        for table_info in parser.tables_data:
            rows = table_info["rows"]
            meta = table_info["metadata"]

            if not rows or len(rows) < 2:
                continue

            # Identify headers and titles
            headings = meta.get("headings", [])
            title = headings[-1] if headings else ""
            raw_unit = meta.get("unit") or self._detect_unit_from_text(title)
            footnote = meta.get("footnote")

            table_id, clean_title, detected_crop, detected_metric = self._parse_table_title(title, crop_hint)

            # Analyze header row
            header_row = rows[0]
            # Identify columns that represent time periods
            # Format 1: District | 2004 | 2005 | ... | 2023
            # Format 2: Districts | 2007 | 2008 | ... | 2018 upto May
            period_cols: List[Tuple[int, str]] = []
            start_col = 1

            for col_idx in range(1, len(header_row)):
                col_name = header_row[col_idx].strip()
                if col_name and not col_name.lower().startswith("district"):
                    period_cols.append((col_idx, col_name))

            if not period_cols and len(rows) > 2:
                # Maybe header is in second row
                header_row = rows[1]
                for col_idx in range(1, len(header_row)):
                    col_name = header_row[col_idx].strip()
                    if col_name and not col_name.lower().startswith("district"):
                        period_cols.append((col_idx, col_name))
                data_rows = rows[2:]
            else:
                data_rows = rows[1:]

            for row in data_rows:
                if not row or not any(row):
                    continue

                district_cell = row[0].strip() if len(row) > 0 else ""

                # Ignore repeating sub-headers, footnotes, and layout anchors
                if (
                    not district_cell
                    or district_cell.startswith("*")
                    or "estimated" in district_cell.lower()
                    or district_cell.lower() in ("district", "districts", "year", "top", "unit")
                ):
                    continue

                for col_idx, period_label in period_cols:
                    if col_idx < len(row):
                        raw_val = row[col_idx].strip()
                    else:
                        raw_val = ""

                    rec = DCSExtractionRecord(
                        crop=detected_crop,
                        metric=detected_metric,
                        raw_metric=clean_title or title or detected_metric,
                        raw_unit=raw_unit or "",
                        district=district_cell,
                        period_label=period_label,
                        raw_value=raw_val,
                        table_id=table_id,
                        table_title=clean_title or title,
                        source_url=source_url,
                        footnote=footnote,
                    )
                    extracted_records.append(rec)

        return extracted_records

    def extract_from_file(
        self,
        file_path: Union[str, Path],
        source_url: str = "",
        crop_hint: str = "",
    ) -> List[DCSExtractionRecord]:
        """Extracts records from a local DCS HTML file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"DCS source file not found: {path}")

        content = path.read_text(encoding="utf-8", errors="replace")
        effective_url = source_url or f"file://{path.resolve().as_posix()}"
        return self.extract_from_html(content, source_url=effective_url, crop_hint=crop_hint)

    def _detect_unit_from_text(self, text: str) -> str:
        """Detects unit string from surrounding text or table title."""
        if not text:
            return ""
        if re.search(r"Metric\s*Tons|Unit\s*:\s*MT", text, re.IGNORECASE):
            return "Metric Tons"
        if re.search(r"Hectares|Unit\s*:\s*Ha", text, re.IGNORECASE):
            return "Hectares"
        if re.search(r"Kg/Ha|kg\s*per\s*ha", text, re.IGNORECASE):
            return "Kg/Ha"
        if re.search(r"Farmers|Number", text, re.IGNORECASE):
            return "Count"
        return ""

    def _parse_table_title(self, title: str, crop_hint: str) -> Tuple[str, str, str, str]:
        """
        Parses table title to extract table_id, clean_title, crop, and metric.
        Examples:
            'CO3 – 1: Potato Production' -> ('CO3 - 1', 'Potato Production', 'Potato', 'production')
            'Table F07 - 1 Big Onion Production 2004-2023' -> ('Table F07 - 1', 'Big Onion Production', 'Big Onion', 'production')
            'CO3 – 2: Potato Extent' -> ('CO3 - 2', 'Potato Extent', 'Potato', 'extent')
        """
        table_id = ""
        clean_title = title.strip()
        crop = crop_hint or "Unknown"
        metric = "unknown"

        # Extract table ID code
        code_match = re.search(r"^([A-Z0-9]+\s*[\u2013-]\s*\d+|Table\s+[A-Za-z0-9-]+)\s*[:\s-]*(.*)", title, re.IGNORECASE)
        if code_match:
            table_id = code_match.group(1).replace("\u2013", "-").strip()
            clean_title = code_match.group(2).strip()

        # Identify metric
        title_lower = title.lower()
        if "production" in title_lower:
            metric = "production"
        elif "extent" in title_lower:
            metric = "extent"
        elif "yield" in title_lower:
            metric = "yield"
        elif "farmer" in title_lower:
            metric = "farmers"
        elif "import" in title_lower:
            metric = "imports"

        # Identify crop
        crops = [
            "Potato", "Big Onion", "Red Onion", "Chilli", "Green Chilli",
            "Cassava", "Tomato", "Carrot", "Cabbage", "Beans", "Pumpkin",
            "Brinjal", "Sugarcane"
        ]
        for c in crops:
            if re.search(rf"\b{re.escape(c)}\b", title, re.IGNORECASE):
                crop = c
                break

        return table_id, clean_title, crop, metric
