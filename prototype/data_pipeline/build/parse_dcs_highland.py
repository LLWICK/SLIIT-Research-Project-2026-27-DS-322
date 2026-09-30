"""Parse DCS highland-crop HTML tables into a tidy seasonal supply file.

Usage (from the prototype folder):
  python -m data_pipeline.build.parse_dcs_highland
"""
from __future__ import annotations

import argparse
import re
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from data_pipeline.config import RAW_DIR, processed_dir

FILES = {
    "carrot": "HighlandCrops_C031_Carrot.html",
    "leeks": "HighlandCrops_C036_Leeks.html",
    "tomato": "HighlandCrops_C028_Tomatoes.html",
}

DISTRICT_FIX = {
    "kaluthara": "Kalutara",
    "mullativu": "Mullaitivu",
    "mullaitivu": "Mullaitivu",
    "hanbantota": "Hambantota",
    "hambantota": "Hambantota",
    "nuwara eliya": "Nuwara Eliya",
    "moneragala": "Monaragala",
    "monaragala": "Monaragala",
}


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._cur: list[str] = []
        self._in_td = False
        self._buf: list[str] = []

    def _flush_row(self) -> None:
        if self._cur:
            self.rows.append(self._cur)
        self._cur = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._flush_row()
        elif tag == "td":
            self._in_td = True
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._in_td:
            self._cur.append("".join(self._buf).replace("\xa0", " ").strip())
            self._in_td = False
        elif tag == "tr":
            self._flush_row()

    def close(self) -> None:
        self._flush_row()
        super().close()

    def handle_data(self, data: str) -> None:
        if self._in_td:
            self._buf.append(data)


def _canonical_district(name: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", name).strip()
    if not cleaned or cleaned.lower().startswith("national"):
        return None
    return DISTRICT_FIX.get(cleaned.lower(), cleaned)


def _to_float(text: str) -> float | None:
    raw = text.replace(",", "").replace("\xa0", "").strip()
    if raw in {"", "-", "—", "na", "NA", "*"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_html(path: Path, crop: str) -> pd.DataFrame:
    parser = _TableParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    parser.close()
    header = next((row for row in parser.rows if row and row[0].strip().lower() == "district"), None)
    if header is None:
        raise ValueError(f"No district header in {path}")
    years = [int(cell) for cell in header[1:] if re.fullmatch(r"\d{4}", cell.strip())]
    if not years:
        raise ValueError(f"No year columns in {path}")

    records: list[dict] = []
    for row in parser.rows:
        if not row or row[0].strip().lower() in {"district", ""}:
            continue
        if row[0].strip().lower() in {"yala", "extent", "production"}:
            continue
        district = _canonical_district(row[0])
        if district is None:
            continue
        values = row[1:]
        needed = len(years) * 6
        if len(values) < needed:
            values = values + [""] * (needed - len(values))
        for index, year in enumerate(years):
            block = values[index * 6 : index * 6 + 6]
            for season, extent, production in (
                ("Yala", block[0], block[3]),
                ("Maha", block[1], block[4]),
            ):
                records.append(
                    {
                        "crop": crop,
                        "district": district,
                        "year": year,
                        "season": season,
                        "extent_ha": _to_float(extent),
                        "production_mt": _to_float(production),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(f"No district rows parsed from {path}")
    return frame


def build(raw_dir: Path | None = None) -> pd.DataFrame:
    folder = (raw_dir or RAW_DIR) / "dcs" / "highland_crops_timeseries"
    frames = []
    for crop, name in FILES.items():
        path = folder / name
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(parse_html(path, crop))
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["crop", "district", "year", "season"], keep="last")
    return out.sort_values(["crop", "district", "year", "season"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse DCS highland HTML to a seasonal CSV")
    parser.add_argument("--raw-dir", type=Path, default=None)
    args = parser.parse_args()
    frame = build(args.raw_dir)
    destination = processed_dir() / "dcs_highland_seasonal.csv"
    frame.to_csv(destination, index=False)
    print(f"wrote {destination} rows={len(frame)} crops={sorted(frame.crop.unique())}")


if __name__ == "__main__":
    main()
