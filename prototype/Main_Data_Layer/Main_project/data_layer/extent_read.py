"""Read Carrot, Leeks, and Tomato extent from the downloaded DOA PDFs.

The district tables are images. A crop is kept only when the area rows
add up to the Sri Lanka row and to each printed subtotal. A dash is missing.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import re
from pathlib import Path

import numpy as np
import pdfplumber
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from data_layer.extent import REGIONS
from data_layer.paths import PACKAGE_DIR

PDF_DIR = PACKAGE_DIR / "raw" / "doa_crop_forecast"
SNAPSHOT_CSV = PACKAGE_DIR / "extent_snapshots.csv"
CROP_WORDS = {"carrot": "carrot", "leeks": "leek", "tomato": "tomato"}
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
NAME_KEY = {
    re.sub(r"[^a-z0-9]", "", name.lower()): name
    for name, _, _ in REGIONS
}
DASHES = {"-", "–", "—", "−", "一", "_", "—"}
CSV_FIELDS = (
    "source",
    "report_date",
    "season_label",
    "as_of_date",
    "crop_code",
    "region_name",
    "region_level",
    "region_group",
    "target_ha",
    "achievement_ha",
    "target_missing",
    "achievement_missing",
    "page_number",
    "source_file",
)


def _line_positions(img: Image.Image, horizontal: bool, dark_value: int, min_ratio: float, min_gap: int) -> list[int]:
    gray = img.convert("L")
    width, height = gray.size
    px = gray.load()
    found = []
    previous = -100
    if horizontal:
        span = range(height)
        # Step 1. Dotted rules in these scans fall on one pixel parity, so a
        # step of 2 misses the whole line.
        sample = range(10, width - 10)
    else:
        span = range(width)
        sample = range(120, height - 30, 2)
    for pos in span:
        dark = samples = 0
        for other in sample:
            samples += 1
            value = px[other, pos] if horizontal else px[pos, other]
            if value < dark_value:
                dark += 1
        if samples and dark / samples >= min_ratio and pos - previous > min_gap:
            found.append(pos)
            previous = pos
    return found


def _data_row_count(lines: list[int]) -> int:
    return len(_data_rows(lines))


def _lines(img: Image.Image, horizontal: bool) -> list[int]:
    strict = _line_positions(img, horizontal, dark_value=80, min_ratio=0.45, min_gap=3)
    if not horizontal or _data_row_count(strict) >= 35:
        return strict
    # Some reports print dotted rules. Those stay lighter than a solid line
    # but are still darker than the page across the full width.
    dotted = _line_positions(img, horizontal, dark_value=160, min_ratio=0.40, min_gap=8)
    if _data_row_count(dotted) > _data_row_count(strict):
        return dotted
    return strict


def _data_rows(horizontal: list[int]) -> list[tuple[int, int]]:
    gaps = [(top, bottom, bottom - top) for top, bottom in zip(horizontal, horizontal[1:])]
    best: list[tuple[int, int, int]] = []
    best_at = 0
    current: list[tuple[int, int, int]] = []
    current_at = 0
    for index, (top, bottom, gap) in enumerate(gaps):
        if gap < 14:
            if len(current) > len(best):
                best, best_at = current, current_at
            current = []
            continue
        if not current:
            current = [(top, bottom, gap)]
            current_at = index
            continue
        typical = sorted(item[2] for item in current)[len(current) // 2]
        if typical * 0.65 <= gap <= typical * 1.55:
            current.append((top, bottom, gap))
        else:
            if len(current) > len(best):
                best, best_at = current, current_at
            current = [(top, bottom, gap)]
            current_at = index
    if len(current) > len(best):
        best, best_at = current, current_at
    if not best:
        return []
    typical = sorted(item[2] for item in best)[len(best) // 2]
    end = best_at + len(best)
    extended = False
    while end < len(gaps):
        gap = gaps[end][2]
        normal = typical * 0.65 <= gap <= typical * 1.55
        tall = typical * 1.55 < gap <= typical * 2.6
        if tall or (extended and normal):
            best.append(gaps[end])
            end += 1
            extended = True
            continue
        break
    return [(top, bottom) for top, bottom, _gap in best]


def _ocr(engine: RapidOCR, image: Image.Image) -> list[tuple[float, float, str, float]]:
    scaled = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
    result, _ = engine(np.array(scaled))
    items = []
    if not result:
        return items
    for box, text, score in result:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        items.append((min(xs) / 2, (min(ys) + max(ys)) / 4, text.strip(), float(score), max(xs) / 2))
    return items


def _column_bounds(vertical: list[int], label_x: float) -> tuple[int, int] | None:
    for left, right in zip(vertical, vertical[1:]):
        if left <= label_x <= right:
            return left + 2, right - 2
    return None


def _crop_columns(header: list[tuple[float, float, str, float]], vertical: list[int]) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    targets = []
    achievements = []
    for item in header:
        text = item[2].lower()
        left = item[0]
        right = item[4] if len(item) > 4 else left
        if "target" in text and "achiev" in text:
            targets.append(left)
            achievements.append((left + right) / 2)
        elif text.startswith("target"):
            targets.append(left)
        elif "achiev" in text:
            achievements.append(left)
    found = {}
    for crop, word in CROP_WORDS.items():
        names = [item[0] for item in header if word in item[2].lower()]
        if not names:
            continue
        name_x = names[0]
        earlier_targets = [x for x in targets if x <= name_x + 40]
        if not earlier_targets:
            continue
        target_x = max(earlier_targets)
        later_achievements = [x for x in achievements if x > target_x]
        if not later_achievements:
            continue
        achievement_x = min(later_achievements)
        target_box = _column_bounds(vertical, target_x)
        achievement_box = _column_bounds(vertical, achievement_x)
        if target_box and achievement_box and target_box != achievement_box:
            found[crop] = (target_box, achievement_box)
    return found


def _parse_number(text: str) -> float | None | str:
    cleaned = text.strip().replace(",", "").replace(" ", "")
    if cleaned in DASHES or cleaned == "":
        return None
    cleaned = cleaned.replace("O", "0")
    try:
        return float(cleaned)
    except ValueError:
        return "bad"


def _row_index(rows: list[tuple[int, int]], y: float, top: int) -> int | None:
    absolute = top + y
    best_index = None
    best_distance = 10**9
    for index, (row_top, row_bottom) in enumerate(rows):
        distance = abs(absolute - (row_top + row_bottom) / 2)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    row_height = rows[0][1] - rows[0][0]
    if best_distance > max(22, row_height * 0.45):
        return None
    return best_index


def _read_column(engine: RapidOCR, img: Image.Image, box: tuple[int, int], rows: list[tuple[int, int]]) -> list[tuple[float | None, float]]:
    values: list[tuple[float | None, float]] = [(None, 0.0) for _ in rows]
    left, right = box
    if right - left < 8:
        return values
    top = rows[0][0]
    bottom = rows[-1][1]
    items = _ocr(engine, img.crop((left, top, right, bottom)))
    grouped: dict[int, list[tuple[float, str, float]]] = {}
    for x, y, text, score, *_rest in items:
        index = _row_index(rows, y, top)
        if index is None:
            continue
        grouped.setdefault(index, []).append((x, text, score))
    for index, parts in grouped.items():
        parts.sort()
        parsed = _parse_number("".join(text for _x, text, _score in parts))
        if parsed == "bad":
            continue
        values[index] = (parsed, min(score for _x, _text, score in parts))
    return values


def _reread_cell(engine: RapidOCR, img: Image.Image, box: tuple[int, int], row: tuple[int, int]) -> tuple[float | None, float] | None:
    left, right = box
    top, bottom = row
    cell = img.crop((left, top + 1, right, bottom - 1))
    if cell.width < 4 or cell.height < 4:
        return None
    gray = cell.convert("L")
    px = gray.load()
    dark = [(x, y) for y in range(gray.height) for x in range(gray.width) if px[x, y] < 110]
    if len(dark) < 8:
        return None
    ys = [point[1] for point in dark]
    if max(ys) - min(ys) <= 4:
        return (None, 1.0)
    big = cell.resize((cell.width * 6, cell.height * 6), Image.Resampling.NEAREST)
    result, _ = engine(np.array(big))
    if not result:
        return None
    text, score = max(result, key=lambda item: item[2])[1:]
    parsed = _parse_number(text)
    if parsed == "bad":
        return None
    return parsed, float(score)


def _repair(engine: RapidOCR, img: Image.Image, box: tuple[int, int], rows: list[tuple[int, int]], values: list[tuple[float | None, float]]) -> list[float | None]:
    repaired = []
    for row, (value, score) in zip(rows, values):
        if value is None or score >= 0.99:
            repaired.append(value)
            continue
        reread = _reread_cell(engine, img, box, row)
        repaired.append(value if reread is None else reread[0])
    return repaired


def _canonical(text: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", text.lower())
    return NAME_KEY.get(key, " ".join(text.split()))


def _regions(names: list[str]) -> list[tuple[str, str, str]] | None:
    classified = []
    group = "provincial"
    seen = set()
    for name in names:
        compact = re.sub(r"[^a-z]", "", name.lower())
        if "srilanka" in compact:
            level, group_name = "national", "national"
            name = "Sri Lanka"
        elif "interprovincial" in compact:
            level, group_name, group = "subtotal", "inter_provincial", "inter_provincial"
            name = "Inter Provincial DDA Area"
        elif "provincialdda" in compact:
            level, group_name, group = "subtotal", "provincial", "provincial"
            name = "Provincial DDA Areas"
        elif compact.startswith("mahaweliarea"):
            level, group_name, group = "subtotal", "mahaweli", "mahaweli"
            name = "Mahaweli Areas"
        else:
            level, group_name = "area", group
            name = _canonical(name)
        if name in seen:
            return None
        seen.add(name)
        classified.append((name, level, group_name))
    if not any(level == "national" for _, level, _ in classified):
        return None
    return classified


def _accept(regions, target, achievement) -> str | None:
    for column, values in (("target", target), ("achievement", achievement)):
        national = [value for (_n, level, _g), value in zip(regions, values) if level == "national"]
        if len(national) != 1 or national[0] is None:
            return f"{column} has no Sri Lanka figure"
        areas = [value for (_n, level, _g), value in zip(regions, values) if level == "area" and value is not None]
        if abs(sum(areas) - national[0]) > 0.05:
            return f"{column} areas {sum(areas)} != Sri Lanka {national[0]}"
        for group in ("provincial", "inter_provincial", "mahaweli"):
            subtotal = [
                value
                for (_n, level, group_name), value in zip(regions, values)
                if level == "subtotal" and group_name == group
            ]
            leaves = [
                value
                for (_n, level, group_name), value in zip(regions, values)
                if level == "area" and group_name == group and value is not None
            ]
            # A blank group heading is a label, not a total. Check the group
            # only when that row actually prints a number.
            if not subtotal or subtotal[0] is None:
                continue
            if abs(sum(leaves) - subtotal[0]) > 0.05:
                return f"{column} {group} {sum(leaves)} != subtotal {subtotal[0]}"
    return None


def _month_end(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"


def _metadata(text: str, source_file: str) -> dict | None:
    dates = re.findall(r"(\d{1,2})\s*(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(20\d{2})", text, re.I)
    report = None
    for day, month_name, year in dates:
        month = MONTHS.get(month_name.lower())
        if month and 1 <= int(day) <= 31:
            report = f"{int(year):04d}-{month:02d}-{int(day):02d}"
            break
    season = None
    maha = re.search(r"Maha\s+(20\d{2})\s*/\s*(\d{2})", text, re.I) or re.search(
        r"(20\d{2})\s*/\s*(\d{2})\s*Maha", text, re.I
    )
    yala = re.search(r"Yala\s+(20\d{2})", text, re.I) or re.search(r"(20\d{2})\s*Yala", text, re.I)
    if maha:
        season = f"{maha.group(1)}/{maha.group(2)} Maha"
        start_year = int(maha.group(1))
        end_year = int(maha.group(1)[:2] + maha.group(2))
    elif yala:
        season = f"{yala.group(1)} Yala"
        start_year = end_year = int(yala.group(1))
    else:
        return None
    as_of_month = None
    for match in re.finditer(r"end(?:\s+of)?\s+([A-Za-z]+)", text, re.I):
        month = MONTHS.get(match.group(1).lower())
        if month:
            as_of_month = month
            break
    if as_of_month is None or report is None:
        return None
    month = as_of_month
    year = end_year if month <= 3 else start_year
    return {
        "report_date": report,
        "season_label": season,
        "as_of_date": _month_end(year, month),
        "source_file": source_file,
    }


def _page_image(page) -> Image.Image | None:
    if not page.images:
        return None
    image = max(page.images, key=lambda item: item.get("srcsize", (0, 0))[0] * item.get("srcsize", (0, 0))[1])
    width, height = image.get("srcsize", (0, 0))
    if width * height > 2000 * 1600 or max(width, height) < 1200 or min(width, height) < 600:
        return None
    data = image["stream"].get_data()
    if len(data) == width * height * 3:
        image_out = Image.frombytes("RGB", (width, height), data)
    elif len(data) == width * height:
        image_out = Image.frombytes("L", (width, height), data).convert("RGB")
    else:
        return None
    if image_out.height > image_out.width:
        image_out = image_out.rotate(90, expand=True)
    return image_out


def _district_names(items: list[tuple[float, float, str, float]], rows: list[tuple[int, int]]) -> list[str]:
    grouped: dict[int, list[tuple[str, float]]] = {}
    for _x, y, text, score, *_rest in items:
        index = _row_index(rows, y, rows[0][0])
        if index is None or score < 0.5 or not text:
            continue
        grouped.setdefault(index, []).append((text, score))
    names = []
    for index in range(len(rows)):
        choices = grouped.get(index, [])
        known = [
            text
            for text, _score in choices
            if re.sub(r"[^a-z0-9]", "", text.lower()) in NAME_KEY
        ]
        pool = known or [text for text, _score in choices]
        names.append(max(pool, key=len) if pool else "")
    return names


def _district_right(vertical: list[int], header: list[tuple[float, float, str, float]], columns: dict) -> int:
    labels = [item[0] for item in header if "district" in item[2].lower()]
    if labels:
        rights = [line for line in vertical if line > labels[0] + 30]
        if rights:
            return rights[0]
    for left, right in zip(vertical, vertical[1:]):
        if right - left > 140:
            return right
    return min(box[0][0] for box in columns.values())


def _table_from_image(engine: RapidOCR, img: Image.Image) -> dict[str, tuple[list, list[float | None], list[float | None]]] | None:
    horizontal = _lines(img, horizontal=True)
    vertical = _lines(img, horizontal=False)
    rows = _data_rows(horizontal)
    if len(rows) < 10 or len(vertical) < 8:
        return None
    header_bottom = rows[0][0]
    header = _ocr(engine, img.crop((0, 0, img.width, header_bottom)))
    columns = _crop_columns(header, vertical)
    if not columns:
        return None
    district_right = _district_right(vertical, header, columns)
    district_items = _ocr(engine, img.crop((2, rows[0][0], district_right, rows[-1][1])))
    names = _district_names(district_items, rows)
    if any(not name for name in names):
        return None
    regions = _regions(names)
    if regions is None:
        return None
    tables = {}
    for crop, (target_box, achievement_box) in columns.items():
        target = _read_column(engine, img, target_box, rows)
        achievement = _read_column(engine, img, achievement_box, rows)
        target_values = [value for value, _score in target]
        achievement_values = [value for value, _score in achievement]
        reason = _accept(regions, target_values, achievement_values)
        if reason:
            target_values = _repair(engine, img, target_box, rows, target)
            achievement_values = _repair(engine, img, achievement_box, rows, achievement)
            reason = _accept(regions, target_values, achievement_values)
        tables[crop] = (regions, target_values, achievement_values, reason)
    return tables


def _mentions_crop(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ("carrot", "leek", "tomato", "torrac", "skeel", "otamot"))


def _rendered_page(page) -> Image.Image | None:
    # Some reports draw the district table as rotated text instead of an image.
    # Rasterizing shows the table, and a 270 degree turn makes those rows horizontal.
    image = page.to_image(resolution=140).original.convert("RGB")
    text = (page.extract_text() or "").lower()
    if any(word in text for word in ("obmoloc", "torrac", "ayilearawun", "skeel")):
        image = image.rotate(270, expand=True)
    if min(image.size) < 700:
        return None
    return image


def read_pdf(engine: RapidOCR, path: Path) -> tuple[list[dict], list[str]]:
    pdf = pdfplumber.open(path)
    text_pages = [index for index in (0, 1, 4, 5, 6, 7) if index < len(pdf.pages)]
    text = "\n".join((pdf.pages[index].extract_text() or "") for index in text_pages)
    meta = _metadata(text, path.name)
    if meta is None:
        return [], [f"{path.name}: could not read the report date or season"]
    accepted = []
    notes = []
    for page_number, page in enumerate(pdf.pages, start=1):
        image = _page_image(page)
        tables = _table_from_image(engine, image) if image is not None else None
        if not tables and _mentions_crop(page.extract_text() or ""):
            rendered = _rendered_page(page)
            if rendered is not None:
                tables = _table_from_image(engine, rendered)
        if not tables:
            continue
        for crop, (regions, target, achievement, reason) in tables.items():
            if reason:
                notes.append(f"{path.name} p{page_number} {crop}: {reason}")
                continue
            meta_row = {**meta, "page_number": page_number}
            for (name, level, group), target_ha, achievement_ha in zip(regions, target, achievement):
                accepted.append(
                    {
                        "source": "doa",
                        "report_date": meta_row["report_date"],
                        "season_label": meta_row["season_label"],
                        "as_of_date": meta_row["as_of_date"],
                        "crop_code": crop,
                        "region_name": name,
                        "region_level": level,
                        "region_group": group,
                        "target_ha": "" if target_ha is None else target_ha,
                        "achievement_ha": "" if achievement_ha is None else achievement_ha,
                        "target_missing": target_ha is None,
                        "achievement_missing": achievement_ha is None,
                        "page_number": page_number,
                        "source_file": path.name,
                    }
                )
            national_target = next(value for (_name, level, _group), value in zip(regions, target) if level == "national")
            national_achievement = next(value for (_name, level, _group), value in zip(regions, achievement) if level == "national")
            notes.append(
                f"{path.name} p{page_number} {crop}: accepted "
                f"target {national_target} achievement {national_achievement}"
            )
    if not any(" accepted " in note for note in notes):
        notes.append(f"{path.name}: no crop passed the sum check")
    return accepted, notes


def write_snapshots(rows: list[dict]) -> None:
    with SNAPSHOT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="Read one PDF filename.")
    args = parser.parse_args()
    paths = sorted(PDF_DIR.glob("*.pdf"))
    if args.only:
        paths = [path for path in paths if args.only in path.name]
    engine = RapidOCR()
    rows = []
    for path in paths:
        print(path.name, flush=True)
        accepted, notes = read_pdf(engine, path)
        rows.extend(accepted)
        for note in notes:
            print(" ", note, flush=True)
    if not args.only:
        write_snapshots(rows)
        crops = {(row["source_file"], row["crop_code"]) for row in rows}
        print(f"{len(crops)} accepted crop snapshots, {len(rows)} rows")
        print(SNAPSHOT_CSV)


if __name__ == "__main__":
    main()
