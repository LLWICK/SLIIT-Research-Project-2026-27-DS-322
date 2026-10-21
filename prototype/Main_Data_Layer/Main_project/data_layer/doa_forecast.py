"""Download the English DOA crop-forecast PDFs.

Source page:
https://doa.gov.lk/sepc-downloads_en/

Only the "Crop Forecast Monthly Report" section is used. Those files are
Google Drive links. Sinhala and Tamil links point at the same file as English.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from data_layer.paths import PACKAGE_DIR

PAGE_URL = "https://doa.gov.lk/sepc-downloads_en/"
SECTION = "Crop Forecast Monthly Report"
RAW_DIR = PACKAGE_DIR / "raw" / "doa_crop_forecast"
MANIFEST = RAW_DIR / "manifest.csv"

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "cobweb-research/1.0 (crop-forecast download)"

MANIFEST_FIELDS = (
    "title",
    "slug",
    "drive_id",
    "source_url",
    "local_path",
    "bytes",
    "sha256",
    "status",
    "downloaded_at",
)


def _clean(text: str) -> str:
    return " ".join(text.split())


def _slug(title: str) -> str:
    text = title.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text[:80]


def _drive_id(url: str) -> str:
    match = re.search(r"/file/d/([A-Za-z0-9_-]+)", url)
    if not match:
        raise ValueError(f"Not a Google Drive file link: {url}")
    return match.group(1)


def list_reports() -> list[dict[str, str]]:
    response = SESSION.get(PAGE_URL, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    heading = None
    for tag in soup.find_all("h4"):
        if _clean(tag.get_text()) == SECTION:
            heading = tag
            break
    if heading is None:
        raise RuntimeError(f"Could not find the {SECTION!r} heading.")

    reports: list[dict[str, str]] = []
    title: str | None = None
    for el in heading.find_all_next():
        if el.name in ("h2", "h3", "h4"):
            break
        if el.name == "strong":
            text = _clean(el.get_text())
            if "Yala" in text or "Maha" in text:
                title = text
        elif el.name == "a" and title and el.get("href") and _clean(el.get_text()) == "E":
            url = el["href"]
            reports.append(
                {
                    "title": title,
                    "slug": _slug(title),
                    "drive_id": _drive_id(url),
                    "source_url": url,
                }
            )
            title = None

    if not reports:
        raise RuntimeError("The crop-forecast section had no English links.")
    ids = [row["drive_id"] for row in reports]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Two reports share a Google Drive file id.")
    return reports


def _save_pdf(response: requests.Response, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".part")
    hasher = hashlib.sha256()
    first = b""
    try:
        with temp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                if len(first) < 5:
                    first += chunk[: 5 - len(first)]
                hasher.update(chunk)
                handle.write(chunk)
        if not first.startswith(b"%PDF"):
            raise RuntimeError("Google Drive did not return a PDF.")
        temp.replace(dest)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    finally:
        response.close()
    return hasher.hexdigest()


def download_one(report: dict[str, str], dest: Path) -> dict[str, str]:
    response = SESSION.get(
        "https://drive.usercontent.google.com/download",
        params={"id": report["drive_id"], "export": "download", "confirm": "t"},
        stream=True,
        timeout=180,
    )
    response.raise_for_status()
    digest = _save_pdf(response, dest)
    return {
        **report,
        "local_path": str(dest.relative_to(PACKAGE_DIR)),
        "bytes": str(dest.stat().st_size),
        "sha256": digest,
        "status": "downloaded",
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _is_pdf(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF"


def _already_present(report: dict[str, str], dest: Path) -> dict[str, str]:
    return {
        **report,
        "local_path": str(dest.relative_to(PACKAGE_DIR)),
        "bytes": str(dest.stat().st_size),
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "status": "already_present",
        "downloaded_at": "",
    }


def _failed(report: dict[str, str], exc: Exception) -> dict[str, str]:
    return {
        **report,
        "local_path": "",
        "bytes": "",
        "sha256": "",
        "status": f"failed: {exc}",
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_manifest(rows: list[dict[str, str]]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def download_reports(reports: list[dict[str, str]], force: bool = False) -> list[dict[str, str]]:
    saved: list[dict[str, str]] = []
    for index, report in enumerate(reports, start=1):
        dest = RAW_DIR / f"{report['slug']}.pdf"
        if dest.exists() and _is_pdf(dest) and not force:
            print(f"[{index}/{len(reports)}] skip {dest.name}")
            saved.append(_already_present(report, dest))
            continue
        print(f"[{index}/{len(reports)}] {report['title']}")
        try:
            saved.append(download_one(report, dest))
        except Exception as exc:
            print(f"  failed: {exc}")
            saved.append(_failed(report, exc))
        time.sleep(1)
    write_manifest(saved)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="Print the report list and exit.")
    parser.add_argument("--only", default="", help="Download titles containing this text.")
    parser.add_argument("--force", action="store_true", help="Download again even if the PDF exists.")
    args = parser.parse_args()

    reports = list_reports()
    if args.only:
        reports = [row for row in reports if args.only.lower() in row["title"].lower()]
        if not reports:
            raise SystemExit(f"No report title contains {args.only!r}.")
    if args.list:
        for row in reports:
            print(f"{row['title']}  {row['source_url']}")
        print(f"{len(reports)} reports")
        return

    saved = download_reports(reports, force=args.force)
    ok = sum(1 for row in saved if row["status"] in ("downloaded", "already_present"))
    print(f"{ok}/{len(saved)} PDFs in {RAW_DIR}")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
