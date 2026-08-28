"""Smoke the engine and every Streamlit page, including the Dashboard slider."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.live_match import match_forecast
from lib.load import load_studio

PAGES = [
    "Dashboard",
    "Results",
    "Data",
    "Features",
    "Training",
    "Comparison",
    "Uncertainty",
    "Novelty",
]


def check_engine() -> None:
    live = load_studio()["live"]
    low = match_forecast(live, "carrot", "colombo", 0.25)
    mid = match_forecast(live, "carrot", "colombo", 0.85)
    high = match_forecast(live, "carrot", "colombo", 1.50)
    assert low and mid and high
    assert low["predicted_price"] > mid["predicted_price"] > high["predicted_price"], (
        low["predicted_price"],
        mid["predicted_price"],
        high["predicted_price"],
    )
    print("engine ok", low["predicted_price"], mid["predicted_price"], high["predicted_price"])


def check_pages(base: str = "http://localhost:8501") -> None:
    from playwright.sync_api import sync_playwright

    out = Path(__file__).resolve().parent / "shots"
    out.mkdir(exist_ok=True)
    errors: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        for name in PAGES:
            page.goto(f"{base}/?page={name}", wait_until="networkidle", timeout=90000)
            page.wait_for_timeout(1200)
            plot = page.locator(".js-plotly-plot")
            if plot.count():
                plot.first.wait_for(state="visible", timeout=20000)
                page.wait_for_timeout(800)
            elif name != "Results":
                page.wait_for_timeout(1800)
            exception = page.locator("[data-testid='stException']")
            if exception.count():
                errors.append(f"{name}: {exception.first.inner_text()[:240]}")
            page.screenshot(path=str(out / f"{name.lower()}.png"), full_page=True)
            print("page", name)

        page.goto(f"{base}/?page=Dashboard", wait_until="networkidle", timeout=90000)
        page.wait_for_selector("[data-expected-price]", timeout=30000)
        page.wait_for_timeout(1500)
        hero = page.locator("[data-expected-price]")
        before = hero.get_attribute("data-expected-price")
        slider = page.locator("[data-testid='stSlider']").first
        box = slider.bounding_box()
        if not box:
            errors.append("Dashboard slider not visible")
        else:
            page.mouse.click(box["x"] + 18, box["y"] + box["height"] / 2)
            page.wait_for_timeout(2500)
        after = hero.get_attribute("data-expected-price")
        page.screenshot(path=str(out / "dashboard-slider.png"), full_page=True)
        print("slider", before, "->", after)
        if before == after:
            errors.append(f"Dashboard price did not move after slider ({before})")
        browser.close()

    if errors:
        raise SystemExit(" | ".join(errors))
    print("pages ok")


if __name__ == "__main__":
    check_engine()
    if "--pages" in sys.argv:
        check_pages()
