"""Render method_figure.html to PNG + PDF via headless Chromium (Playwright).

Usage: .venv/bin/python3 viz/render_figure.py
Outputs viz/method_figure.png and viz/method_figure.pdf next to the HTML.
"""
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
HTML = HERE / "method_figure.html"
PNG = HERE / "method_figure.png"
PDF = HERE / "method_figure.pdf"


def main() -> None:
    url = HTML.as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        # 2x device scale → crisp PNG for slides/paper.
        page = browser.new_page(viewport={"width": 1240, "height": 900},
                                device_scale_factor=2)
        page.goto(url, wait_until="networkidle")
        # Tight PNG around the figure content.
        fig = page.query_selector(".fig")
        (fig or page).screenshot(path=str(PNG))
        # PDF: print background colors, landscape, fit the wide layout.
        page.pdf(path=str(PDF), landscape=True, print_background=True,
                 prefer_css_page_size=False, width="13in", height="9in",
                 margin={"top": "0.3in", "bottom": "0.3in",
                         "left": "0.3in", "right": "0.3in"})
        browser.close()
    print(f"Wrote {PNG} ({PNG.stat().st_size} bytes)")
    print(f"Wrote {PDF} ({PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
