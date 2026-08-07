"""docs/math.html を PDF に組む。

    python3 -m pip install playwright
    python3 docs/build_pdf.py     # → docs/sieve-bench-math.pdf

LaTeX を使っていないのは、この環境に無いからというだけでなく、
和文の組版で外部依存を増やしたくないため。Chromium の印刷パイプラインを使う。
和文は IPAGothic、欧文は Liberation、数式の変数は Liberation Serif の斜体。
"""

import pathlib
import sys

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "math.html"
OUT = HERE / "sieve-bench-math.pdf"

FOOTER = ('<div style="width:100%;font-size:8px;color:#889;'
          'font-family:sans-serif;padding:0 18mm;text-align:right">'
          'sieve-bench / 数学的基礎 — <span class="pageNumber"></span>'
          ' / <span class="totalPages"></span></div>')


def main():
    if not SRC.exists():
        sys.exit(f"{SRC} が無い")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto("file://" + str(SRC))
        page.wait_for_timeout(900)
        page.emulate_media(media="print")
        page.pdf(path=str(OUT), format="A4", print_background=True,
                 display_header_footer=True,
                 header_template="<div></div>", footer_template=FOOTER,
                 margin={"top": "18mm", "bottom": "16mm",
                         "left": "20mm", "right": "20mm"})
        browser.close()
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
