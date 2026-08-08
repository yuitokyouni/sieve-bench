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
# 引数でソースを選べる。既定は math.html（後方互換）。
_name = sys.argv[1] if len(sys.argv) > 1 else "math"
SRC = HERE / f"{_name}.html"
OUT = HERE / f"sieve-bench-{_name}.pdf"

FOOTER = ('<div style="width:100%;font-size:8px;color:#889;'
          'font-family:sans-serif;padding:0 18mm;text-align:right">'
          f'sieve-bench / {_name} — <span class="pageNumber"></span>'
          ' / <span class="totalPages"></span></div>')


def main():
    if not SRC.exists():
        sys.exit(f"{SRC} が無い")
    import os
    exe = os.environ.get("CHROMIUM_PATH")
    if not exe:
        for cand in ("/opt/pw-browsers/chromium",):
            if os.path.exists(cand):
                exe = cand
                break
    with sync_playwright() as pw:
        browser = (pw.chromium.launch(executable_path=exe) if exe
                   else pw.chromium.launch())
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
