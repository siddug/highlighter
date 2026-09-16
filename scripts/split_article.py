"""One-time split of article.html into a template and an editable content file.

After this runs, `web/article.content.html` is the source of truth for the prose and
`web/article.template.html` holds the surrounding chrome. `scripts/build_article.py`
recombines them, and the TipTap editor at /edit.html reads and writes the content file
through the server.

Run once:  python3 scripts/split_article.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT / "web" / "article.html"
CONTENT = ROOT / "web" / "article.content.html"
TEMPLATE = ROOT / "web" / "article.template.html"

MARKER = "<!--CONTENT-->"


def main() -> None:
    html = ARTICLE.read_text(encoding="utf-8")

    # The editable region is the main .wrap div, which starts after the masthead's
    # </header> and runs to the final </div> before </body>.
    head_end = html.index("</header>") + len("</header>")
    open_wrap = html.index('<div class="wrap">', head_end)
    body_start = open_wrap + len('<div class="wrap">')
    body_end = html.rindex("</div>\n  </body>")

    prefix = html[:body_start]
    content = html[body_start:body_end]
    suffix = html[body_end:]

    # No reformatting: template + content must reproduce the original byte for byte, so
    # that a co-writing diff shows only what was actually edited.
    TEMPLATE.write_text(prefix + MARKER + suffix, encoding="utf-8")
    CONTENT.write_text(content, encoding="utf-8")

    print(f"template  {len(prefix) + len(suffix):>7,} chars -> {TEMPLATE.name}")
    print(f"content   {len(content):>7,} chars -> {CONTENT.name}")


if __name__ == "__main__":
    main()
