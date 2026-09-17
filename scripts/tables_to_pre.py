"""Turn every table in the article into a preformatted text block.

Markdown editors vary wildly in table support, and the failure mode is bad: cells collapse
into a column of loose lines, so "bit / row / flag / true when / fires / 0 / 632 /
CAPITALIZED" is what a reader sees. A monospace block always renders, everywhere.

Columns are aligned by measured width, numeric columns right-aligned, so the result reads
as a table rather than as a list that happens to have spaces in it.

Operates in place on web/article.content.html, preserving any edits already made to it.

Run:  python3 scripts/tables_to_pre.py
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "web" / "article.content.html"

# A column is right-aligned when its body cells are predominantly numeric: digits with
# optional sign, decimal point, thousands separators, percent, or the × / ~ we use.
NUMERIC = re.compile(r"^[−\-+~<>]?[\d.,]+[%×]?$|^[\d,]+$|^—$|^$")


def cell_text(cell) -> str:
    text = cell.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)


def table_to_text(table) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if cells:
            rows.append([cell_text(c) for c in cells])
    if not rows:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    has_header = bool(table.find("th"))
    body = rows[1:] if has_header else rows

    widths = [max(len(r[i]) for r in rows) for i in range(width)]
    right = [
        all(NUMERIC.match(r[i]) for r in body) if body else False
        for i in range(width)
    ]

    def render(cells: list[str]) -> str:
        parts = [
            cells[i].rjust(widths[i]) if right[i] else cells[i].ljust(widths[i])
            for i in range(width)
        ]
        return "  ".join(parts).rstrip()

    lines = []
    if has_header:
        lines.append(render(rows[0]))
        lines.append("-" * min(len(render(rows[0])), sum(widths) + 2 * (width - 1)))
    lines.extend(render(r) for r in body)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--content", type=Path, default=CONTENT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    backup = args.content.with_suffix(".pre-tables.bak")
    soup = BeautifulSoup(args.content.read_text(encoding="utf-8"), "html.parser")

    tables = soup.find_all("table")
    if not tables:
        print("no tables found — nothing to do")
        return

    for table in tables:
        text = table_to_text(table)
        if not text:
            continue
        pre = soup.new_tag("pre")
        code = soup.new_tag("code")
        code.string = text
        pre.append(code)
        if args.dry_run:
            print(text)
            print("-" * 60)
        table.replace_with(pre)

    if args.dry_run:
        print(f"\n{len(tables)} tables would be converted")
        return

    shutil.copy(args.content, backup)
    args.content.write_text(str(soup), encoding="utf-8")
    print(f"backed up to {backup.name}")
    print(f"converted {len(tables)} tables to <pre> blocks")


if __name__ == "__main__":
    main()
