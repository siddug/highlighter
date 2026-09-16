"""Render web/article.content.html into the static reader page.

The content file is the source of truth: the TipTap editor at /edit.html writes it, and
this regenerates the static article so the reader view never drifts from what was edited.
serve.py calls this automatically after every save.

Run:  python3 scripts/build_article.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "web" / "article.content.html"
TEMPLATE = ROOT / "web" / "article.template.html"
ARTICLE = ROOT / "web" / "article.html"
DIST = ROOT / "dist" / "article.html"

MARKER = "<!--CONTENT-->"


def render() -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    if MARKER not in template:
        raise SystemExit(f"{TEMPLATE} has no {MARKER} marker")
    return template.replace(MARKER, CONTENT.read_text(encoding="utf-8"))


def main() -> None:
    html = render()
    ARTICLE.write_text(html, encoding="utf-8")
    written = [ARTICLE]

    # Keep the served copy in step when a build already exists, so edits are visible
    # without re-running vite.
    if DIST.parent.exists():
        DIST.write_text(html, encoding="utf-8")
        written.append(DIST)

    for path in written:
        print(f"wrote {path.relative_to(ROOT)}  ({len(html):,} chars)")


if __name__ == "__main__":
    try:
        main()
    except (OSError, SystemExit) as exc:
        print(f"build_article failed: {exc}", file=sys.stderr)
        raise
