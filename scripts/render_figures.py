"""Render the article's inline SVG and bar-chart figures to PNG images.

The article is written to be pasted into a blog that understands TipTap plus images, but
not bespoke HTML blocks. Hand-built SVG and CSS bar charts are exactly the sort of thing
that will not survive that trip, so they become images instead.

Two of the twelve figures are deliberately skipped: Figure 1 is prose and Figure 4 is a
code listing. Turning readable text into a picture would lose the ability to edit or search
it, so those convert to blockquotes and a code block instead.

Run:  python3 scripts/render_figures.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "web" / "article.content.html"
CSS = ROOT / "web" / "article.css"
OUT = ROOT / "web" / "figures"

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WIDTH = 880
SCALE = 2  # retina, so the images stay sharp when a blog scales them


def slug(label: str) -> str:
    match = re.match(r"Figure (\d+)", label)
    number = match.group(1).zfill(2) if match else "xx"
    rest = re.sub(r"[^a-z0-9]+", "-", label.split("—")[-1].strip().lower()).strip("-")
    return f"fig-{number}-{rest}"[:60]


def render(html: str, path: Path) -> None:
    """Screenshot one figure, then trim the surrounding whitespace."""
    page = f"""<!doctype html><meta charset="utf-8">
<link rel="stylesheet" href="{CSS.as_uri()}">
<style>
  body {{ margin:0; padding:16px; background:#fbfaf8; width:{WIDTH}px; }}
  figure {{ margin:0; }}
</style>
<div>{html}</div>"""

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "fig.html"
        src.write_text(page, encoding="utf-8")
        shot = Path(tmp) / "shot.png"
        subprocess.run(
            [
                CHROME, "--headless", "--disable-gpu", "--no-sandbox",
                f"--force-device-scale-factor={SCALE}",
                "--hide-scrollbars", "--virtual-time-budget=4000",
                f"--window-size={WIDTH + 32},2400",
                f"--screenshot={shot}", src.as_uri(),
            ],
            capture_output=True,
            check=True,
        )
        trim(shot, path)


def trim(src: Path, dest: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    image = Image.open(src).convert("RGB")
    background = Image.new("RGB", image.size, image.getpixel((1, 1)))
    from PIL import ImageChops  # noqa: PLC0415

    box = ImageChops.difference(image, background).getbbox()
    pad = 8 * SCALE
    if box:
        left, top, right, bottom = box
        box = (
            max(0, left - pad),
            max(0, top - pad),
            min(image.width, right + pad),
            min(image.height, bottom + pad),
        )
        image = image.crop(box)
    image.save(dest, optimize=True)


def main() -> None:
    if not Path(CHROME).exists():
        raise SystemExit(f"Chrome not found at {CHROME}")

    soup = BeautifulSoup(CONTENT.read_text(encoding="utf-8"), "html.parser")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    rendered = skipped = 0
    for figure in soup.find_all("figure"):
        label_el = figure.select_one(".fig-label")
        label = label_el.get_text(strip=True) if label_el else "figure"

        is_picture = figure.find("svg") is not None or figure.select_one(".bars") is not None
        if not is_picture:
            print(f"  skip   {label}  (text — converts to markup, not an image)")
            skipped += 1
            continue

        # Drop the label and caption: they become surrounding text, not part of the picture.
        body = BeautifulSoup(str(figure), "html.parser")
        for node in body.select(".fig-label, figcaption"):
            node.decompose()

        path = OUT / f"{slug(label)}.png"
        render(str(body), path)
        size = path.stat().st_size
        print(f"  render {label:<52} {size / 1024:>6.0f} KB")
        rendered += 1

    total = sum(p.stat().st_size for p in OUT.glob("*.png"))
    print(f"\n{rendered} rendered, {skipped} left as text · {total / 1024:.0f} KB total -> {OUT}")


if __name__ == "__main__":
    main()
