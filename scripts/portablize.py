"""Convert the article into a portable TipTap-native document.

The article was written as bespoke HTML: callout boxes, definition boxes, worked examples,
parameter-budget strips, inline SVG figures. All of that renders beautifully here and
survives nothing — paste it into a blog and the custom blocks are stripped, leaving holes.

This rewrites it using only elements a standard TipTap schema understands:

    div.callout / define / worked / reading-demo  ->  blockquote
    div.budget                                    ->  table
    div.part-head                                 ->  heading + paragraph
    figure with svg or bars                       ->  image + italic caption
    figure with a code listing                    ->  code block + italic caption
    nav.toc                                       ->  bold text + list
    p.filename                                    ->  italic paragraph
    pre with colour spans                         ->  plain code block

Nothing is lost except the syntax colouring inside code blocks, which was hand-written
markup rather than content — any blog will re-highlight it.

Run:  python3 scripts/render_figures.py && python3 scripts/portablize.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "web" / "article.content.html"
BACKUP = ROOT / "web" / "article.content.custom-html.bak"
FIGURES = ROOT / "web" / "figures"


def new_tag(soup: BeautifulSoup, name: str, text: str = "", **attrs):
    tag = soup.new_tag(name, **attrs)
    if text:
        tag.string = text
    return tag


def figure_slug(label: str) -> str:
    match = re.match(r"Figure (\d+)", label)
    number = match.group(1).zfill(2) if match else "xx"
    rest = re.sub(r"[^a-z0-9]+", "-", label.split("—")[-1].strip().lower()).strip("-")
    return f"fig-{number}-{rest}"[:60]


def caption_paragraph(soup: BeautifulSoup, figure) -> object | None:
    cap = figure.find("figcaption")
    if not cap:
        return None
    para = soup.new_tag("p")
    em = soup.new_tag("em")
    for child in list(cap.children):
        em.append(child.extract())
    para.append(em)
    return para


def convert_figure(soup: BeautifulSoup, figure) -> list:
    """A figure becomes an image, or — when it is really text — markup that stays editable."""
    label_el = figure.select_one(".fig-label")
    label = label_el.get_text(strip=True) if label_el else "Figure"
    caption = caption_paragraph(soup, figure)
    out: list = []

    is_picture = figure.find("svg") is not None or figure.select_one(".bars") is not None
    if is_picture:
        path = FIGURES / f"{figure_slug(label)}.png"
        if not path.exists():
            raise SystemExit(f"missing {path} — run scripts/render_figures.py first")
        img = soup.new_tag("img", src=f"./figures/{path.name}", alt=label)
        para = soup.new_tag("p")
        para.append(img)
        out.append(para)
    else:
        # Text-bearing figures keep their content: a picture of a code listing cannot be
        # edited, searched or read by a screen reader.
        heading = soup.new_tag("p")
        strong = new_tag(soup, "strong", label)
        heading.append(strong)
        out.append(heading)
        for node in figure.find_all(["pre", "div"], recursive=False):
            if node.name == "pre":
                out.append(plain_code(soup, node))
            elif "reading-demo" in (node.get("class") or []):
                out.append(convert_reading_demo(soup, node))

    if caption:
        out.append(caption)
    return out


def plain_code(soup: BeautifulSoup, pre) -> object:
    """Flatten a coloured code block to plain text inside pre > code."""
    text = pre.get_text()
    fresh = soup.new_tag("pre")
    code = soup.new_tag("code")
    code.string = text
    fresh.append(code)
    return fresh


def lead_in(soup: BeautifulSoup, block, selector: str, style: str):
    """Pull a block's label out and re-attach it as emphasis on the first paragraph."""
    tag_el = block.select_one(selector)
    label = tag_el.get_text(strip=True) if tag_el else ""
    if tag_el:
        tag_el.decompose()
    if not label:
        return None
    marker = soup.new_tag(style)
    marker.string = label
    return marker


def convert_boxed(soup: BeautifulSoup, block, selector: str) -> object:
    """callout / define / worked all become a blockquote led by their label in bold."""
    quote = soup.new_tag("blockquote")
    marker = lead_in(soup, block, selector, "strong")

    paragraphs = block.find_all("p", recursive=False) or [block]
    for i, para in enumerate(paragraphs):
        fresh = soup.new_tag("p")
        if i == 0 and marker:
            fresh.append(marker)
            fresh.append(NavigableString(" — "))
        for child in list(para.children):
            fresh.append(child.extract())
        quote.append(fresh)

    # Some boxes carry a table or list rather than only paragraphs.
    for extra in block.find_all(["table", "ul", "ol", "pre"], recursive=False):
        quote.append(extra.extract() if extra.name != "pre" else plain_code(soup, extra))
    return quote


def convert_reading_demo(soup: BeautifulSoup, block) -> object:
    quote = soup.new_tag("blockquote")
    marker = lead_in(soup, block, ".tag", "em")
    para = soup.new_tag("p")
    if marker:
        para.append(marker)
        para.append(soup.new_tag("br"))
    for child in list(block.children):
        para.append(child.extract())
    quote.append(para)
    return quote


def convert_budget(soup: BeautifulSoup, block) -> list:
    label_el = block.select_one(".tag")
    label = label_el.get_text(strip=True) if label_el else "Parameter budget"

    heading = soup.new_tag("p")
    heading.append(new_tag(soup, "strong", label))

    table = soup.new_tag("table")
    head = soup.new_tag("thead")
    row = soup.new_tag("tr")
    row.append(new_tag(soup, "th", "stage"))
    row.append(new_tag(soup, "th", "parameters"))
    head.append(row)
    table.append(head)

    body = soup.new_tag("tbody")
    for entry in block.select("dl > div"):
        dt = entry.find("dt")
        dd = entry.find("dd")
        tr = soup.new_tag("tr")
        tr.append(new_tag(soup, "td", dt.get_text(strip=True) if dt else ""))
        tr.append(new_tag(soup, "td", dd.get_text(strip=True) if dd else ""))
        body.append(tr)

    foot = block.select_one(".foot")
    if foot:
        spans = foot.find_all("span", recursive=False)
        tr = soup.new_tag("tr")
        tr.append(new_tag(soup, "td", spans[0].get_text(strip=True) if spans else "running total"))
        tr.append(new_tag(soup, "td", spans[-1].get_text(" ", strip=True) if spans else ""))
        body.append(tr)
    table.append(body)
    return [heading, table]


def convert_part_head(soup: BeautifulSoup, block) -> list:
    label = block.select_one(".label")
    title = block.find("h2")
    blurb = block.find("p")
    out = []
    text = title.get_text(strip=True) if title else ""
    if label:
        text = f"{label.get_text(strip=True).title()} — {text}"
    out.append(new_tag(soup, "h2", text))
    if blurb:
        para = soup.new_tag("p")
        for child in list(blurb.children):
            para.append(child.extract())
        out.append(para)
    return out


def convert_toc(soup: BeautifulSoup, nav) -> list:
    out = [new_tag(soup, "p", "")]
    out[0].append(new_tag(soup, "strong", "Contents"))
    items = soup.new_tag("ol")
    for entry in nav.select("ol > li"):
        li = soup.new_tag("li")
        li.string = entry.get_text(" ", strip=True)
        items.append(li)
    out.append(items)
    return out


def main() -> None:
    if not BACKUP.exists():
        shutil.copy(CONTENT, BACKUP)
        print(f"backed up original to {BACKUP.name}")

    soup = BeautifulSoup(BACKUP.read_text(encoding="utf-8"), "html.parser")
    counts: dict[str, int] = {}

    def bump(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    for nav in soup.select("nav.toc"):
        nav.replace_with(*convert_toc(soup, nav))
        bump("table of contents")

    for block in soup.select("div.part-head"):
        block.replace_with(*convert_part_head(soup, block))
        bump("part divider -> heading")

    for figure in soup.find_all("figure"):
        figure.replace_with(*convert_figure(soup, figure))
        bump("figure")

    for selector, label in (("div.callout", ".tag"), ("div.define", ".term"), ("div.worked", ".tag")):
        for block in soup.select(selector):
            block.replace_with(convert_boxed(soup, block, label))
            bump(f"{selector} -> blockquote")

    for block in soup.select("div.reading-demo"):
        block.replace_with(convert_reading_demo(soup, block))
        bump("reading demo -> blockquote")

    for block in soup.select("div.budget"):
        block.replace_with(*convert_budget(soup, block))
        bump("budget -> table")

    for para in soup.select("p.filename"):
        fresh = soup.new_tag("p")
        fresh.append(new_tag(soup, "em", para.get_text(strip=True)))
        para.replace_with(fresh)
        bump("filename -> italic")

    for pre in soup.find_all("pre"):
        pre.replace_with(plain_code(soup, pre))
        bump("code block flattened")

    # Section numbers live in <span class="num"> inside each heading, where CSS put them on
    # their own line. Without that CSS they would render as "01The problem", so fold the
    # number into the heading text before the class is stripped.
    for span in soup.select("h2 > span.num, h3 > span.num, h4 > span.num"):
        number = span.get_text(strip=True)
        span.replace_with(NavigableString(f"{number} · " if number else ""))
        bump("heading number folded in")

    # Strip presentational attributes; the portable document carries no bespoke styling.
    for element in soup.find_all(True):
        if element.name == "img":
            continue
        for attr in ("class", "style", "data-raw"):
            element.attrs.pop(attr, None)

    # <b> is a presentational leftover; TipTap emits <strong>.
    for bold in soup.find_all("b"):
        bold.name = "strong"
        bump("b -> strong")

    # Spans only existed to carry classes. With those gone they are empty wrappers that
    # some editors will choke on, so merge their text into the parent.
    for span in soup.find_all("span"):
        span.unwrap()
        bump("bare span unwrapped")

    CONTENT.write_text(str(soup), encoding="utf-8")

    print()
    for name, n in sorted(counts.items()):
        print(f"  {n:>3}  {name}")
    remaining = {t.name for t in soup.find_all(True)}
    print(f"\nelements now in use: {', '.join(sorted(remaining))}")
    print(f"wrote {CONTENT.relative_to(ROOT)}  ({len(str(soup)):,} chars)")


if __name__ == "__main__":
    main()
