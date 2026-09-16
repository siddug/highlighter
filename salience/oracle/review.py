"""Render labeled paragraphs as an HTML page for hand inspection.

Label quality is the ceiling on model quality, so before spending money at scale we look
at what the oracle actually produced. Each paragraph is shown with word shading
proportional to the soft target, followed by the telegraphic residue a reader would see —
which is the real test: can you tell what the paragraph said from the bold words alone?

Run:  python3 -m salience.oracle.review
      open data/labeled/review.html
"""

from __future__ import annotations

import argparse
import html
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from salience.features import WORD, extract_features

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "labeled"

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Oracle label review</title>
<style>
  :root {{ --mark: 24 95% 53%; }}
  body {{ margin: 0; padding: 2rem clamp(1rem,4vw,3rem) 5rem; background: #fbfaf8; color: #16150f;
         font: 15px/1.65 ui-sans-serif, system-ui, sans-serif; }}
  .wrap {{ max-width: 58rem; margin: 0 auto; }}
  h1 {{ font-size: 1.4rem; letter-spacing: -0.02em; margin: 0 0 0.25rem; }}
  .summary {{ color: #6d6a5f; font-size: 0.85rem; margin-bottom: 2rem; }}
  table.sum {{ border-collapse: collapse; margin: 0.75rem 0 0; font-size: 0.8rem;
               font-family: ui-monospace, Menlo, monospace; }}
  table.sum td, table.sum th {{ padding: 0.15rem 0.9rem 0.15rem 0; text-align: left; }}
  .card {{ background: #fff; border: 1px solid #e2ded4; border-radius: 8px;
           padding: 1.1rem 1.25rem; margin-bottom: 1rem; }}
  .meta {{ font-family: ui-monospace, Menlo, monospace; font-size: 0.72rem; color: #6d6a5f;
           text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.6rem;
           display: flex; gap: 1.1rem; flex-wrap: wrap; }}
  .residue {{ margin-top: 0.85rem; padding-top: 0.75rem; border-top: 1px dashed #e2ded4;
              font-family: ui-monospace, Menlo, monospace; font-size: 0.85rem; color: #3b3830; }}
  .residue span {{ color: #9a968a; }}
  mark {{ background: hsl(var(--mark) / var(--a)); border-radius: 2px; padding: 0.05em 0; }}
</style>
<div class="wrap">
<h1>Oracle label review</h1>
<div class="summary">{summary}</div>
{cards}
</div>
"""


def card(rec: dict) -> str:
    feats = extract_features(rec["text"])
    targets = rec["targets"]

    pieces: list[str] = []
    residue: list[str] = []
    w = 0
    for f in feats:
        text = html.escape(f.text)
        if f.cls != WORD:
            pieces.append(text)
            continue
        target = targets[w] if w < len(targets) else 0.0
        if target > 0:
            pieces.append(f'<mark style="--a:{0.12 + 0.6 * target:.2f}">{text}</mark>')
        else:
            pieces.append(text)
        if target >= 0.5:
            residue.append(text)
        w += 1

    rate = sum(1 for t in targets if t >= 0.5) / max(1, len(targets))
    unanimous = sum(1 for t in targets if t == 1.0)
    borderline = sum(1 for t in targets if 0 < t < 1)

    return (
        '<div class="card">'
        f'<div class="meta"><span>{html.escape(rec["domain"])}</span>'
        f'<span>{rec["word_count"]} words</span>'
        f"<span>{rate:.0%} marked</span>"
        f"<span>{unanimous} unanimous</span>"
        f"<span>{borderline} borderline</span></div>"
        f'<div class="text">{"".join(pieces)}</div>'
        f'<div class="residue">{" ".join(residue) or "<span>nothing marked</span>"}</div>'
        "</div>"
    )


def summarize(records: list[dict]) -> str:
    by_domain: dict[str, list[float]] = defaultdict(list)
    agreement: list[float] = []
    for rec in records:
        targets = rec["targets"]
        if not targets:
            continue
        by_domain[rec["domain"]].append(sum(1 for t in targets if t >= 0.5) / len(targets))
        # Fraction of marked words the samples fully agreed on.
        marked = [t for t in targets if t > 0]
        if marked:
            agreement.append(sum(1 for t in marked if t == 1.0) / len(marked))

    rows = "".join(
        f"<tr><td>{html.escape(d)}</td><td>{len(v)}</td>"
        f"<td>{statistics.mean(v):.1%}</td></tr>"
        for d, v in sorted(by_domain.items())
    )
    mean_agree = statistics.mean(agreement) if agreement else 0.0
    return (
        f"{len(records)} paragraphs. "
        f"Of the words any sample marked, {mean_agree:.0%} were marked unanimously — "
        f"higher means the oracle is more self-consistent."
        f'<table class="sum"><tr><th>domain</th><th>n</th><th>marked</th></tr>{rows}</table>'
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=OUT_DIR / "labels.jsonl")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "review.html")
    ap.add_argument("--sample", type=int, default=120, help="paragraphs to render")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    if not args.labels.exists():
        raise SystemExit(f"{args.labels} not found — run: python3 -m salience.oracle.run")

    with args.labels.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    summary = summarize(records)

    # Show a domain-balanced sample rather than the head of the file.
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_domain[rec["domain"]].append(rec)
    rng = random.Random(args.seed)
    per = max(1, args.sample // max(1, len(by_domain)))
    shown: list[dict] = []
    for domain in sorted(by_domain):
        pool = by_domain[domain]
        shown.extend(rng.sample(pool, min(per, len(pool))))
    rng.shuffle(shown)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        PAGE.format(summary=summary, cards="\n".join(card(r) for r in shown)), encoding="utf-8"
    )
    print(f"wrote {len(shown)} of {len(records)} paragraphs to {args.out}")


if __name__ == "__main__":
    main()
