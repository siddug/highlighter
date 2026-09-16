"""Generate a standalone page for hand-labelling a gold set.

Every number in this project is measured against an LLM that agrees with itself 0.704 of
the time. That tells us how well the model imitates the oracle, and nothing at all about
whether the oracle matches a human's actual taste.

This builds a self-contained HTML tool — no server, no build step, no dependencies — for
labelling held-out paragraphs by clicking words. Progress is saved to localStorage after
every click, so it survives a closed tab, and partial work is still usable.

Run:  python3 -m salience.gold.build
      open data/gold/label.html
      ... click words, then Export ...
      python3 -m salience.gold.compare ~/Downloads/gold.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from salience.features import WORD, extract_features
from salience.train.data import load_examples, split

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labeled" / "labels10k.jsonl"
OUT_DIR = ROOT / "data" / "gold"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Gold set — mark the words that carry the meaning</title>
<style>
:root {
  --paper:#fbfaf8; --card:#fff; --ink:#14130e; --soft:#55524a; --faint:#8f8b80;
  --rule:#e4e0d6; --flame:hsl(22 94% 52%);
  --serif:ui-serif,Georgia,serif; --sans:ui-sans-serif,system-ui,sans-serif;
  --mono:ui-monospace,Menlo,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 var(--sans)}
.wrap{max-width:52rem;margin:0 auto;padding:2rem 1.25rem 6rem}
header{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;
  border-bottom:1px solid var(--rule);padding-bottom:.9rem;margin-bottom:1.5rem}
h1{font-family:var(--serif);font-size:1.3rem;margin:0;font-weight:600}
.meta{font-family:var(--mono);font-size:.75rem;color:var(--faint)}
.progress{height:4px;background:var(--rule);border-radius:2px;overflow:hidden;margin-bottom:1.5rem}
.progress span{display:block;height:100%;background:var(--flame);transition:width .2s}
.guide{background:var(--card);border:1px solid var(--rule);border-radius:8px;
  padding:.85rem 1.1rem;margin-bottom:1.25rem;font-size:.86rem;color:var(--soft)}
.guide b{color:var(--ink)}
.card{background:var(--card);border:1px solid var(--rule);border-radius:10px;
  padding:1.6rem 1.7rem;margin-bottom:1.1rem}
.domain{font-family:var(--mono);font-size:.66rem;letter-spacing:.12em;text-transform:uppercase;
  color:var(--faint);margin-bottom:1rem}
.text{font-family:var(--serif);font-size:1.12rem;line-height:1.95}
.w{cursor:pointer;border-radius:3px;padding:.06em .1em;margin:0 -.02em;
  transition:background .12s,color .12s;user-select:none}
.w:hover{background:hsl(22 94% 52% / .14)}
.w.on{background:hsl(22 94% 52% / .5);font-weight:500}
.bar{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap;margin-top:1.4rem;
  padding-top:1.1rem;border-top:1px solid var(--rule)}
button{appearance:none;border:1px solid var(--rule);background:var(--card);color:var(--ink);
  border-radius:999px;padding:.42rem 1.1rem;font:inherit;font-size:.86rem;cursor:pointer}
button:hover{border-color:var(--flame);color:var(--flame)}
button.primary{background:var(--flame);border-color:var(--flame);color:#fff}
button.primary:hover{opacity:.9;color:#fff}
.count{font-family:var(--mono);font-size:.8rem;color:var(--faint);margin-left:auto}
.count b{color:var(--ink)}
.count.over b{color:var(--flame)}
.residue{font-family:var(--mono);font-size:.84rem;line-height:1.9;color:var(--soft);
  margin-top:1rem;padding-top:.9rem;border-top:1px dashed var(--rule);min-height:1.9rem}
.residue:empty::before{content:'click words above';color:var(--faint)}
kbd{font-family:var(--mono);font-size:.72rem;background:var(--rule);border-radius:3px;
  padding:.1em .4em}
.done{text-align:center;padding:3rem 1rem}
.done h2{font-family:var(--serif)}
</style></head><body><div class="wrap">
<header>
  <h1>Mark the words that carry the meaning</h1>
  <span class="meta" id="pos"></span>
</header>
<div class="progress"><span id="bar" style="width:0%"></span></div>

<div class="guide">
  Click the words someone could read <b>on their own</b>, in order, and still know what the
  paragraph said. Keep negations — <b>not</b>, <b>without</b>, <b>failed</b> — they flip the
  meaning. Aim for roughly a fifth of the words, but trust your judgement over the counter.
  <br />
  <kbd>&larr;</kbd> <kbd>&rarr;</kbd> move · <kbd>u</kbd> undo all · your work saves itself.
</div>

<div id="stage"></div>

<script>
const DATA = __DATA__;
const KEY = 'gold-v1';
let marks = JSON.parse(localStorage.getItem(KEY) || '{}');
let i = Number(localStorage.getItem(KEY + ':at') || 0);

const $ = id => document.getElementById(id);
const save = () => {
  localStorage.setItem(KEY, JSON.stringify(marks));
  localStorage.setItem(KEY + ':at', String(i));
};

function render() {
  if (i >= DATA.length) return finish();
  const item = DATA[i];
  const on = new Set(marks[item.id] || []);
  const target = Math.round(item.words.length * 0.2);

  $('pos').textContent = `${i + 1} / ${DATA.length} · ${Object.keys(marks).length} done`;
  $('bar').style.width = `${(Object.keys(marks).length / DATA.length) * 100}%`;

  const words = item.words.map((w, k) =>
    `<span class="w ${on.has(k) ? 'on' : ''}" data-k="${k}">${w}</span>`).join(' ');

  $('stage').innerHTML = `
    <div class="card">
      <div class="domain">${item.domain}</div>
      <div class="text" id="text">${words}</div>
      <div class="residue" id="residue"></div>
      <div class="bar">
        <button id="prev">&larr; back</button>
        <button id="clear">undo all</button>
        <button class="primary" id="next">next &rarr;</button>
        <span class="count ${on.size > target * 1.5 ? 'over' : ''}">
          <b>${on.size}</b> of ${item.words.length} · target ~${target}</span>
      </div>
    </div>`;

  $('text').onclick = e => {
    const k = e.target.dataset?.k;
    if (k === undefined) return;
    const set = new Set(marks[item.id] || []);
    set.has(+k) ? set.delete(+k) : set.add(+k);
    marks[item.id] = [...set].sort((a, b) => a - b);
    save(); render();
  };
  $('next').onclick = () => { if (!marks[item.id]) marks[item.id] = []; i++; save(); render(); };
  $('prev').onclick = () => { if (i > 0) i--; save(); render(); };
  $('clear').onclick = () => { marks[item.id] = []; save(); render(); };
  $('residue').textContent = item.words.filter((_, k) => on.has(k)).join(' ');
}

function finish() {
  const blob = new Blob([JSON.stringify({ marks }, null, 1)], { type: 'application/json' });
  $('stage').innerHTML = `<div class="done"><h2>Done — ${Object.keys(marks).length} paragraphs</h2>
    <p>Download the file, then run<br><code>python3 -m salience.gold.compare &lt;file&gt;</code></p>
    <p><a id="dl" download="gold.json">Download gold.json</a></p>
    <button id="back">&larr; review</button></div>`;
  $('dl').href = URL.createObjectURL(blob);
  $('back').onclick = () => { i = DATA.length - 1; render(); };
}

addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') $('next')?.click();
  if (e.key === 'ArrowLeft') $('prev')?.click();
  if (e.key === 'u') $('clear')?.click();
});
render();
</script></div></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    with args.labels.open(encoding="utf-8") as fh:
        records = {json.loads(line)["id"]: json.loads(line) for line in fh if line.strip()}

    # Held-out only: labelling paragraphs the model trained on would measure nothing.
    _, val = split(load_examples(args.labels), 0.15, args.seed)
    by_domain: dict[str, list] = defaultdict(list)
    for ex in val:
        by_domain[ex.domain].append(records[ex.pid])

    rng = random.Random(args.seed)
    per = args.count // len(by_domain)
    chosen = []
    for domain in sorted(by_domain):
        pool = by_domain[domain]
        chosen.extend(rng.sample(pool, min(per, len(pool))))
    rng.shuffle(chosen)

    data = [
        {
            "id": r["id"],
            "domain": r["domain"],
            "words": [f.text for f in extract_features(r["text"]) if f.cls == WORD],
        }
        for r in chosen
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "label.html").write_text(
        PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8"
    )
    (args.out / "sample.json").write_text(json.dumps([r["id"] for r in chosen]))

    counts = defaultdict(int)
    for r in chosen:
        counts[r["domain"]] += 1
    print(f"{len(data)} held-out paragraphs -> {args.out / 'label.html'}")
    for d, c in sorted(counts.items()):
        print(f"  {d:<11}{c}")
    print(f"\nopen {args.out / 'label.html'}")


if __name__ == "__main__":
    main()
