# Message Highlighter

A ~39k-parameter neural model that marks the words in a paragraph that carry its meaning —
enough that reading only the highlights tells you what it said, without summarizing.

The architecture is deliberately cloned from Vercel Labs'
[gpu-lexer](https://gpu-lexer.vercel.app), which does language-agnostic syntax highlighting
with 41,321 parameters in 27KB of WebGPU compute shaders. The point of this project is as
much to learn that recipe as to ship the tool.

```
feature-bag embedding → local 5-token window → learned Blelloch scan → gated MLP head
```

No learned vocabulary, no transformer, no attention. Tokens are reduced to hand-engineered
integers that index rows of one shared embedding table, and the rows are summed. Global
context comes from a parallel prefix scan with per-tree-level parameters, which reaches the
whole document in log depth.

## Status

Working end to end. Trained on 10,000 LLM-labeled paragraphs, quantized to 6 bits, running
in the browser at **39,361 parameters / 21.1 KB brotli** — smaller than gpu-lexer's 27.4 KB.

| Phase | |
|---|---|
| 1. Thin slice | done — tokenizer, model, CPU inference, demo, cross-language conformance |
| 2. Oracle | done — 12k-paragraph corpus, Mistral labeler, 10k labeled with soft targets |
| 3. Train | done — training loop, 6-bit quantization, base-63 export, weights in the demo |
| 4. WGSL | not started — the CPU path is the reference and the fallback |

### Where it stands (1,497 held-out paragraphs)

| | P@15% | P@20% | AP | Spearman |
|---|---|---|---|---|
| random | 0.264 | 0.254 | 0.297 | −0.010 |
| position (topic sentence) | 0.385 | 0.372 | 0.387 | 0.160 |
| tf-idf | 0.508 | 0.495 | 0.531 | 0.480 |
| not-a-stopword | 0.600 | 0.580 | 0.596 | 0.506 |
| **model (39,361 params)** | **0.765** | **0.725** | **0.747** | **0.661** |

Going from 1k to 10k labeled paragraphs moved P@15% from 0.693 to 0.765 and removed the
overfitting entirely — at 1k, validation peaked by epoch 6 and then declined.

### How much better could it get?

`salience/eval/ceiling.py` measures the oracle against itself. Two independent annotations
of the same paragraph correlate at **ρ = 0.704**, so the 3-sample average we train on has a
reliability of 0.877 and **no model can score above 0.936** — our 0.661 is 71% of the
achievable maximum, not of 1.0.

`salience/eval/sweep.py` then tries every lever. None of them work:

| | params | train ρ | val ρ |
|---|---|---|---|
| baseline | 39,361 | 0.690 | 0.661 |
| d=64 + bigger head | 116,961 | 0.717 | **0.661** |
| window ±4 | 39,489 | 0.689 | 0.658 |
| lr 1e-2 | 39,361 | 0.706 | 0.660 |
| cross-entropy only | 39,361 | 0.688 | 0.662 |

Tripling parameters raises *training* score and leaves validation untouched. Adding corpus
IDF as a feature buys +0.005. The diagnosis is in `salience/eval/idf_probe.py` and a
frequency breakdown: **mean error is flat at 0.34 for every word seen fewer than 500 times
in training** — unseen, seen twice, seen fifty times, identical. The model has learned the
generic signals (stopwords, casing, numbers, position) and is learning nothing word-specific.

The likely fix is semantic features — cluster a pretrained embedding space and ship the
cluster id — which reintroduces a vocabulary and trades against the size thesis. See
section 18 of the article.

### What it looks like

Top 20% of words on text from outside the training distribution:

> **Slack** — "Hey team, heads up: the staging deploy failed twice this morning because the migration lock wasn't released. I've rolled back to build 412 and we are NOT shipping today…"
> → `staging deploy failed twice lock wasn't build 412 NOT post 4pm`

> **Contract** — "The tenant shall not sublet the premises without the prior written consent of the landlord, which consent shall not be unreasonably withheld…"
> → `tenant not sublet premises without consent not sublease liable`

Negation survives, which is the failure mode every frequency-based keyword extractor has:
drop the "not" and the residue asserts the opposite of the source.

## Layout

```
shared/spec/features.md    normative spec — the source of truth for both implementations
shared/golden/             cross-language fixtures (features, scan, end-to-end scores)
weights/model.json         6-bit weights, one base-63 character per parameter

ts/src/features.ts         tokenizer + feature packing
ts/src/model.ts            dimensions, weight container, initialization
ts/src/cpu.ts              CPU reference forward pass
ts/src/weights.ts          base-63 decoder

salience/features.py       the same tokenizer, in Python
salience/data/corpus.py    mixed-domain corpus fetcher
salience/oracle/           LLM labeling: prompt, batch runner, review page
salience/train/            scan, model, data, loss, training loop, quantization, export
salience/eval/             metrics, baselines, comparison report

scripts/make_golden.py     regenerate feature fixtures
scripts/make_scan_golden.py  regenerate scan fixtures
web/                       Vite demo
```

## Running it

```bash
npm install
npm run dev                          # client at localhost:5173
npm test                             # TypeScript, incl. cross-language conformance
python3 -m pytest salience/tests -q  # Python
```

### The app

```bash
npm run build && python3 serve.py    # http://localhost:8000
```

Three pages: the highlighter at `/`, a long-form write-up of how it was built at
`/article.html`, and a TipTap editor for that article at `/edit.html`.

The editor writes through to `web/article.content.html` on disk — not localStorage — so
the article is co-writable: edit in the browser, and the change is a file both authors can
read and diff. Saving also regenerates the static reader page, so the two never drift.

The article is written in a **portable subset** — paragraphs, headings, lists,
blockquotes, tables, code blocks and images, and nothing else. It was originally bespoke
HTML (callout boxes, definition boxes, inline SVG), which rendered well here and would
survive nothing: paste it into a blog and the custom blocks are stripped, leaving holes.
`scripts/render_figures.py` turns the ten picture-figures into PNGs and
`scripts/portablize.py` rewrites the rest — callouts and definitions become blockquotes,
budget strips become tables. The two text-bearing figures stay as text, because a picture
of a code listing cannot be edited, searched, or read aloud.

`web/edit/roundtrip.test.ts` is the load-bearing part. ProseMirror silently discards
anything its schema does not recognise, so ten assertions pin images, blockquotes, tables,
headings, code blocks, highlights, heading anchors and total word count — and require the
second save to be byte-identical to the first.

`serve.py` is standard library only — no web framework. It serves the built client and
exposes the PyTorch model at `POST /api/score`, so the **Run on: browser / server** toggle
runs the same paragraph through both runtimes and displays the largest per-word
disagreement. It reads about `6e-7`, which is the conformance guarantee made visible.

```bash
curl -s localhost:8000/api/health
curl -s -X POST localhost:8000/api/score \
  -H 'Content-Type: application/json' \
  -d '{"text":"Engineers did not report any fault."}'
```

For client work, `npm run dev` proxies `/api` to port 8000, so both can run at once.
`?runtime=server` preselects the server path.

Worth noting: the browser is *faster* — roughly 17 ms against PyTorch's 60 ms on a
65-word paragraph. At 39k parameters the framework overhead dominates the arithmetic,
which is much of the argument for shipping the model to the client in the first place.

Rebuilding the whole pipeline from scratch:

```bash
python3 -m salience.data.corpus --per-domain 2400        # fetch 12k paragraphs
secretspec set MISTRAL_API_KEY                           # once
secretspec run --reason "label corpus" -- \
  python3 -m salience.oracle.run --limit 10000 --samples 3
python3 -m salience.oracle.review                        # eyeball data/labeled/review.html
python3 -m salience.train.run --epochs 40
python3 -m salience.eval.report                          # vs. baselines
python3 -m salience.train.export                         # -> weights/model.json
npm test                                                 # confirm the browser agrees
```

## Conformance is the whole discipline

Three implementations have to agree: the PyTorch trainer, the TypeScript CPU path, and
eventually the WGSL shaders. Each has its own fixture, generated by Python and verified by
TypeScript.

| Fixture | Pins |
|---|---|
| `golden/features.jsonl` | tokenizer + feature packing, 2,000 adversarial inputs |
| `golden/scan.jsonl` | the learned Blelloch scan, 8 sequence lengths |
| `golden/scores.jsonl` | the whole chain, end to end, on held-out paragraphs |

After changing `shared/spec/features.md`, regenerate and run both suites:

```bash
python3 scripts/make_golden.py && python3 scripts/make_scan_golden.py
npm test && python3 -m pytest salience/tests -q
```

**Never edit a fixture to make a test pass.** If conformance fails, one implementation has
drifted from the spec — read `shared/spec/features.md` to decide which one is wrong.

## Notes from building it

Things that were not obvious, recorded so they are not rediscovered the hard way.

- **The scan is not associative.** Its combine operator carries a different parameter at
  each tree level, so it describes one *specific* balanced binary tree. A sequential loop
  builds a right-leaning tree and gets different numbers; `torch.associative_scan` and the
  minGRU `cumsum` trick compute the mathematical prefix, which is also not this. It has to
  be hand-written, identically, everywhere.
- **Gate the carry, not the decay.** Putting the per-level gate on `z` makes every
  down-sweep parameter dead — the final fold reads only the carry, so those gradients are
  exactly zero. Found by checking for zero gradients, not by any test.
- **Gates must start open.** `FORGET_BIAS_INIT = 2.0` and `LEVEL_GATE_INIT = 3.0`. At the
  default the carry decays as `0.5^distance` and underflows float32 within ~100 tokens: the
  model starts with no long-range context and no gradient with which to learn any.
- **6-bit quantization is free here.** Post-training, held-out Spearman moved 0.6218 →
  0.6221. Quantization-aware training was planned and turned out to be unnecessary.
- **The oracle has a natural rate.** Prompt wording moved it from 40% to 26% and then no
  further; a third attempt made it worse. What the training needs is the *ranking* anyway —
  the budget term in the loss sets the model's own rate.
