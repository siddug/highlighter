# Highlighter

A **39,361-parameter** neural model that marks the words in a paragraph which carry its
meaning — enough that reading only the highlights tells you what it said, without
summarizing. It ships as **21.1 KB** of Brotli-compressed weights and runs entirely in the
browser.

**[Try it](https://highlighter.siddg.com)** · **[Read the full write-up](https://highlighter.siddg.com/article)**

This README is the short version. The [article](https://highlighter.siddg.com/article) is
the long one, and explains every line of the implementation from first principles.

## Why not just summarize?

A summary is *new text*. A model decided what mattered and then wrote fresh sentences
asserting it, and you cannot check those sentences against the source without reading the
source — which is what you were trying to avoid.

Highlighting **selects**. Every word you read is a word the author wrote, in the order they
wrote it. Nothing can be hallucinated because nothing is produced.

```
Hey team, heads up: the staging deploy failed twice this morning because the migration
lock wasn't released. I've rolled back to build 412 and we are NOT shipping today…

→ heads staging deploy failed twice. lock wasn't. rolled build 412. NOT. post 4pm.
```

The summary of that message drops `build 412` and `4pm`. The highlighter keeps `NOT` —
which is the failure mode every frequency-based keyword extractor has: drop the negation
and the residue asserts the opposite of the source.

## The architecture

Deliberately cloned from Vercel Labs' [gpu-lexer](https://gpu-lexer.vercel.app), which does
language-agnostic syntax highlighting with 41,321 parameters in 27.4 KB of WebGPU compute
shaders. There is no repo and no blog post for it, so the design was decoded from the
published npm bundle. Learning that recipe was as much the point as shipping the tool.

```
feature-bag embedding → local 5-token window → learned Blelloch scan → gated MLP head
```

No transformer, no attention, no learned vocabulary.

| Stage | What it does | Params |
|---|---|---|
| **Feature bag** | Each token is reduced to ~19 hand-engineered integers (casing, suffix, length bucket, stopword rank, sentence position, two surface hashes…). Each indexes a row of one shared 737×32 table, and the rows are **summed**. | 23,584 |
| **Local window** | ±2 tokens with per-position weights, plus two pointer tokens, then `tanh`. | 256 |
| **Learned scan** | A Blelloch parallel prefix scan with per-tree-level parameters, run in both directions. Whole-paragraph context in log depth, fully parallel. | 5,760 |
| **Head** | concat[local, global] → sigmoid gate → tanh hidden → one logit. | 9,761 |

**A vocabulary you do not store.** An embedding table for 50k words at d=32 is 1.6M
parameters on its own — 40× the entire budget. Hashing hand-picked features into one shared
table and summing the rows buys most of the benefit for 23,584.

**The scan is not associative.** Its combine operator carries a different parameter at each
tree level, so it describes one *specific* balanced binary tree. A sequential loop builds a
right-leaning tree and gets different numbers; `torch.associative_scan` and the minGRU
`cumsum` trick compute the mathematical prefix, which is also not this. It has to be
hand-written, identically, in every language that runs the model.

The 32 scan channels spread themselves across a **78× range** of memory lengths — some
forget within a token, one keeps influence for ~67.

## The labels

gpu-lexer had a free, deterministic, infinite oracle: Shiki. "Which words matter" has none,
so manufacturing the oracle was the project; the model was the easy half.

12,000 paragraphs were fetched across five registers — Wikipedia, CNN/DailyMail news,
IvyPanda essays, Enron work email, and arXiv abstracts — and 10,000 were labelled by
`mistral-small-latest`, **three independent samples each**. Per-word agreement frequency
becomes a soft target in [0,1], so disagreement is signal rather than noise. The oracle
marks 26.4% of words.

No corpus is committed to this repo; `salience/data/corpus.py` re-fetches it in one command.

## Results

1,497 held-out paragraphs the model never saw:

| | P@15% | P@20% | AP | Spearman |
|---|---|---|---|---|
| random | 0.264 | 0.254 | 0.297 | −0.010 |
| position (topic sentence) | 0.385 | 0.372 | 0.387 | 0.160 |
| tf-idf | 0.508 | 0.495 | 0.531 | 0.480 |
| not-a-stopword | 0.600 | 0.580 | 0.596 | 0.506 |
| **this model** | **0.765** | **0.725** | **0.747** | **0.661** |

On a 65-word paragraph the browser takes **17 ms** against PyTorch's 61 ms on the same
machine. At 39k parameters the framework overhead dwarfs the arithmetic, which is most of
the argument for shipping the model to the client rather than serving it.

## Is 0.661 good?

Not against 1.0, because the labels are not ground truth. Two independent annotations of the
same paragraph correlate at **ρ = 0.704**. Spearman-Brown says averaging three of them gives
a target with reliability 0.877, and a perfect model scores **√0.877 = 0.936** against a
target that noisy. So 0.661 is 71% of the achievable maximum.

`salience/eval/sweep.py` then pulls every lever — capacity, learning rate, window width,
loss design. Nothing moves: ten configurations all land between 0.656 and 0.662. Tripling
the parameters raises the *training* score and leaves validation untouched.

The diagnosis is a frequency breakdown. Bucket validation error by how often each word
appeared in training:

```
times seen in training  val words  mean absolute error
------------------------------------------------------
0 — never seen              4,833                0.341
1–2                         3,992                0.348
3–10                        8,213                0.348
11–50                      18,640                0.350
51–500                     40,614                0.339
500+                       64,837                0.231
```

A word the model has never seen and a word it has seen fifty times get the same error. It is
not learning word-specific importance at all — it has learned the generic signals
(stopwords, casing, numbers, position) and applies them uniformly. That is a knowledge
problem, not a capacity problem, and more of the same data will not fix it.

The likely fix is semantic features: cluster a pretrained embedding space and ship the
cluster id. That reintroduces a vocabulary and trades against the size thesis.

## Notes from building it

- **Gate the carry, not the decay.** Putting the per-level gate on `z` makes every
  down-sweep parameter dead — the final fold reads only the carry, so those gradients are
  exactly zero. Every conformance test passed throughout, because TypeScript and PyTorch
  agreed on the *wrong answer*. Only a gradient check found it.
- **Gates must start open.** `FORGET_BIAS_INIT = 2.0`, `LEVEL_GATE_INIT = 3.0`. At the
  default the carry decays as `0.5^distance` and underflows float32 within ~100 tokens: the
  model starts structurally incapable of long range, with no gradient to learn otherwise.
- **6-bit quantization is free here.** Held-out Spearman moved 0.6218 → 0.6221.
  Quantization-aware training was budgeted and turned out to be unnecessary.
- **The tokenizer was shredding every number.** `1.6` became three tokens. Quantities are
  exactly the facts a highlighter exists to preserve. Caught by reading output, not by a test.

## Running it

```bash
npm install
npm run dev                          # try page at localhost:5173
npm test                             # incl. cross-language conformance
python3 -m pytest salience/tests -q
```

With the PyTorch model as well, so the page can compare runtimes:

```bash
npm run build && python3 serve.py    # http://localhost:8000
```

A **Run on: browser / server** toggle appears whenever `serve.py` is reachable, and reports
the largest per-word disagreement between the two runtimes. It reads about `6e-7`. The
hosted build has no such endpoint, so the toggle stays hidden there.

Rebuilding the pipeline from scratch:

```bash
python3 -m salience.data.corpus --per-domain 2400        # fetch 12k paragraphs
secretspec set MISTRAL_API_KEY                           # once
secretspec run --reason "label corpus" -- \
  python3 -m salience.oracle.run --limit 10000 --samples 3
python3 -m salience.train.run --epochs 40
python3 -m salience.eval.report                          # vs. baselines
python3 -m salience.train.export                         # -> weights/model.json
npm test                                                 # confirm the browser agrees
```

## Layout

```
shared/spec/features.md    normative spec — the source of truth for both implementations
shared/golden/             cross-language fixtures (features, scan, end-to-end scores)
weights/model.json         6-bit weights, one base-63 character per parameter

ts/src/                    tokenizer, features, CPU forward pass, base-63 decoder
salience/features.py       the same tokenizer, in Python
salience/data/             mixed-domain corpus fetcher
salience/oracle/           LLM labeling: prompt, batch runner, review page
salience/train/            scan, model, loss, training loop, quantization, export
salience/eval/             metrics, baselines, ceiling, sweep, meaning preservation
web/                       the try page
```

## Conformance is the whole discipline

Three implementations have to agree: the PyTorch trainer, the TypeScript CPU path, and
eventually WGSL shaders. A written spec is normative and Python generates fixtures that
TypeScript verifies — integers exactly, floats within 1e-6.

**Never edit a fixture to make a test pass.** If conformance fails, one implementation has
drifted from `shared/spec/features.md`.

```bash
python3 scripts/make_golden.py && python3 scripts/make_scan_golden.py
npm test && python3 -m pytest salience/tests -q
```
