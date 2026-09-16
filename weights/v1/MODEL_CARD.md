# v1-small — frozen 2026-09-16

The shipped model, preserved before any relaxation of the size constraint.

| | |
|---|---|
| parameters | 39,361 |
| spec version | v2 |
| wire size | 39.9 KB raw · 21.1 KB brotli |
| sha256 (first 16) | `1728e159348c4ed3` |

## Architecture
d_model 32 · feature-bag embedding (737x32) · local window +/-2 with 2 pointers ·
bidirectional learned Blelloch scan · gated MLP head (gate 16, hidden 72) · 6-bit weights.

## Training
10,000 LLM-labelled paragraphs (mistral-small, 3 samples each, prompt p2), 40 epochs,
AdamW + OneCycle at 3e-3, composite loss (soft BCE + ListNet + budget).

## Held-out results (1,497 paragraphs)

| | P@15% | P@20% | AP | Spearman |
|---|---|---|---|---|
| not-a-stopword baseline | 0.600 | 0.580 | 0.596 | 0.506 |
| **v1-small** | **0.765** | **0.725** | **0.747** | **0.661** |
| oracle ceiling | — | — | — | 0.936 |

## Reproducing
`scores.jsonl` here pins the exact per-word outputs on 40 held-out paragraphs.
ts/src/scores.golden.test.ts checks the browser against it to within 2e-4.

## Known properties
- Underfits: train Spearman 0.686 vs val 0.661, both far below the 0.936 ceiling.
- Capacity-insensitive: 3x the parameters moves validation by 0.000.
- Learns no word-specific importance: mean error is flat at 0.34 for every word
  seen fewer than 500 times in training.
