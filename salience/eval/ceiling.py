"""How good could this model possibly get?

Chasing a target score is meaningless without knowing the maximum. Our labels are not
ground truth — they are three samples from an LLM that disagrees with itself. If two
independent annotations of the same paragraph only correlate at 0.7, then a *perfect*
model cannot score 1.0 against their average, because the average still contains noise.

This is standard psychometrics:

1. **Split-half reliability.** Correlate annotation A with annotation B, paragraph by
   paragraph. That is how much the oracle agrees with itself — the signal-to-noise ratio
   of a single annotation.
2. **Spearman-Brown.** Averaging k annotations reduces noise predictably:
   `r_k = k·r / (1 + (k-1)·r)`. This gives the reliability of the k-sample target we
   actually train against.
3. **The ceiling.** A model predicting the *true* underlying score would correlate with
   our noisy target at `sqrt(r_k)`. That is the highest number any model can achieve here,
   and it is the number our 0.661 should be compared against.

Run:  python3 -m salience.eval.ceiling
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from salience.eval.metrics import precision_recall_at_k, spearman

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labeled" / "labels10k.jsonl"


def samples_to_vector(sample: list[int], word_count: int) -> np.ndarray:
    v = np.zeros(word_count)
    for i in sample:
        if 0 <= i < word_count:
            v[i] = 1.0
    return v


def spearman_brown(r: float, k: int) -> float:
    """Reliability of a k-sample average, given single-sample reliability r."""
    if r <= 0:
        return 0.0
    return (k * r) / (1 + (k - 1) * r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=LABELS)
    args = ap.parse_args()

    with args.labels.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    pair_rho: list[float] = []
    pair_p20: list[float] = []
    rates: list[float] = []
    sample_counts: set[int] = set()

    for rec in records:
        n = rec["word_count"]
        samples = rec["samples"]
        sample_counts.add(len(samples))
        if len(samples) < 2 or n < 4:
            continue

        vectors = [samples_to_vector(s, n) for s in samples]
        for a, b in itertools.combinations(range(len(vectors)), 2):
            rho = spearman(vectors[a], vectors[b])
            if not math.isnan(rho):
                pair_rho.append(rho)
            # How often does one annotation's top-20% land on the other's marks?
            p, _ = precision_recall_at_k(vectors[a], vectors[b], 0.20, threshold=0.5)
            pair_p20.append(p)
        for v in vectors:
            rates.append(v.mean())

    k = max(sample_counts)
    r1 = float(np.mean(pair_rho))
    rk = spearman_brown(r1, k)
    ceiling_rho = math.sqrt(rk)

    print(f"labels            {args.labels.name} · {len(records):,} paragraphs · {k} samples each")
    print(f"annotation pairs  {len(pair_rho):,}")
    print()
    print("ORACLE SELF-AGREEMENT (the label noise floor)")
    print(f"  single-pair Spearman        r1 = {r1:.3f}")
    print(f"  P@20% of one sample vs another   {np.mean(pair_p20):.3f}")
    print(f"  mean marks per annotation        {np.mean(rates):.1%} of words")
    print()
    print("IMPLIED CEILING")
    print(f"  reliability of the {k}-sample target   r{k} = {rk:.3f}   (Spearman-Brown)")
    print(f"  max Spearman any model can reach      = {ceiling_rho:.3f}   (sqrt of that)")
    print()

    for extra in (5, 9, 15):
        r = spearman_brown(r1, extra)
        print(f"  with {extra:>2} samples per paragraph:  ceiling = {math.sqrt(r):.3f}")

    print()
    print("Read this as: a model scoring the ceiling is not 'perfect', it is 'as close as")
    print("these labels allow'. Beyond that, the remaining error is the oracle's own")
    print("inconsistency, and the only way past it is better or more numerous labels.")


if __name__ == "__main__":
    main()
