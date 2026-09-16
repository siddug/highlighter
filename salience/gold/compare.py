"""Compare human judgement against the oracle and the model.

The question this answers is the one no amount of agreeing-with-Mistral can: **is the
oracle a good proxy for what a person actually considers important?**

Three agreements get measured, and the interesting one is not the model's:

    oracle vs human   how good a teacher we hired
    model  vs human   how good the student is at the real task
    model  vs oracle  how well the student imitates, which is what training optimized

If model-vs-human is close to oracle-vs-human, the student has learned everything the
teacher had to give, and further training against that teacher is wasted effort. If
oracle-vs-human is low, the whole labelling approach needs rethinking rather than scaling.

Run:  python3 -m salience.gold.compare ~/Downloads/gold.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from salience.eval.metrics import precision_recall_at_k, spearman
from salience.features import WORD, extract_features
from salience.train.data import build_example, collate
from salience.train.model import SalienceModel

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labeled" / "labels10k.jsonl"
CKPT = ROOT / "data" / "checkpoints" / "model.pt"
RATE = 0.20


@torch.no_grad()
def model_scores(model: SalienceModel, record: dict) -> np.ndarray:
    batch = collate([build_example(record)])
    probs = torch.sigmoid(model(batch["rows"], batch["ptrs"], batch["lengths"]))[0]
    return probs[batch["is_word"][0]].numpy()


def agreement(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman between two score vectors, plus precision@20% of `a` against `b`."""
    rho = spearman(a, b)
    prec, _ = precision_recall_at_k(a, b, RATE, threshold=0.5)
    return rho, prec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gold", type=Path, help="gold.json exported from the labelling page")
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    args = ap.parse_args()

    marks = json.loads(args.gold.read_text())["marks"]
    marks = {k: v for k, v in marks.items() if v}  # skip paragraphs left empty
    if not marks:
        raise SystemExit("no paragraphs were labelled")

    with args.labels.open(encoding="utf-8") as fh:
        records = {json.loads(line)["id"]: json.loads(line) for line in fh if line.strip()}

    model = SalienceModel()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.eval()

    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    rates = {"human": [], "oracle": []}
    by_domain: dict[str, list[float]] = defaultdict(list)

    for pid, chosen in marks.items():
        record = records.get(pid)
        if record is None:
            continue
        words = [f.text for f in extract_features(record["text"]) if f.cls == WORD]
        n = len(words)

        human = np.zeros(n)
        human[[i for i in chosen if i < n]] = 1.0
        oracle = np.array(record["targets"][:n], dtype=float)
        mdl = model_scores(model, record)
        if len(oracle) != n or len(mdl) != n:
            continue

        pairs["oracle vs human"].append(agreement(oracle, human))
        pairs["model vs human"].append(agreement(mdl, human))
        pairs["model vs oracle"].append(agreement(mdl, oracle))
        by_domain[record["domain"]].append(agreement(mdl, human)[0])
        rates["human"].append(human.mean())
        rates["oracle"].append((oracle >= 0.5).mean())

    n_used = len(pairs["oracle vs human"])
    print(f"{n_used} hand-labelled paragraphs\n")
    print(f"you marked   {np.mean(rates['human']):.1%} of words")
    print(f"oracle marks {np.mean(rates['oracle']):.1%}\n")

    print(f"{'agreement':<20}{'spearman':>10}{'P@20%':>9}")
    print("-" * 39)
    for name in ("oracle vs human", "model vs human", "model vs oracle"):
        vals = [p for p in pairs[name] if not np.isnan(p[0])]
        rho = float(np.mean([v[0] for v in vals]))
        prec = float(np.mean([v[1] for v in vals]))
        print(f"{name:<20}{rho:>10.3f}{prec:>9.3f}")

    print(f"\n{'model vs human, by domain':<28}{'spearman':>10}")
    print("-" * 38)
    for d in sorted(by_domain):
        print(f"{d:<28}{float(np.mean(by_domain[d])):>10.3f}")

    o = float(np.mean([v[0] for v in pairs["oracle vs human"] if not np.isnan(v[0])]))
    m = float(np.mean([v[0] for v in pairs["model vs human"] if not np.isnan(v[0])]))
    print(f"\nThe model reaches {m / o:.0%} of the oracle's own agreement with you.")
    if m >= o - 0.02:
        print("It has learned essentially everything this teacher can convey — more training")
        print("against the same oracle will not help. Improve the labels, not the model.")


if __name__ == "__main__":
    main()
