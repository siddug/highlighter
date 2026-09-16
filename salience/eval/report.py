"""Compare the trained model against the baselines on the held-out split.

A 39k-parameter network that cannot beat "highlight the non-stopwords" has not earned its
place. This is the script that says so out loud.

Run:  python3 -m salience.eval.report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from salience.eval.baselines import BASELINES, build_idf, words_of
from salience.eval.metrics import evaluate
from salience.train.data import bucket_batches, collate, load_examples, split
from salience.train.model import SalienceModel

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labeled" / "labels.jsonl"
CKPT = ROOT / "data" / "checkpoints" / "model.pt"

COLUMNS = ["P@15%", "P@20%", "P@25%", "R@20%", "AP", "spearman"]


@torch.no_grad()
def model_pairs(model: SalienceModel, examples: list, batch_size: int = 32) -> list:
    model.eval()
    pairs = []
    for group in bucket_batches(examples, batch_size, shuffle=False):
        batch = collate(group)
        probs = torch.sigmoid(model(batch["rows"], batch["ptrs"], batch["lengths"])).numpy()
        mask = batch["is_word"].numpy()
        targets = batch["target"].numpy()
        for i in range(len(group)):
            sel = mask[i]
            if sel.any():
                pairs.append((probs[i][sel], targets[i][sel]))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    examples = load_examples(args.labels)
    _, val = split(examples, args.val_fraction, args.seed)

    with args.labels.open(encoding="utf-8") as fh:
        records = {}
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                records[rec["id"]] = rec

    # Baselines must score exactly the paragraphs the model is evaluated on.
    val_texts = [records[ex.pid] for ex in val]
    print(f"validation: {len(val)} paragraphs\n")

    rows: list[tuple[str, dict[str, float]]] = []

    print("computing idf over the corpus...", end=" ", flush=True)
    idf = build_idf()
    print(f"{len(idf):,} terms")

    for name, fn in BASELINES.items():
        pairs = []
        for rec in val_texts:
            words = words_of(rec["text"])
            targets = np.array(rec["targets"][: len(words)], dtype=float)
            if len(words) != len(targets):
                continue
            pairs.append((fn(words, idf).astype(float), targets))
        rows.append((name, evaluate(pairs)))

    model = SalienceModel()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    rows.append(("model (39k)", evaluate(model_pairs(model, val))))

    width = max(len(n) for n, _ in rows) + 2
    print(f"\n{'':<{width}}" + "".join(f"{c:>10}" for c in COLUMNS))
    for name, m in rows:
        print(f"{name:<{width}}" + "".join(f"{m.get(c, float('nan')):>10.3f}" for c in COLUMNS))

    best_baseline = max(
        (m.get("spearman", 0.0) for n, m in rows if n != "model (39k)"), default=0.0
    )
    model_score = rows[-1][1].get("spearman", 0.0)
    verdict = "beats" if model_score > best_baseline else "DOES NOT BEAT"
    print(f"\nmodel {verdict} the best baseline on spearman "
          f"({model_score:.3f} vs {best_baseline:.3f})")


if __name__ == "__main__":
    main()
