"""Find out what is actually limiting the score.

The diagnosis before running this: train Spearman 0.686 against validation 0.661 — a gap
of only 0.025, with an oracle ceiling of 0.936. The model cannot fit its own training set,
so it is **underfitting**. More labels would not obviously help; capacity, features or
optimization would.

This sweeps the levers and reports which ones move the number, so the answer to "how do we
get to 0.9" is measured instead of argued.

Run:  python3 -m salience.eval.sweep --epochs 30
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from salience.eval.metrics import evaluate
from salience.train.data import bucket_batches, collate, load_examples, split
from salience.train.loss import LossWeights, compute_loss
from salience.train.model import ModelConfig, SalienceModel, parameter_count

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labeled" / "labels10k.jsonl"
CEILING = 0.936  # see salience/eval/ceiling.py


@dataclass
class Run:
    name: str
    config: ModelConfig
    lr: float = 3e-3
    weights: LossWeights | None = None
    note: str = ""


def build_runs() -> list[Run]:
    return [
        Run("baseline", ModelConfig(), note="the shipped model"),
        # Capacity, one axis at a time.
        Run("wider head", ModelConfig(hid=144), note="head only"),
        Run("d48", ModelConfig(d_model=48), note="wider everything"),
        Run("d64", ModelConfig(d_model=64), note="wider everything"),
        Run("d64 + head", ModelConfig(d_model=64, hid=192, gate=32)),
        # Context, which costs almost nothing in parameters.
        Run("window 9", ModelConfig(window=9), note="+/-4 instead of +/-2"),
        Run("d64 + window 9", ModelConfig(d_model=64, hid=192, gate=32, window=9)),
        # Optimization.
        Run("baseline lr 1e-2", ModelConfig(), lr=1e-2),
        # Is the composite loss earning its keep?
        Run("bce only", ModelConfig(), weights=LossWeights(listnet=0.0, budget=0.0)),
        Run("no budget", ModelConfig(), weights=LossWeights(budget=0.0)),
    ]


@torch.no_grad()
def spearman_of(model: SalienceModel, examples: list, batch_size: int) -> dict[str, float]:
    model.eval()
    pairs = []
    for group in bucket_batches(examples, batch_size, shuffle=False):
        batch = collate(group)
        probs = torch.sigmoid(model(batch["rows"], batch["ptrs"], batch["lengths"])).numpy()
        mask = batch["is_word"].numpy()
        target = batch["target"].numpy()
        for i in range(len(group)):
            sel = mask[i]
            if sel.any():
                pairs.append((probs[i][sel], target[i][sel]))
    model.train()
    return evaluate(pairs)


def train_one(run: Run, train_set: list, val_set: list, epochs: int, batch_size: int, seed: int) -> dict:
    torch.manual_seed(seed)
    model = SalienceModel(run.config)
    weights = run.weights or LossWeights()

    opt = torch.optim.AdamW(model.parameters(), lr=run.lr, weight_decay=1e-4)
    steps = epochs * max(1, len(bucket_batches(train_set, batch_size, shuffle=False)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=run.lr, total_steps=steps, pct_start=0.15)

    best = {"spearman": -1.0}
    start = time.time()

    for epoch in range(1, epochs + 1):
        for group in bucket_batches(train_set, batch_size, shuffle=True, seed=seed + epoch):
            batch = collate(group)
            logits = model(batch["rows"], batch["ptrs"], batch["lengths"])
            loss, _ = compute_loss(logits, batch["target"], batch["is_word"], weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()

        if epoch % 5 == 0 or epoch == epochs:
            metrics = spearman_of(model, val_set, batch_size)
            if metrics["spearman"] > best["spearman"]:
                best = metrics

    train_metrics = spearman_of(model, train_set[:1200], batch_size)
    return {
        "params": parameter_count(model),
        "val": best,
        "train_spearman": train_metrics["spearman"],
        "seconds": time.time() - start,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", help="substring filter on run name")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "checkpoints" / "sweep.json")
    args = ap.parse_args()

    examples = load_examples(args.labels)
    train_set, val_set = split(examples, 0.15, args.seed)
    print(f"{len(train_set)} train / {len(val_set)} val · {args.epochs} epochs each\n")

    runs = [r for r in build_runs() if args.only in (None, "") or args.only in r.name]
    results: dict[str, dict] = {}

    header = f"{'run':<20}{'params':>9}{'train':>8}{'val ρ':>8}{'P@15%':>8}{'AP':>8}{'% ceil':>8}{'min':>6}"
    print(header)
    print("-" * len(header))

    for run in runs:
        out = train_one(run, train_set, val_set, args.epochs, args.batch_size, args.seed)
        results[run.name] = {"note": run.note, **out}
        v = out["val"]
        print(
            f"{run.name:<20}{out['params']:>9,}{out['train_spearman']:>8.3f}"
            f"{v['spearman']:>8.3f}{v['P@15%']:>8.3f}{v['AP']:>8.3f}"
            f"{v['spearman'] / CEILING:>7.0%}{out['seconds'] / 60:>6.1f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nceiling = {CEILING:.3f} (oracle self-agreement)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
