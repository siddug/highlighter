"""Train the salience model.

Run:  python3 -m salience.train.run --epochs 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from salience.eval.metrics import evaluate, format_metrics
from salience.train.data import bucket_batches, collate, load_examples, split
from salience.train.loss import LossWeights, compute_loss
from salience.train.model import SalienceModel, parameter_count

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labeled" / "labels.jsonl"
CKPT_DIR = ROOT / "data" / "checkpoints"


def batch_to(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate_split(model: SalienceModel, examples: list, batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for group in bucket_batches(examples, batch_size, shuffle=False):
        batch = batch_to(collate(group), device)
        logits = model(batch["rows"], batch["ptrs"], batch["lengths"])
        probs = torch.sigmoid(logits).cpu().numpy()
        mask = batch["is_word"].cpu().numpy()
        targets = batch["target"].cpu().numpy()
        for i in range(len(group)):
            sel = mask[i]
            if sel.any():
                pairs.append((probs[i][sel], targets[i][sel]))
    model.train()
    return evaluate(pairs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=CKPT_DIR / "model.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if not args.labels.exists():
        raise SystemExit(f"{args.labels} not found — run: python3 -m salience.oracle.run")

    print(f"loading {args.labels}")
    examples = load_examples(args.labels)
    train_set, val_set = split(examples, args.val_fraction, args.seed)
    print(f"  {len(train_set)} train / {len(val_set)} val paragraphs")

    model = SalienceModel().to(device)
    print(f"  {parameter_count(model):,} parameters")

    weights = LossWeights()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * max(1, len(bucket_batches(train_set, args.batch_size, shuffle=False)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)

    best = -1.0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        batches = bucket_batches(train_set, args.batch_size, shuffle=True, seed=args.seed + epoch)
        totals: dict[str, float] = {}

        for group in batches:
            batch = batch_to(collate(group), device)
            logits = model(batch["rows"], batch["ptrs"], batch["lengths"])
            loss, parts = compute_loss(logits, batch["target"], batch["is_word"], weights)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()

            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v

        metrics = evaluate_split(model, val_set, args.batch_size, device)
        avg = {k: v / len(batches) for k, v in totals.items()}
        score = metrics.get("spearman", 0.0)

        flag = ""
        if score > best:
            best = score
            # Plain strings only: torch.load defaults to weights_only=True, which rejects
            # pickled Path objects.
            torch.save(
                {"model": model.state_dict(), "args": {k: str(v) for k, v in vars(args).items()}},
                args.out,
            )
            flag = "  *"

        print(
            f"epoch {epoch:3d}  loss {avg['loss']:.4f} "
            f"(bce {avg['bce']:.3f} list {avg['listnet']:.3f} budget {avg['budget']:.4f})  "
            f"| val {format_metrics(metrics)}{flag}"
        )

    print(f"\nbest val spearman {best:.4f} -> {args.out}  ({time.time() - start:.0f}s)")
    (CKPT_DIR / "last_metrics.json").write_text(
        json.dumps(evaluate_split(model, val_set, args.batch_size, device), indent=2)
    )


if __name__ == "__main__":
    main()
