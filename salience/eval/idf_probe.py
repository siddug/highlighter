"""Is corpus-wide word frequency the missing feature?

The capacity sweep showed that tripling parameters raises the training score but leaves
validation pinned at 0.66. That is an information limit, not a capacity limit — the model
cannot see something it needs.

One candidate stands out. The feature set includes *within-document* frequency and a
stopword bucket covering the 200 most common words, but nothing about how common a word is
across English generally. Meanwhile the tf-idf baseline — which is nothing *but* that
signal — scores 0.480 Spearman on its own. That is real information the model is blind to.

This bolts an IDF bucket onto the feature set and retrains. Python only: if it helps, the
spec earns a proper v3 and the TypeScript side follows. If it does not, we learned that
cheaply.

Run:  python3 -m salience.eval.idf_probe --epochs 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from salience.eval.baselines import build_idf
from salience.eval.metrics import evaluate
from salience.features import ROW_TOTAL, WORD, ascii_lower, extract_features
from salience.train.data import Example, bucket_batches, split
from salience.train.loss import LossWeights, compute_loss
from salience.train.model import MAX_ROWS, PAD_ROW, ModelConfig, SalienceModel, parameter_count

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labeled" / "labels10k.jsonl"
CEILING = 0.936

IDF_ROWS = 16
IDF_BASE = ROW_TOTAL
PAD = ROW_TOTAL + IDF_ROWS


def idf_bucket(value: float, lo: float, hi: float) -> int:
    if hi <= lo:
        return 0
    frac = (value - lo) / (hi - lo)
    return max(0, min(IDF_ROWS - 1, int(frac * IDF_ROWS)))


def build_example(record: dict, idf: dict[str, float], lo: float, hi: float, use_idf: bool):
    feats = extract_features(record["text"])
    if not feats:
        return None
    targets = record["targets"]
    t = len(feats)

    rows = torch.full((t, MAX_ROWS + 1), PAD, dtype=torch.long)
    ptrs = torch.empty((t, 2), dtype=torch.long)
    is_word = torch.zeros(t, dtype=torch.bool)
    target = torch.zeros(t, dtype=torch.float32)

    word_index = 0
    for i, f in enumerate(feats):
        rows[i, : len(f.rows)] = torch.tensor(f.rows, dtype=torch.long)
        if use_idf and f.cls == WORD:
            rows[i, len(f.rows)] = IDF_BASE + idf_bucket(idf.get(ascii_lower(f.text), hi), lo, hi)
        ptrs[i, 0] = f.sentence_first_word
        ptrs[i, 1] = f.prev_sentence_last_word
        if f.cls == WORD:
            is_word[i] = True
            if word_index < len(targets):
                target[i] = targets[word_index]
            word_index += 1

    if word_index != len(targets):
        return None
    return Example(rows, ptrs, is_word, target, t, record.get("domain", "?"), record["id"])


class IdfModel(SalienceModel):
    """Same architecture, embedding table widened by IDF_ROWS."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        d = config.d_model
        self.emb = torch.nn.Embedding(PAD + 1, d, padding_idx=PAD)
        torch.nn.init.normal_(self.emb.weight, std=0.1)
        with torch.no_grad():
            self.emb.weight[PAD].zero_()


def collate(batch: list[Example]) -> dict[str, torch.Tensor]:
    """Local copy of train.data.collate — this probe carries one extra row per token."""
    from salience.train.scan import next_pow2

    padded = next_pow2(max(ex.length for ex in batch))
    n = len(batch)
    width = batch[0].rows.shape[1]

    rows = torch.full((n, padded, width), PAD, dtype=torch.long)
    ptrs = torch.full((n, padded, 2), -1, dtype=torch.long)
    is_word = torch.zeros((n, padded), dtype=torch.bool)
    target = torch.zeros((n, padded), dtype=torch.float32)
    lengths = torch.tensor([ex.length for ex in batch], dtype=torch.long)

    for i, ex in enumerate(batch):
        t = ex.length
        rows[i, :t] = ex.rows
        ptrs[i, :t] = ex.ptrs
        is_word[i, :t] = ex.is_word
        target[i, :t] = ex.target

    return {"rows": rows, "ptrs": ptrs, "is_word": is_word, "target": target, "lengths": lengths}


@torch.no_grad()
def score_split(model, examples, batch_size: int = 32) -> dict[str, float]:
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


def run(tag: str, train_set: list, val_set: list, epochs: int, seed: int) -> None:
    torch.manual_seed(seed)
    model = IdfModel(ModelConfig())
    weights = LossWeights()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    steps = epochs * max(1, len(bucket_batches(train_set, 32, shuffle=False)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.15)

    best = {"spearman": -1.0}
    start = time.time()
    for epoch in range(1, epochs + 1):
        for group in bucket_batches(train_set, 32, shuffle=True, seed=seed + epoch):
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
            m = score_split(model, val_set)
            if m["spearman"] > best["spearman"]:
                best = m

    train_m = score_split(model, train_set[:1200])
    print(
        f"{tag:<22}{parameter_count(model):>9,}{train_m['spearman']:>8.3f}"
        f"{best['spearman']:>8.3f}{best['P@15%']:>8.3f}{best['AP']:>8.3f}"
        f"{best['spearman'] / CEILING:>8.0%}{(time.time() - start) / 60:>6.1f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("computing corpus idf...", end=" ", flush=True)
    idf = build_idf()
    values = list(idf.values())
    lo, hi = min(values), max(values)
    print(f"{len(idf):,} terms, range {lo:.2f}-{hi:.2f}")

    with args.labels.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    header = f"{'run':<22}{'params':>9}{'train':>8}{'val rho':>8}{'P@15%':>8}{'AP':>8}{'% ceil':>8}{'min':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    for use_idf in (False, True):
        examples = [
            ex for ex in (build_example(r, idf, lo, hi, use_idf) for r in records) if ex is not None
        ]
        train_set, val_set = split(examples, 0.15, args.seed)
        run("with idf" if use_idf else "control (no idf)", train_set, val_set, args.epochs, args.seed)

    print(f"\nceiling = {CEILING:.3f}")


if __name__ == "__main__":
    main()
