"""Does handing the model word meanings actually help?

The v1 diagnosis: mean error is flat at 0.34 for every word seen fewer than 500 times in
training, capacity changes nothing, and corpus IDF buys only +0.005. The model learns
generic signals and nothing word-specific.

This tests the structural fix — give it pretrained meaning instead of asking it to learn
meaning — in two forms, because they cost very different numbers of bytes:

    clusters K    one extra embedding row per word, chosen by k-means group.
                  Adds K x 32 parameters and a word -> id table (~57 KB).
                  Slots into the feature bag without changing the architecture.

    projected d   the word's PCA'd embedding added straight into the token vector,
                  through a learned d x 32 projection. Adds d x 32 parameters and a
                  word -> vector table (~900 KB at 6 bits). Strictly more information.

Words outside the 29k vocabulary fall back to a dedicated unknown row (clusters) or a zero
vector (projected), so behaviour on unseen text degrades to v1 rather than breaking.

Run:  python3 -m salience.semantic.train_v2 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from salience.eval.metrics import evaluate
from salience.features import ROW_TOTAL, WORD, ascii_lower, extract_features
from salience.train.data import bucket_batches, split
from salience.train.loss import LossWeights, compute_loss
from salience.train.model import MAX_ROWS, ModelConfig, SalienceModel
from salience.train.scan import next_pow2

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labeled" / "labels10k.jsonl"
SEM_DIR = ROOT / "data" / "semantic"
CEILING = 0.936


@dataclass
class Variant:
    name: str
    clusters: int = 0  # 0 = off
    projected: int = 0  # 0 = off


@dataclass
class Example:
    rows: torch.Tensor
    ptrs: torch.Tensor
    sem: torch.Tensor  # (T, d) projected vectors, or (T, 0)
    is_word: torch.Tensor
    target: torch.Tensor
    length: int
    domain: str
    pid: str


class SemanticModel(SalienceModel):
    """v1 plus optional pretrained word meaning."""

    def __init__(self, config: ModelConfig, variant: Variant) -> None:
        super().__init__(config)
        d = config.d_model
        self.variant = variant

        if variant.clusters:
            # One row per cluster, plus one for out-of-vocabulary words.
            self.n_rows = ROW_TOTAL + variant.clusters + 1
            self.pad_row = self.n_rows
            self.emb = nn.Embedding(self.n_rows + 1, d, padding_idx=self.pad_row)
            nn.init.normal_(self.emb.weight, std=0.1)
            with torch.no_grad():
                self.emb.weight[self.pad_row].zero_()
        else:
            self.n_rows = ROW_TOTAL
            self.pad_row = ROW_TOTAL

        self.sem_proj = nn.Linear(variant.projected, d, bias=False) if variant.projected else None
        if self.sem_proj is not None:
            nn.init.normal_(self.sem_proj.weight, std=0.05)

    def embed_with_semantics(self, rows: torch.Tensor, sem: torch.Tensor) -> torch.Tensor:
        out = self.emb(rows).sum(dim=2)
        if self.sem_proj is not None:
            out = out + self.sem_proj(sem)
        return out

    def forward(self, rows, ptrs, lengths, sem=None):  # type: ignore[override]
        from salience.train.model import _gather_seq, _reverse_index  # noqa: PLC0415

        emb = self.embed_with_semantics(rows, sem) if sem is not None else self.emb(rows).sum(2)
        local = self.local(emb, ptrs)
        gf = self.fwd(local, lengths)
        rev = _reverse_index(local.shape[1], lengths)
        gb = _gather_seq(self.bwd(_gather_seq(local, rev), lengths), rev)
        x = torch.cat([local, gf, gb], dim=-1)
        gate = torch.sigmoid(self.head_gate(x))
        hidden = torch.tanh(self.head_hid(torch.cat([x, gate], dim=-1)))
        return self.head_out(hidden).squeeze(-1)

    def shipped_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters()) - self.config.d_model


class Lookup:
    """word -> cluster id and word -> projected vector, with graceful unknowns."""

    def __init__(self, sem_dir: Path, variant: Variant) -> None:
        vocab = json.loads((sem_dir / "vocab.json").read_text())
        self.index = {w: i for i, w in enumerate(vocab)}
        self.clusters = (
            np.load(sem_dir / f"clusters{variant.clusters}.npy") if variant.clusters else None
        )
        self.projected = (
            np.load(sem_dir / f"projected{variant.projected}.npy") if variant.projected else None
        )
        self.unknown_cluster = variant.clusters  # the extra row

    def cluster_row(self, word: str) -> int:
        i = self.index.get(word)
        base = ROW_TOTAL
        return base + (self.unknown_cluster if i is None else int(self.clusters[i]))

    def vector(self, word: str, dims: int) -> np.ndarray:
        i = self.index.get(word)
        return np.zeros(dims, dtype=np.float32) if i is None else self.projected[i]


def build_example(record: dict, variant: Variant, look: Lookup | None, pad_row: int):
    feats = extract_features(record["text"])
    if not feats:
        return None
    targets = record["targets"]
    t = len(feats)
    width = MAX_ROWS + (1 if variant.clusters else 0)

    rows = torch.full((t, width), pad_row, dtype=torch.long)
    ptrs = torch.empty((t, 2), dtype=torch.long)
    sem = torch.zeros((t, variant.projected), dtype=torch.float32)
    is_word = torch.zeros(t, dtype=torch.bool)
    target = torch.zeros(t, dtype=torch.float32)

    w_i = 0
    for i, f in enumerate(feats):
        rows[i, : len(f.rows)] = torch.tensor(f.rows, dtype=torch.long)
        if f.cls == WORD:
            lower = ascii_lower(f.text)
            if variant.clusters and look:
                rows[i, len(f.rows)] = look.cluster_row(lower)
            if variant.projected and look:
                sem[i] = torch.from_numpy(look.vector(lower, variant.projected))
            is_word[i] = True
            if w_i < len(targets):
                target[i] = targets[w_i]
            w_i += 1
        ptrs[i, 0] = f.sentence_first_word
        ptrs[i, 1] = f.prev_sentence_last_word

    if w_i != len(targets):
        return None
    return Example(rows, ptrs, sem, is_word, target, t, record.get("domain", "?"), record["id"])


def collate(batch: list[Example], pad_row: int) -> dict[str, torch.Tensor]:
    padded = next_pow2(max(ex.length for ex in batch))
    n, width, sdim = len(batch), batch[0].rows.shape[1], batch[0].sem.shape[1]

    out = {
        "rows": torch.full((n, padded, width), pad_row, dtype=torch.long),
        "ptrs": torch.full((n, padded, 2), -1, dtype=torch.long),
        "sem": torch.zeros((n, padded, sdim), dtype=torch.float32),
        "is_word": torch.zeros((n, padded), dtype=torch.bool),
        "target": torch.zeros((n, padded), dtype=torch.float32),
        "lengths": torch.tensor([ex.length for ex in batch], dtype=torch.long),
    }
    for i, ex in enumerate(batch):
        t = ex.length
        out["rows"][i, :t] = ex.rows
        out["ptrs"][i, :t] = ex.ptrs
        out["sem"][i, :t] = ex.sem
        out["is_word"][i, :t] = ex.is_word
        out["target"][i, :t] = ex.target
    return out


@torch.no_grad()
def score(model, examples, pad_row, use_sem):
    model.eval()
    pairs = []
    for g in bucket_batches(examples, 32, shuffle=False):
        b = collate(g, pad_row)
        logits = model(b["rows"], b["ptrs"], b["lengths"], b["sem"] if use_sem else None)
        probs = torch.sigmoid(logits).numpy()
        mask, tgt = b["is_word"].numpy(), b["target"].numpy()
        for i in range(len(g)):
            sel = mask[i]
            if sel.any():
                pairs.append((probs[i][sel], tgt[i][sel]))
    model.train()
    return evaluate(pairs)


def run(variant: Variant, records: list[dict], epochs: int, seed: int) -> dict:
    look = Lookup(SEM_DIR, variant) if (variant.clusters or variant.projected) else None
    torch.manual_seed(seed)
    model = SemanticModel(ModelConfig(), variant)
    examples = [
        ex for ex in (build_example(r, variant, look, model.pad_row) for r in records) if ex
    ]
    train_set, val_set = split(examples, 0.15, seed)
    use_sem = bool(variant.projected)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    steps = epochs * max(1, len(bucket_batches(train_set, 32, shuffle=False)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.15)
    weights = LossWeights()

    best = {"spearman": -1.0}
    start = time.time()
    for epoch in range(1, epochs + 1):
        for g in bucket_batches(train_set, 32, shuffle=True, seed=seed + epoch):
            b = collate(g, model.pad_row)
            logits = model(b["rows"], b["ptrs"], b["lengths"], b["sem"] if use_sem else None)
            loss, _ = compute_loss(logits, b["target"], b["is_word"], weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()
        if epoch % 5 == 0 or epoch == epochs:
            m = score(model, val_set, model.pad_row, use_sem)
            if m["spearman"] > best["spearman"]:
                best = m
                torch.save({"model": model.state_dict(), "variant": vars(variant)},
                           ROOT / "data" / "checkpoints" / f"v2-{variant.name.replace(' ', '-')}.pt")

    train_m = score(model, train_set[:1200], model.pad_row, use_sem)
    return {
        "params": model.shipped_parameters(),
        "train": train_m["spearman"],
        "val": best,
        "minutes": (time.time() - start) / 60,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with args.labels.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    variants = [
        Variant("v1 baseline"),
        Variant("clusters 256", clusters=256),
        Variant("clusters 1024", clusters=1024),
        Variant("projected 32", projected=32),
    ]

    header = f"{'variant':<18}{'params':>9}{'train':>8}{'val rho':>9}{'P@15%':>8}{'AP':>8}{'% ceil':>8}{'min':>6}"
    print(f"{len(records):,} paragraphs · {args.epochs} epochs each\n{header}\n{'-' * len(header)}")
    results = {}
    for v in variants:
        r = run(v, records, args.epochs, args.seed)
        results[v.name] = r
        print(
            f"{v.name:<18}{r['params']:>9,}{r['train']:>8.3f}{r['val']['spearman']:>9.3f}"
            f"{r['val']['P@15%']:>8.3f}{r['val']['AP']:>8.3f}"
            f"{r['val']['spearman'] / CEILING:>7.0%}{r['minutes']:>6.1f}"
        )

    (ROOT / "data" / "checkpoints" / "v2_sweep.json").write_text(json.dumps(results, indent=2))
    print(f"\nceiling {CEILING:.3f} · v1 shipped 0.661")


if __name__ == "__main__":
    main()
