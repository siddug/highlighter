"""Turn labeled paragraphs into batched tensors.

The one non-obvious constraint: **batches must be bucketed by next_pow2(token_count)**.

The scan's level parameters are indexed by tree depth, and tree depth is log2 of the padded
length. A 90-token paragraph padded to 128 and the same paragraph padded to 256 therefore
produce different numbers. If we batched by raw length, a paragraph's prediction would
depend on which other paragraphs happened to share its batch — and training would not match
inference. Bucketing makes each item's padded length equal to its own next_pow2, always.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from salience.features import WORD, extract_features
from salience.train.model import MAX_ROWS, PAD_ROW
from salience.train.scan import next_pow2


@dataclass
class Example:
    rows: torch.Tensor  # (T, MAX_ROWS) int64
    ptrs: torch.Tensor  # (T, 2) int64, -1 when absent
    is_word: torch.Tensor  # (T,) bool
    target: torch.Tensor  # (T,) float32, soft label at word positions, 0 elsewhere
    length: int
    domain: str
    pid: str


def build_example(record: dict) -> Example | None:
    feats = extract_features(record["text"])
    if not feats:
        return None

    targets = record["targets"]
    t = len(feats)

    rows = torch.full((t, MAX_ROWS), PAD_ROW, dtype=torch.long)
    ptrs = torch.empty((t, 2), dtype=torch.long)
    is_word = torch.zeros(t, dtype=torch.bool)
    target = torch.zeros(t, dtype=torch.float32)

    word_index = 0
    for i, f in enumerate(feats):
        if len(f.rows) > MAX_ROWS:
            raise ValueError(f"token {f.text!r} selected {len(f.rows)} rows, MAX_ROWS={MAX_ROWS}")
        rows[i, : len(f.rows)] = torch.tensor(f.rows, dtype=torch.long)
        ptrs[i, 0] = f.sentence_first_word
        ptrs[i, 1] = f.prev_sentence_last_word
        if f.cls == WORD:
            is_word[i] = True
            if word_index < len(targets):
                target[i] = targets[word_index]
            word_index += 1

    # A mismatch means the labels were produced under a different spec version.
    if word_index != len(targets):
        return None

    return Example(rows, ptrs, is_word, target, t, record.get("domain", "unknown"), record["id"])


def load_examples(path: Path) -> list[Example]:
    with path.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    out = [ex for ex in (build_example(r) for r in records) if ex is not None]
    if len(out) < len(records):
        print(f"  skipped {len(records) - len(out)} records whose labels did not align")
    return out


def collate(batch: list[Example]) -> dict[str, torch.Tensor]:
    """Pad to the bucket's power-of-two length. All items must share next_pow2(length)."""
    padded = next_pow2(max(ex.length for ex in batch))
    assert all(next_pow2(ex.length) == padded for ex in batch), "batch crosses a bucket boundary"

    n = len(batch)
    rows = torch.full((n, padded, MAX_ROWS), PAD_ROW, dtype=torch.long)
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

    return {
        "rows": rows,
        "ptrs": ptrs,
        "is_word": is_word,
        "target": target,
        "lengths": lengths,
    }


def bucket_batches(
    examples: list[Example], batch_size: int, shuffle: bool, seed: int = 0
) -> list[list[Example]]:
    """Group into batches that never cross a next_pow2 boundary."""
    buckets: dict[int, list[Example]] = {}
    for ex in examples:
        buckets.setdefault(next_pow2(ex.length), []).append(ex)

    rng = random.Random(seed)
    batches: list[list[Example]] = []
    for _, group in sorted(buckets.items()):
        if shuffle:
            rng.shuffle(group)
        batches.extend(group[i : i + batch_size] for i in range(0, len(group), batch_size))

    if shuffle:
        rng.shuffle(batches)
    return batches


def split(examples: list[Example], val_fraction: float, seed: int = 0) -> tuple[list, list]:
    """Deterministic train/val split, stratified by domain so every domain is evaluated."""
    by_domain: dict[str, list[Example]] = {}
    for ex in examples:
        by_domain.setdefault(ex.domain, []).append(ex)

    rng = random.Random(seed)
    train: list[Example] = []
    val: list[Example] = []
    for domain in sorted(by_domain):
        group = by_domain[domain][:]
        rng.shuffle(group)
        cut = max(1, int(len(group) * val_fraction))
        val.extend(group[:cut])
        train.extend(group[cut:])
    return train, val
