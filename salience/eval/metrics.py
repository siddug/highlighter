"""Evaluation metrics for word salience.

Token-level F1 is the obvious metric and the wrong one. A reader consumes the **top k**
words of a paragraph, so the operating point is a rank cutoff, and what matters is whether
the right words are at the top — not whether the probabilities are calibrated.

Implemented with numpy only; scipy and sklearn are not worth the dependency for this.
"""

from __future__ import annotations

import numpy as np


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, matching scipy.stats.rankdata."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)

    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def precision_recall_at_k(
    scores: np.ndarray, targets: np.ndarray, frac: float, threshold: float = 0.5
) -> tuple[float, float]:
    """Precision and recall when taking the top `frac` of a paragraph's words."""
    n = len(scores)
    if n == 0:
        return 0.0, 0.0
    k = max(1, round(frac * n))
    top = np.argsort(-scores, kind="mergesort")[:k]
    relevant = targets >= threshold
    hits = relevant[top].sum()
    return hits / k, (hits / relevant.sum()) if relevant.any() else 0.0


def average_precision(scores: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    """Area under the precision-recall curve. Threshold-free and imbalance-robust."""
    relevant = targets >= threshold
    total = relevant.sum()
    if total == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    hits = 0
    acc = 0.0
    for rank, idx in enumerate(order, start=1):
        if relevant[idx]:
            hits += 1
            acc += hits / rank
    return acc / total


def spearman(scores: np.ndarray, targets: np.ndarray) -> float:
    """Rank correlation. The metric closest to 'is the ordering right'."""
    if len(scores) < 2 or np.all(targets == targets[0]) or np.all(scores == scores[0]):
        return float("nan")
    a, b = _rankdata(scores), _rankdata(targets)
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def evaluate(paragraphs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Aggregate metrics over (scores, targets) pairs, one pair per paragraph."""
    out: dict[str, list[float]] = {}

    def add(key: str, value: float) -> None:
        if not np.isnan(value):
            out.setdefault(key, []).append(value)

    for scores, targets in paragraphs:
        if len(scores) == 0:
            continue
        for frac in (0.15, 0.20, 0.25):
            p, r = precision_recall_at_k(scores, targets, frac)
            add(f"P@{int(frac * 100)}%", p)
            add(f"R@{int(frac * 100)}%", r)
        add("AP", average_precision(scores, targets))
        add("spearman", spearman(scores, targets))

    return {k: float(np.mean(v)) for k, v in out.items()}


def format_metrics(metrics: dict[str, float]) -> str:
    order = ["P@15%", "P@20%", "P@25%", "R@20%", "AP", "spearman"]
    keys = [k for k in order if k in metrics] + [k for k in metrics if k not in order]
    return "  ".join(f"{k} {metrics[k]:.3f}" for k in keys)
