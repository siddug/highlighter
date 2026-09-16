"""Baselines the model has to beat to justify existing.

A 39k-parameter network that cannot outscore "highlight the non-stopwords" is not worth
shipping, and it would be easy to never find out. These are deliberately cheap and
deliberately hard to beat.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

from salience.features import WORD, ascii_lower, extract_features

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "raw" / "paragraphs.jsonl"


def words_of(text: str) -> list[str]:
    return [ascii_lower(f.text) for f in extract_features(text) if f.cls == WORD]


def build_idf(corpus: Path = CORPUS) -> dict[str, float]:
    """Inverse document frequency over the full corpus, for the TF-IDF baseline."""
    df: Counter[str] = Counter()
    n = 0
    with corpus.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            n += 1
            df.update(set(words_of(json.loads(line)["text"])))
    return {w: math.log(n / (1 + c)) for w, c in df.items()}


def score_random(words: list[str], _idf: dict[str, float], seed: int = 0) -> np.ndarray:
    rng = random.Random(seed)
    return np.array([rng.random() for _ in words])


def score_not_stopword(words: list[str], _idf: dict[str, float]) -> np.ndarray:
    """Mark everything that is not a function word. The bar to clear."""
    from salience.features import _stopword_bucket  # noqa: PLC0415

    return np.array([0.0 if _stopword_bucket(w) > 0 else 1.0 for w in words])


def score_tfidf(words: list[str], idf: dict[str, float]) -> np.ndarray:
    tf = Counter(words)
    default = max(idf.values()) if idf else 1.0
    return np.array([tf[w] * idf.get(w, default) for w in words])


def score_position(words: list[str], _idf: dict[str, float]) -> np.ndarray:
    """Earlier is more important — the topic-sentence heuristic."""
    n = max(1, len(words))
    return np.array([1.0 - i / n for i in range(len(words))])


BASELINES = {
    "random": score_random,
    "not-stopword": score_not_stopword,
    "tf-idf": score_tfidf,
    "position": score_position,
}
