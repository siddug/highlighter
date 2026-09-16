"""Give the model word meanings it cannot learn on its own.

The diagnosis from salience/eval/sweep.py: mean error is flat at 0.34 for every word
seen fewer than 500 times in training. The model learns generic signals — stopwords,
casing, numbers, position — and nothing word-specific. Tripling the parameters does not
help because there is no per-word signal being stored.

So stop asking it to learn meanings from 10,000 paragraphs, and hand them over instead.
This module embeds the corpus vocabulary with `mistral-embed` and caches the result.
salience/semantic/reduce.py turns those 1024-dim vectors into something small enough to
ship.

Mean-centring is not optional
-----------------------------
Sentence-embedding models handle bare words badly: every vector shares a large common
component, so everything looks similar to everything. Measured on 42 words in six
semantic groups, comparing within-group against between-group cosine:

    bare words, raw            within 0.752  between 0.672  separation +0.080
    bare words, mean-centred   within 0.203  between -0.063  separation +0.266
    in a template, raw         within 0.784  between 0.736  separation +0.048
    in a template, centred     within 0.141  between -0.052  separation +0.193

Centring more than triples the usable structure, and wrapping the word in a carrier
phrase makes things worse. So: bare words, then subtract the mean.

Run:  secretspec run --reason "embed vocabulary" -- python3 -m salience.semantic.embeddings
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from salience.features import WORD, ascii_lower, extract_features

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "raw" / "paragraphs.jsonl"
OUT_DIR = ROOT / "data" / "semantic"

API_URL = "https://api.mistral.ai/v1/embeddings"
MODEL = "mistral-embed"
BATCH = 256


def build_vocabulary(corpus: Path, min_count: int, limit: int) -> list[str]:
    """Corpus words by descending frequency. Frequency order matters: if we later have to
    truncate, the words we drop are the ones the model sees least often anyway."""
    counts: collections.Counter[str] = collections.Counter()
    with corpus.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            for f in extract_features(json.loads(line)["text"]):
                if f.cls == WORD:
                    counts[ascii_lower(f.text)] += 1
    kept = [w for w, c in counts.most_common() if c >= min_count]
    return kept[:limit]


def api_key() -> str:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise SystemExit('MISTRAL_API_KEY unset — use: secretspec run --reason "..." -- ...')
    return key


def embed_batch(key: str, texts: list[str], attempts: int = 6) -> np.ndarray:
    body = json.dumps({"model": MODEL, "input": texts}).encode("utf-8")
    last: Exception | None = None
    for attempt in range(attempts):
        req = urllib.request.Request(
            API_URL,
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return np.array([e["embedding"] for e in payload["data"]], dtype=np.float32)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (408, 429) and exc.code < 500:
                raise RuntimeError(f"HTTP {exc.code}: {exc.read()[:200]!r}") from exc
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last = exc
        time.sleep(min(45, 2**attempt) * (1 + random.random() * 0.3))
    raise RuntimeError(f"embedding batch failed after {attempts} attempts: {last}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--min-count", type=int, default=2, help="drop words this rare")
    ap.add_argument("--limit", type=int, default=60_000)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    vocab_path, vec_path = args.out / "vocab.json", args.out / "embeddings.npy"

    print("building vocabulary...", end=" ", flush=True)
    vocab = build_vocabulary(args.corpus, args.min_count, args.limit)
    print(f"{len(vocab):,} words (count >= {args.min_count})")

    # Resume support: embedding is cheap but not instant, and losing it to a dropped
    # connection would be annoying.
    done = 0
    chunks: list[np.ndarray] = []
    if vocab_path.exists() and vec_path.exists():
        cached_vocab = json.loads(vocab_path.read_text())
        if cached_vocab[: len(cached_vocab)] == vocab[: len(cached_vocab)]:
            chunks.append(np.load(vec_path))
            done = len(cached_vocab)
            print(f"  resuming: {done:,} already embedded")

    key = api_key()
    start = time.time()
    for i in range(done, len(vocab), BATCH):
        chunks.append(embed_batch(key, vocab[i : i + BATCH]))
        n = min(i + BATCH, len(vocab))
        rate = (n - done) / max(1e-9, time.time() - start)
        print(f"  {n:,}/{len(vocab):,}  {rate:.0f} words/s", end="\r")
        if (i // BATCH) % 20 == 0:  # checkpoint periodically
            np.save(vec_path, np.concatenate(chunks))
            vocab_path.write_text(json.dumps(vocab[:n]))

    vectors = np.concatenate(chunks)
    np.save(vec_path, vectors)
    vocab_path.write_text(json.dumps(vocab))

    print(f"\n{vectors.shape[0]:,} x {vectors.shape[1]} embeddings -> {vec_path}")
    print(f"  {vectors.nbytes / 1e6:.0f} MB float32 on disk (not shipped; reduce.py shrinks it)")


if __name__ == "__main__":
    main()
