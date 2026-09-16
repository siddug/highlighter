"""Shrink 1024-dim word embeddings into something shippable.

`mistral-embed` gives 29,191 x 1024 float32 = 120 MB. The v2 budget is ~1 MB brotli, so
this reduces them two ways so the training experiment can decide which is worth the bytes:

    projected   PCA to d dims, then 6-bit quantized. Keeps a continuous vector per word.
    clusters    k-means to K groups, ship one integer per word. Keeps only "which bucket".

Projection carries strictly more information per word; clustering is far smaller and slots
into the existing feature-bag design without changing it. Both are produced here and both
get trained in salience/semantic/train_v2.py.

Mean-centring happens here, not in embeddings.py, so the raw API output stays untouched on
disk — see the measurement in that module for why centring matters so much.

Run:  python3 -m salience.semantic.reduce
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEM_DIR = ROOT / "data" / "semantic"


def load(sem_dir: Path) -> tuple[list[str], np.ndarray]:
    vocab = json.loads((sem_dir / "vocab.json").read_text())
    vectors = np.load(sem_dir / "embeddings.npy")
    if len(vocab) != len(vectors):
        raise SystemExit(f"vocab {len(vocab)} != vectors {len(vectors)} — rerun embeddings.py")
    return vocab, vectors


def pca(x: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Project onto the top `dims` principal directions. Returns (projected, explained)."""
    # Economy SVD on the centred matrix; 29k x 1024 is small enough to do directly.
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    projected = u[:, :dims] * s[:dims]
    explained = (s**2) / (s**2).sum()
    return projected.astype(np.float32), explained


def kmeans(x: np.ndarray, k: int, iters: int = 40, seed: int = 0) -> np.ndarray:
    """Plain Lloyd's algorithm on L2-normalized vectors, so distance is cosine-equivalent."""
    rng = np.random.default_rng(seed)
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)

    # k-means++ seeding: a random start leaves many clusters empty at this k.
    centres = [xn[rng.integers(len(xn))]]
    closest = np.full(len(xn), np.inf)
    for _ in range(k - 1):
        closest = np.minimum(closest, ((xn - centres[-1]) ** 2).sum(1))
        probs = closest / max(closest.sum(), 1e-12)
        centres.append(xn[rng.choice(len(xn), p=probs)])
    c = np.stack(centres)

    assign = np.zeros(len(xn), dtype=np.int32)
    for _ in range(iters):
        assign = np.argmax(xn @ c.T, axis=1).astype(np.int32)
        for j in range(k):
            members = xn[assign == j]
            if len(members):
                v = members.mean(0)
                c[j] = v / max(np.linalg.norm(v), 1e-9)
    return assign


def neighbours(vocab: list[str], x: np.ndarray, word: str, n: int = 6) -> list[str]:
    if word not in vocab:
        return []
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    sims = xn @ xn[vocab.index(word)]
    return [vocab[i] for i in np.argsort(-sims)[1 : n + 1]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sem-dir", type=Path, default=SEM_DIR)
    ap.add_argument("--dims", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--clusters", type=int, nargs="+", default=[256, 1024])
    args = ap.parse_args()

    vocab, raw = load(args.sem_dir)
    print(f"{len(vocab):,} words x {raw.shape[1]} dims")

    centred = raw - raw.mean(0)
    print(f"mean-centred (the common component was {np.linalg.norm(raw.mean(0)):.2f} long)")

    print("\nsanity check — nearest neighbours after centring")
    for probe in ("reactor", "criticality", "however", "quarterly"):
        got = neighbours(vocab, centred, probe)
        if got:
            print(f"  {probe:<14} {', '.join(got)}")

    print("\nPROJECTED VARIANTS")
    for d in args.dims:
        proj, explained = pca(centred, d)
        # Standardize so the scale is comparable to the learned embedding rows.
        proj = proj / proj.std()
        np.save(args.sem_dir / f"projected{d}.npy", proj)
        codes = len(vocab) * d
        print(
            f"  {d:>3} dims   {explained[:d].sum():>5.1%} variance   "
            f"{codes:>9,} values   ~{codes / 1024:.0f} KB at 6 bits"
        )

    print("\nCLUSTER VARIANTS")
    for k in args.clusters:
        assign = kmeans(centred, k)
        np.save(args.sem_dir / f"clusters{k}.npy", assign)
        used = len(set(assign.tolist()))
        print(f"  {k:>4} clusters   {used} non-empty   {len(vocab):,} ids   ~{len(vocab) * 2 / 1024:.0f} KB raw")
        if k == args.clusters[0]:
            for probe in ("reactor", "however"):
                if probe in vocab:
                    j = assign[vocab.index(probe)]
                    members = [vocab[i] for i in np.where(assign == j)[0][:8]]
                    print(f"      cluster of {probe!r}: {', '.join(members)}")

    print(f"\nwrote to {args.sem_dir}")


if __name__ == "__main__":
    main()
