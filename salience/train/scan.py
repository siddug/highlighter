"""The learned Blelloch prefix scan, in PyTorch.

This is a direct port of `blelloch` in ts/src/cpu.ts and must stay numerically identical
to it. shared/golden/scan.jsonl pins the two together.

Why it is hand-written rather than borrowed
-------------------------------------------
The combine operator carries a different learned parameter at each level of the tree, so
it is **not associative**. It defines one specific balanced binary tree, and anything that
computes "the prefix scan" by another route computes a different number:

* a sequential `for t in range(T)` loop builds a right-leaning tree, not a balanced one;
* `torch.associative_scan` assumes true associativity (and is prototype-only: private
  import, requires torch.compile, CUDA-only codegen);
* the Heinsen / minGRU `cumsum + logcumsumexp` trick computes the mathematical prefix.

All three are wrong here. The up-sweep/down-sweep below is ~40 lines, is differentiable
through ordinary autograd, and runs in log2(T) vectorized steps — faster than a loop as
well as more correct.

Padding
-------
T must be a power of two and padding must sit at the **tail**, holding the identity
element (z=1, w=0). Padding then cannot affect real positions, because the exclusive
prefix at position i combines only leaves 0..i-1.

Note that the tree depth depends on T, and the level parameters depend on depth — so a
paragraph scanned at T=128 gives different numbers than the same paragraph at T=256. Every
caller must therefore pad to `next_pow2(len)` of the *individual sequence*. The Dataset
buckets batches by next_pow2 precisely so that batching cannot change results.
"""

from __future__ import annotations

import math

import torch

D = 32
LEVELS = 12


def next_pow2(n: int) -> int:
    return 1 << max(0, (max(1, n) - 1).bit_length())


def _combine(
    za: torch.Tensor,
    wa: torch.Tensor,
    zb: torch.Tensor,
    wb: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose segment A (left) with segment B (right) for s_i = z_i * s_{i-1} + w_i.

    The per-level gate `g` multiplies the **carry** term, not the composed decay. An
    earlier version put it on `z`, which made every down-sweep parameter dead: the final
    fold reads only the carry, and the carry never depends on the composed `z`. Gradients
    for `down` were exactly zero. Gating the carry gives both sweeps real influence —
    it is how much of the left segment's accumulated state reaches the right segment,
    learned separately at each level of the tree.
    """
    return za * zb, wb + zb * wa * g


def blelloch_scan(
    z: torch.Tensor, w: torch.Tensor, up: torch.Tensor, down: torch.Tensor
) -> torch.Tensor:
    """Inclusive prefix scan over the sequence dimension.

    Args:
        z, w: (B, T, D) gate and carry, T a power of two, tail padded with z=1, w=0.
        up, down: (LEVELS, D) per-tree-level parameters for the two sweeps.

    Returns:
        (B, T, D) inclusive scan of the carry.
    """
    if z.shape != w.shape:
        raise ValueError(f"z {tuple(z.shape)} and w {tuple(w.shape)} must match")
    b, t, d = z.shape
    if t & (t - 1):
        raise ValueError(f"sequence length {t} is not a power of two")
    if up.shape[-1] != d or down.shape[-1] != d:
        raise ValueError(f"level params have width {up.shape[-1]}, sequence has {d}")

    levels = int(math.log2(t))

    # Up-sweep. Keep the tree as a list of tensors: writing into one buffer in place is
    # where autograd silently produces wrong gradients.
    tree_z: list[torch.Tensor] = [z]
    tree_w: list[torch.Tensor] = [w]
    for level in range(levels):
        cz, cw = tree_z[level], tree_w[level]
        g = torch.sigmoid(up[min(level, LEVELS - 1)])
        pz, pw = _combine(cz[:, 0::2], cw[:, 0::2], cz[:, 1::2], cw[:, 1::2], g)
        tree_z.append(pz)
        tree_w.append(pw)

    # Down-sweep from an identity root yields the exclusive prefix at every leaf.
    cur_z = z.new_ones((b, 1, d))
    cur_w = z.new_zeros((b, 1, d))
    for level in range(levels - 1, -1, -1):
        g = torch.sigmoid(down[min(level, LEVELS - 1)])
        left_z, left_w = tree_z[level][:, 0::2], tree_w[level][:, 0::2]
        # Left child inherits the parent's prefix; right child adds the left subtree total.
        right_z, right_w = _combine(cur_z, cur_w, left_z, left_w, g)
        cur_z = torch.stack([cur_z, right_z], dim=2).reshape(b, -1, d)
        cur_w = torch.stack([cur_w, right_w], dim=2).reshape(b, -1, d)

    # Exclusive -> inclusive. No level parameter here; a plain fold of the leaf.
    return w + z * cur_w


def pad_to_pow2(
    z: torch.Tensor, w: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overwrite positions at or past each sequence's length with the identity element.

    Call this before blelloch_scan whenever a batch contains variable-length sequences.
    """
    b, t, _ = z.shape
    valid = (torch.arange(t, device=z.device)[None, :] < lengths[:, None])[..., None]
    return torch.where(valid, z, torch.ones_like(z)), torch.where(valid, w, torch.zeros_like(w))
