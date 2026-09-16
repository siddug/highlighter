#!/usr/bin/env python3
"""Emit shared/golden/scan.jsonl, pinning the PyTorch scan to the TypeScript one.

The learned Blelloch scan is the one part of the model where a plausible-looking
reimplementation silently computes a different number (see salience/train/scan.py for why
it is not associative). A cross-language fixture is the only way to know the two agree.

Python generates, TypeScript verifies — same direction as the feature fixtures.

Run:  python3 scripts/make_scan_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from salience.train.scan import D, LEVELS, blelloch_scan, next_pow2  # noqa: E402

GOLDEN = ROOT / "shared" / "golden" / "scan.jsonl"

# Lengths chosen to exercise every padding case: exact powers of two, one over, one under,
# and the degenerate n=1.
LENGTHS = (1, 2, 3, 5, 8, 9, 17, 32)


def case(n: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    npad = next_pow2(n)

    z = torch.ones(1, npad, D, dtype=torch.float64)
    w = torch.zeros(1, npad, D, dtype=torch.float64)
    # Gate in (0,1) as sigmoid would produce; carry spanning both signs.
    z[0, :n] = torch.rand(n, D, generator=g, dtype=torch.float64) * 0.9 + 0.05
    w[0, :n] = torch.rand(n, D, generator=g, dtype=torch.float64) * 2 - 1

    up = torch.rand(LEVELS, D, generator=g, dtype=torch.float64) * 4 - 2
    down = torch.rand(LEVELS, D, generator=g, dtype=torch.float64) * 4 - 2

    out = blelloch_scan(z, w, up, down)

    rnd = lambda t: [round(v, 9) for v in t.flatten().tolist()]  # noqa: E731
    return {
        "n": n,
        "npad": npad,
        "z": rnd(z[0, :n]),
        "w": rnd(w[0, :n]),
        "up": rnd(up),
        "down": rnd(down),
        "out": rnd(out[0, :n]),
    }


def main() -> None:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN.open("w", encoding="utf-8") as fh:
        for i, n in enumerate(LENGTHS):
            fh.write(json.dumps(case(n, 1000 + i), separators=(",", ":")) + "\n")
    print(f"wrote {len(LENGTHS)} scan cases to {GOLDEN}")


if __name__ == "__main__":
    main()
