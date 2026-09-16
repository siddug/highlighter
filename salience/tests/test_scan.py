"""Tests for the learned Blelloch scan.

The scan is the part of the model most likely to be subtly wrong: it is not associative,
it walks a specific tree, and a plausible reimplementation produces plausible numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from salience.train.scan import D, LEVELS, blelloch_scan, next_pow2, pad_to_pow2

GOLDEN = Path(__file__).resolve().parents[2] / "shared" / "golden" / "scan.jsonl"

# sigmoid(30) is 1 to within float precision, making the combine truly associative.
SATURATED = torch.full((LEVELS, D), 30.0, dtype=torch.float64)


def sequential(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The plain recurrence s_i = z_i * s_{i-1} + w_i."""
    out = torch.zeros_like(w)
    state = torch.zeros(w.shape[0], D, dtype=w.dtype)
    for t in range(w.shape[1]):
        state = z[:, t] * state + w[:, t]
        out[:, t] = state
    return out


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 64, 256])
def test_matches_the_sequential_recurrence_when_gates_are_open(n: int) -> None:
    torch.manual_seed(n)
    z = torch.rand(2, n, D, dtype=torch.float64) * 0.9 + 0.05
    w = torch.rand(2, n, D, dtype=torch.float64) * 2 - 1
    got = blelloch_scan(z, w, SATURATED, SATURATED)
    assert torch.allclose(got, sequential(z, w), atol=1e-9)


def test_propagates_a_single_leaf_to_the_end() -> None:
    n = 512
    z = torch.ones(1, n, D, dtype=torch.float64)
    w = torch.zeros(1, n, D, dtype=torch.float64)
    w[0, 0] = 1.0
    out = blelloch_scan(z, w, SATURATED, SATURATED)
    assert torch.allclose(out[0, -1], torch.ones(D, dtype=torch.float64), atol=1e-9)


def test_padding_does_not_affect_real_positions() -> None:
    torch.manual_seed(7)
    n = 40
    z = torch.rand(1, n, D, dtype=torch.float64) * 0.9 + 0.05
    w = torch.rand(1, n, D, dtype=torch.float64) * 2 - 1

    short = blelloch_scan(*pad_to_pow2(
        torch.cat([z, torch.zeros(1, 24, D, dtype=torch.float64)], 1),
        torch.cat([w, torch.zeros(1, 24, D, dtype=torch.float64)], 1),
        torch.tensor([n]),
    ), SATURATED, SATURATED)
    assert torch.allclose(short[0, :n], sequential(z, w)[0], atol=1e-9)


def test_down_sweep_parameters_affect_the_output() -> None:
    """Regression: an earlier combine gated `z` instead of the carry, and every
    down-sweep gradient was exactly zero."""
    torch.manual_seed(1)
    z = torch.rand(1, 16, D, dtype=torch.float64) * 0.9 + 0.05
    w = torch.rand(1, 16, D, dtype=torch.float64) * 2 - 1
    up = torch.zeros(LEVELS, D, dtype=torch.float64, requires_grad=True)
    down = torch.zeros(LEVELS, D, dtype=torch.float64, requires_grad=True)

    blelloch_scan(z, w, up, down).sum().backward()
    assert up.grad.abs().sum() > 0
    assert down.grad.abs().sum() > 0


def test_rejects_non_power_of_two_length() -> None:
    with pytest.raises(ValueError, match="power of two"):
        blelloch_scan(torch.ones(1, 7, D), torch.zeros(1, 7, D), SATURATED.float(), SATURATED.float())


@pytest.mark.parametrize("n,expected", [(1, 1), (2, 2), (3, 4), (8, 8), (9, 16), (130, 256)])
def test_next_pow2(n: int, expected: int) -> None:
    assert next_pow2(n) == expected


def test_matches_the_typescript_fixture() -> None:
    """Regression guard on shared/golden/scan.jsonl, which ts/src/scan.golden.test.ts
    also checks. If this drifts, the browser and the trainer have diverged."""
    with GOLDEN.open(encoding="utf-8") as fh:
        cases = [json.loads(line) for line in fh if line.strip()]
    assert len(cases) >= 8

    for c in cases:
        n, npad = c["n"], c["npad"]
        z = torch.ones(1, npad, D, dtype=torch.float64)
        w = torch.zeros(1, npad, D, dtype=torch.float64)
        z[0, :n] = torch.tensor(c["z"], dtype=torch.float64).reshape(n, D)
        w[0, :n] = torch.tensor(c["w"], dtype=torch.float64).reshape(n, D)
        up = torch.tensor(c["up"], dtype=torch.float64).reshape(LEVELS, D)
        down = torch.tensor(c["down"], dtype=torch.float64).reshape(LEVELS, D)

        got = blelloch_scan(z, w, up, down)[0, :n]
        want = torch.tensor(c["out"], dtype=torch.float64).reshape(n, D)
        assert torch.allclose(got, want, atol=1e-8), f"n={n}"
