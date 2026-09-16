"""Tests for 6-bit quantization and the base-63 wire format.

The encoder writes one character per parameter. A silent off-by-one in the zigzag mapping
would corrupt every weight by a small amount — enough to degrade the model, not enough to
crash anything. These tests are the guard.
"""

from __future__ import annotations

import pytest
import torch

from salience.train.quant import (
    ALPHABET,
    QMAX,
    decode,
    encode,
    quantize,
    quantize_tensor,
    relative_error,
    round_trip,
)


def test_alphabet_is_63_distinct_json_safe_characters() -> None:
    assert len(ALPHABET) == 63
    assert len(set(ALPHABET)) == 63
    assert not (set(ALPHABET) & set('"\\\n\r\t'))


def test_every_code_round_trips() -> None:
    codes = torch.arange(-QMAX, QMAX + 1, dtype=torch.float32)
    text = encode(codes)
    assert len(text) == len(codes)
    assert torch.equal(decode(text, 1.0), codes)


def test_zigzag_puts_small_magnitudes_on_early_symbols() -> None:
    # 0,-1,1,-2,2 -> symbols 0,1,2,3,4. Keeps the common case in a narrow byte range,
    # which is what makes the payload compress.
    assert encode(torch.tensor([0.0, -1.0, 1.0, -2.0, 2.0])) == ALPHABET[:5]


def test_one_character_per_parameter() -> None:
    w = torch.randn(737, 32)
    text, _ = quantize_tensor(w)
    assert len(text) == w.numel()


def test_encode_rejects_out_of_range_codes() -> None:
    with pytest.raises(ValueError, match="out of range"):
        encode(torch.tensor([float(QMAX + 1)]))


def test_decode_rejects_foreign_characters() -> None:
    with pytest.raises(ValueError, match="alphabet"):
        decode("!!", 1.0)


def test_quantize_uses_the_full_range() -> None:
    q, _ = quantize(torch.randn(5000))
    assert q.abs().max().item() == QMAX
    assert q.abs().min().item() == 0


def test_quantization_is_idempotent() -> None:
    once = round_trip(torch.randn(2000))
    assert torch.equal(round_trip(once), once)


def test_zero_tensor_does_not_divide_by_zero() -> None:
    out = round_trip(torch.zeros(10))
    assert torch.all(out == 0)


def test_constant_tensor_survives() -> None:
    w = torch.full((100,), 0.37)
    assert torch.allclose(round_trip(w), w, atol=1e-6)


@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e3])
def test_relative_error_is_scale_invariant(scale: float) -> None:
    torch.manual_seed(0)
    w = torch.randn(4000) * scale
    # Six bits over a symmetric range gives a few percent RMS error regardless of units.
    assert relative_error(w) < 0.06


def test_outliers_cost_precision_for_everyone_else() -> None:
    """Per-tensor scaling means one huge weight coarsens the rest — worth knowing."""
    torch.manual_seed(0)
    normal = torch.randn(1000)
    spiked = normal.clone()
    spiked[0] = 100.0
    assert relative_error(spiked) > relative_error(normal) * 3
