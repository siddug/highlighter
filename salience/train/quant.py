"""Six-bit weight quantization and the base-63 wire format.

This is how gpu-lexer fits 41,321 parameters into 27KB: **one character per parameter**.
Each weight becomes an integer in [-31, 31], zigzag-encoded to [0, 62], and written as a
single character from a 63-symbol alphabet. A float32 scale per tensor converts back.

The result compresses well because neighbouring weights land on the same few characters,
which is exactly what Brotli is good at.
"""

from __future__ import annotations

import torch

# 63 symbols, all safe inside a JSON string with no escaping.
ALPHABET = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
assert len(ALPHABET) == 63 and len(set(ALPHABET)) == 63

DECODE = {c: i for i, c in enumerate(ALPHABET)}

# Symmetric 6-bit: 63 levels spanning [-31, 31].
QMAX = 31


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Round to the integer grid. Returns the integer codes and the scale."""
    scale = float(w.abs().max().clamp(min=1e-8) / QMAX)
    q = torch.clamp(torch.round(w / scale), -QMAX, QMAX)
    return q, scale


def dequantize(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q * scale


def fake_quant(w: torch.Tensor) -> torch.Tensor:
    """Quantize and immediately dequantize, passing gradients straight through.

    Used for quantization-aware training: the forward pass sees the values the browser
    will see, while the backward pass behaves as though nothing happened.
    """
    q, scale = quantize(w)
    return w + ((q * scale) - w).detach()


def encode(q: torch.Tensor) -> str:
    """Integer codes in [-31, 31] to one character each."""
    flat = q.flatten().to(torch.int64).tolist()
    out = []
    for v in flat:
        if not -QMAX <= v <= QMAX:
            raise ValueError(f"code {v} out of range")
        # Zigzag: 0,-1,1,-2,2 -> 0,1,2,3,4. Keeps small magnitudes on early symbols.
        out.append(ALPHABET[(v << 1) ^ (v >> 63) if v < 0 else v << 1])
    return "".join(out)


def decode(codes: str, scale: float) -> torch.Tensor:
    """Inverse of `encode`, scaled back to floats."""
    values = []
    for c in codes:
        u = DECODE.get(c)
        if u is None:
            raise ValueError(f"character {c!r} is not in the alphabet")
        values.append((u >> 1) ^ -(u & 1))
    return torch.tensor(values, dtype=torch.float32) * scale


def quantize_tensor(w: torch.Tensor) -> tuple[str, float]:
    q, scale = quantize(w)
    return encode(q), scale


def round_trip(w: torch.Tensor) -> torch.Tensor:
    """Encode then decode, for measuring exactly what quantization costs."""
    codes, scale = quantize_tensor(w)
    return decode(codes, scale).reshape(w.shape)


def relative_error(w: torch.Tensor) -> float:
    """RMS quantization error as a fraction of the tensor's RMS magnitude."""
    err = round_trip(w) - w
    denom = w.pow(2).mean().sqrt().clamp(min=1e-12)
    return float(err.pow(2).mean().sqrt() / denom)
