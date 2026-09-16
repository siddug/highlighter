"""The salience model, in PyTorch.

A parameter-for-parameter mirror of ts/src/model.ts and ts/src/cpu.ts:

    feature-bag embedding -> local 5-token window + 2 pointers
                          -> bidirectional learned Blelloch scan
                          -> gated MLP head -> one logit per word

39,361 parameters. Every tensor here has a counterpart in TENSOR_SHAPES on the TypeScript
side, and salience/train/export.py relies on that correspondence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dataclasses import dataclass

from salience.features import ROW_TOTAL
from salience.train.scan import D, LEVELS, blelloch_scan, pad_to_pow2

WINDOW = 5
PTRS = 2
GATE = 16
HID = 72
HEAD_IN = D * 3


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the network.

    The defaults are the shipped model and match ts/src/model.ts exactly — changing them
    breaks browser conformance until the TypeScript side is updated and the weights are
    re-exported. They are configurable so capacity can be measured rather than guessed.
    """

    d_model: int = D
    hid: int = HID
    gate: int = GATE
    window: int = WINDOW
    ptrs: int = PTRS

    @property
    def head_in(self) -> int:
        return self.d_model * 3

    def describe(self) -> str:
        return f"d{self.d_model}·h{self.hid}·g{self.gate}·w{self.window}"


DEFAULT_CONFIG = ModelConfig()

# Row index used to pad a token's variable-length row list. Its embedding is pinned to
# zero and excluded on export, so it costs nothing in the shipped model.
PAD_ROW = ROW_TOTAL
# class + sentence position + document position, 7 word fields, and at most 8 flags
# (CAPITALIZED and ALL_CAPS are mutually exclusive).
MAX_ROWS = 20

# See shared/spec/features.md, "Initialization requirement". Without this the carry decays
# as 0.5^distance and underflows float32 within ~100 tokens.
FORGET_BIAS_INIT = 2.0

# The scan's per-level carry gate, sigmoid(3.0) ~ 0.95. At zero init the gate sits at 0.5
# and the carry loses half its magnitude at every level of the tree — 0.5^9 over a padded
# 512-token sequence. Same failure mode as FORGET_BIAS_INIT, one level up.
LEVEL_GATE_INIT = 3.0


class ScanBlock(nn.Module):
    """Minimal-GRU gate feeding one direction of the learned prefix scan.

    Two learned projections, named after the GRU literature:

        wz -> z, the update gate      how much of the running state to keep
        wc -> c, the candidate        what this token wants the state to become

    The recurrence they define is a convex blend:

        state = z * state + (1 - z) * c

    The scan itself takes the generic form `state = z * state + w`, so the
    `(1 - z) * c` term is precomputed once per token and handed over as `w`.
    """

    def __init__(self, d: int = D) -> None:
        super().__init__()
        self.wz = nn.Linear(d, d)
        self.wc = nn.Linear(d, d)
        self.up = nn.Parameter(torch.full((LEVELS, d), LEVEL_GATE_INIT))
        self.down = nn.Parameter(torch.full((LEVELS, d), LEVEL_GATE_INIT))
        nn.init.zeros_(self.wz.bias)
        self.wz.bias.data.add_(FORGET_BIAS_INIT)
        nn.init.zeros_(self.wc.bias)

    def forward(self, h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.wz(h))  # update gate, 0..1 per channel
        candidate = torch.tanh(self.wc(h))  # what this token proposes, -1..1
        w = (1 - z) * candidate  # the write term the scan expects
        z, w = pad_to_pow2(z, w, lengths)
        return blelloch_scan(z, w, self.up, self.down)


def _reverse_index(t: int, lengths: torch.Tensor) -> torch.Tensor:
    """Index map reversing the first `len` positions of each row and fixing the tail.

    Reversing the padded tail as well would drag padding into the middle of the sequence,
    where the scan's identity-element guarantee no longer holds. The map is an involution,
    so the same tensor un-reverses the result.
    """
    pos = torch.arange(t, device=lengths.device)[None, :].expand(lengths.shape[0], t)
    flipped = (lengths[:, None] - 1 - pos).clamp(min=0)
    return torch.where(pos < lengths[:, None], flipped, pos)


def _gather_seq(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


class SalienceModel(nn.Module):
    def __init__(self, config: ModelConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.emb = nn.Embedding(ROW_TOTAL + 1, d, padding_idx=PAD_ROW)
        nn.init.normal_(self.emb.weight, std=0.1)
        with torch.no_grad():
            self.emb.weight[PAD_ROW].zero_()

        self.local_bias = nn.Parameter(torch.zeros(d))
        self.local_win = nn.Parameter(torch.randn(config.window, d) * 0.3)
        self.local_ptr = nn.Parameter(torch.randn(config.ptrs, d) * 0.1)

        self.fwd = ScanBlock(d)
        self.bwd = ScanBlock(d)

        self.head_gate = nn.Linear(config.head_in, config.gate)
        self.head_hid = nn.Linear(config.head_in + config.gate, config.hid)
        self.head_out = nn.Linear(config.hid, 1)

    def embed(self, rows: torch.Tensor) -> torch.Tensor:
        """Feature bag: sum the embedding rows each token selects. (B,T,R) -> (B,T,D)."""
        return self.emb(rows).sum(dim=2)

    def local(self, emb: torch.Tensor, ptrs: torch.Tensor) -> torch.Tensor:
        b, t, _ = emb.shape
        half = (self.config.window - 1) // 2

        out = self.local_bias.expand(b, t, self.config.d_model).clone()
        padded = F.pad(emb, (0, 0, half, half))
        for k in range(self.config.window):
            out = out + self.local_win[k] * padded[:, k : k + t]

        for p in range(self.config.ptrs):
            idx = ptrs[:, :, p]
            valid = (idx >= 0).unsqueeze(-1)
            gathered = _gather_seq(emb, idx.clamp(min=0))
            out = out + self.local_ptr[p] * torch.where(valid, gathered, torch.zeros_like(gathered))

        return torch.tanh(out)

    def forward(
        self, rows: torch.Tensor, ptrs: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Return one logit per token. Callers mask to WORD tokens themselves.

        Args:
            rows:    (B, T, MAX_ROWS) int64 embedding row indices, PAD_ROW-padded.
            ptrs:    (B, T, 2) int64 pointer token indices, -1 when absent.
            lengths: (B,) int64 token counts. T must be next_pow2 of each length, so
                     batches must be bucketed by next_pow2 — see salience/train/data.py.
        """
        emb = self.embed(rows)
        local = self.local(emb, ptrs)

        gf = self.fwd(local, lengths)

        rev = _reverse_index(local.shape[1], lengths)
        gb = _gather_seq(self.bwd(_gather_seq(local, rev), lengths), rev)

        x = torch.cat([local, gf, gb], dim=-1)
        gate = torch.sigmoid(self.head_gate(x))
        hidden = torch.tanh(self.head_hid(torch.cat([x, gate], dim=-1)))
        return self.head_out(hidden).squeeze(-1)


def parameter_count(model: SalienceModel) -> int:
    """Shipped parameters: everything except the zero-pinned padding row."""
    return sum(p.numel() for p in model.parameters()) - model.config.d_model
