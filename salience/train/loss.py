"""The training objective.

Three terms, because "which words are important" is three different claims at once:

1. **Soft BCE** — per-word calibration against the oracle's agreement frequency.
2. **ListNet** — the *ordering* of words within a paragraph. Highlighting is a selection
   problem: what matters is which words outrank which, not their absolute probabilities.
   BCE alone is indifferent to ordering as long as each word is individually well
   calibrated, and ordering is exactly what a top-k reader consumes.
3. **Budget** — holds the mean predicted rate near a target, so inference can use a fixed
   0.5 threshold instead of a per-document quantile.

Everything is masked to WORD tokens. The head is never asked about whitespace or
punctuation, and those positions must not contribute gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    bce: float = 1.0
    listnet: float = 1.0
    budget: float = 0.5
    # Oracle positives run ~25%, so a lone positive is worth about three negatives.
    pos_weight: float = 3.0
    # Sharpens the oracle's soft targets into a ranking distribution. Below 1 it emphasises
    # the words samples agreed on most.
    tau: float = 0.5
    target_rate: float = 0.20


def soft_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """Binary cross-entropy against soft targets, weighted toward the positive class."""
    losses = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=torch.tensor(pos_weight, device=logits.device)
    )
    return (losses * mask).sum() / mask.sum().clamp(min=1)


def listnet(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, tau: float) -> torch.Tensor:
    """Cross-entropy between the target and predicted distributions over each paragraph."""
    rows = mask.any(dim=-1)
    if not rows.any():
        return logits.sum() * 0.0

    logits, targets, mask = logits[rows], targets[rows], mask[rows]
    neg = torch.finfo(logits.dtype).min

    log_pred = F.log_softmax(logits.masked_fill(~mask, neg), dim=-1)
    target_dist = F.softmax((targets / tau).masked_fill(~mask, neg), dim=-1)
    return -(target_dist * log_pred).sum(dim=-1).mean()


def budget(logits: torch.Tensor, mask: torch.Tensor, target_rate: float) -> torch.Tensor:
    """Squared error between the mean predicted probability and the desired rate."""
    probs = torch.sigmoid(logits) * mask
    rate = probs.sum() / mask.sum().clamp(min=1)
    return (rate - target_rate) ** 2


def compute_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, w: LossWeights
) -> tuple[torch.Tensor, dict[str, float]]:
    mask_f = mask.float()
    l_bce = soft_bce(logits, targets, mask_f, w.pos_weight)
    l_list = listnet(logits, targets, mask, w.tau)
    l_budget = budget(logits, mask_f, w.target_rate)

    total = w.bce * l_bce + w.listnet * l_list + w.budget * l_budget
    return total, {
        "loss": total.item(),
        "bce": l_bce.item(),
        "listnet": l_list.item(),
        "budget": l_budget.item(),
    }
