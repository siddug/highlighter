"""Export a trained checkpoint to the browser wire format.

Writes weights/model.json — every tensor as a base-63 string, one character per parameter,
plus a float32 scale. Also writes shared/golden/scores.jsonl, which pins the TypeScript
inference path to this exact model: same paragraphs, same scores, end to end.

On quantization-aware training
------------------------------
We measured before building it. Post-training quantization to 6 bits moved held-out
Spearman from 0.6218 to 0.6221 and P@15% by -0.002 — inside the noise. QAT would have been
several hundred lines of straight-through estimators and a second training phase to buy
nothing, so there is none. If a future architecture change makes PTQ lossy, `fake_quant`
in quant.py is the primitive to build it from.

Run:  python3 -m salience.train.export
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from salience.features import ROW_TOTAL, SPEC_VERSION
from salience.train.data import load_examples, split
from salience.train.model import SalienceModel
from salience.train.quant import quantize_tensor, relative_error

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "data" / "checkpoints" / "model.pt"
WEIGHTS = ROOT / "weights" / "model.json"
SCORES = ROOT / "shared" / "golden" / "scores.jsonl"

# PyTorch parameter name -> the path TENSOR_SHAPES uses in ts/src/model.ts.
NAME_MAP = {
    "emb.weight": "emb",
    "local_bias": "localBias",
    "local_win": "localWin",
    "local_ptr": "localPtr",
    "fwd.wz.weight": "fwd.wz",
    "fwd.wz.bias": "fwd.bz",
    "fwd.wc.weight": "fwd.wc",
    "fwd.wc.bias": "fwd.bc",
    "fwd.up": "fwd.up",
    "fwd.down": "fwd.down",
    "bwd.wz.weight": "bwd.wz",
    "bwd.wz.bias": "bwd.bz",
    "bwd.wc.weight": "bwd.wc",
    "bwd.wc.bias": "bwd.bc",
    "bwd.up": "bwd.up",
    "bwd.down": "bwd.down",
    "head_gate.weight": "headGateW",
    "head_gate.bias": "headGateB",
    "head_hid.weight": "headHidW",
    "head_hid.bias": "headHidB",
    "head_out.weight": "headOutW",
    "head_out.bias": "headOutB",
}


def export_weights(model: SalienceModel) -> dict:
    tensors: dict[str, dict] = {}
    total = 0
    worst = ("", 0.0)

    for name, param in model.named_parameters():
        path = NAME_MAP.get(name)
        if path is None:
            raise KeyError(f"no TypeScript name for parameter {name!r}")

        data = param.detach()
        if path == "emb":
            # Drop the zero-pinned padding row; the browser never indexes it.
            data = data[:ROW_TOTAL]

        codes, scale = quantize_tensor(data)
        # main() already quantized in place, so this must be exactly zero — it checks that
        # quantization is idempotent rather than measuring what it cost.
        err = relative_error(data)
        if err > worst[1]:
            worst = (path, err)

        tensors[path] = {"shape": list(data.shape), "scale": scale, "data": codes}
        total += data.numel()

    print(f"  {total:,} parameters, re-quantization drift {worst[1]:.2e} (must be ~0)")
    return {
        "specVersion": SPEC_VERSION,
        "paramCount": total,
        "alphabet": "base63-zigzag",
        "tensors": tensors,
    }


@torch.no_grad()
def export_scores(model: SalienceModel, labels: Path, limit: int) -> list[dict]:
    """Per-word scores on held-out paragraphs, for the TypeScript conformance test."""
    from salience.train.data import collate  # noqa: PLC0415

    with labels.open(encoding="utf-8") as fh:
        records = {}
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                records[rec["id"]] = rec

    _, val = split(load_examples(labels), 0.15, 0)
    model.eval()

    out = []
    for ex in val[:limit]:
        batch = collate([ex])
        probs = torch.sigmoid(model(batch["rows"], batch["ptrs"], batch["lengths"]))[0]
        mask = batch["is_word"][0]
        out.append(
            {
                "text": records[ex.pid]["text"],
                "scores": [round(v, 6) for v in probs[mask].tolist()],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--labels", type=Path, default=ROOT / "data" / "labeled" / "labels.jsonl")
    ap.add_argument("--out", type=Path, default=WEIGHTS)
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--score-count", type=int, default=40)
    args = ap.parse_args()

    model = SalienceModel()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])

    # Quantize in place so the exported scores match what the browser will compute.
    from salience.train.quant import round_trip  # noqa: PLC0415

    with torch.no_grad():
        for p in model.parameters():
            p.copy_(round_trip(p.data))
        model.emb.weight[model.emb.padding_idx].zero_()

    print(f"exporting {args.checkpoint}")
    payload = export_weights(model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    size = args.out.stat().st_size
    print(f"  wrote {args.out} ({size / 1024:.1f} KB raw)")

    scores = export_scores(model, args.labels, args.score_count)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    with args.scores.open("w", encoding="utf-8") as fh:
        for rec in scores:
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"  wrote {len(scores)} score fixtures to {args.scores}")


if __name__ == "__main__":
    main()
