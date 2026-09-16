#!/usr/bin/env python3
"""Local server for the salience model.

Serves the built web client and exposes the PyTorch model over HTTP, so the client can run
the same paragraph through both runtimes and show that they agree. That agreement is not a
hope — ts/src/scores.golden.test.ts pins the two to within 2e-4 on every word.

Standard library only. The project has no web framework dependency and does not need one
for a local test server.

Usage:
    npm run build            # produces dist/
    python3 serve.py         # http://localhost:8000

For client development, run `npm run dev` instead — Vite proxies /api here.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import unquote

import torch

from salience.features import SPEC_VERSION, WORD, extract_features
from salience.train.data import collate
from salience.train.model import MAX_ROWS, PAD_ROW, SalienceModel, parameter_count

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
CKPT = ROOT / "data" / "checkpoints" / "model.pt"

MAX_BODY = 256 * 1024
MAX_CHARS = 20_000

_model: SalienceModel
_lock = Lock()


def load_model(checkpoint: Path) -> SalienceModel:
    from salience.train.quant import round_trip  # noqa: PLC0415

    model = SalienceModel()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    # Serve exactly what the browser serves: the 6-bit quantized weights, not float32.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(round_trip(p.data))
        model.emb.weight[model.emb.padding_idx].zero_()
    model.eval()
    return model


@torch.no_grad()
def score_text(text: str) -> dict:
    feats = extract_features(text)
    if not feats:
        return {"tokens": [], "ms": 0.0}

    start = time.perf_counter()

    n = len(feats)
    rows = torch.full((n, MAX_ROWS), PAD_ROW, dtype=torch.long)
    ptrs = torch.empty((n, 2), dtype=torch.long)
    is_word = torch.zeros(n, dtype=torch.bool)
    for i, f in enumerate(feats):
        rows[i, : len(f.rows)] = torch.tensor(f.rows, dtype=torch.long)
        ptrs[i, 0] = f.sentence_first_word
        ptrs[i, 1] = f.prev_sentence_last_word
        is_word[i] = f.cls == WORD

    from salience.train.data import Example  # noqa: PLC0415

    batch = collate([Example(rows, ptrs, is_word, torch.zeros(n), n, "live", "live")])
    with _lock:
        logits = _model(batch["rows"], batch["ptrs"], batch["lengths"])
    probs = torch.sigmoid(logits)[0]

    tokens = [
        {
            "text": f.text,
            "start": f.start,
            "end": f.end,
            "cls": f.cls,
            "score": round(float(probs[i]), 6) if f.cls == WORD else 0.0,
            # Tokens sharing this value are in the same sentence; the client groups on it.
            "sentence": f.sentence_first_word,
        }
        for i, f in enumerate(feats)
    ]
    return {"tokens": tokens, "ms": round((time.perf_counter() - start) * 1000, 2)}


class Handler(BaseHTTPRequestHandler):
    server_version = "salience/0.1"

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]

        if path == "/api/health":
            self._json(
                200,
                {
                    "ok": True,
                    "specVersion": SPEC_VERSION,
                    "parameters": parameter_count(_model),
                },
            )
            return

        if path.startswith("/api/"):
            self._json(404, {"error": "unknown endpoint"})
            return

        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/api/score":
            self._json(404, {"error": "unknown endpoint"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            self._json(413, {"error": f"body must be 1..{MAX_BODY} bytes"})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = payload["text"]
            if not isinstance(text, str):
                raise TypeError("text must be a string")
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as exc:
            self._json(400, {"error": f"expected JSON with a string 'text' field: {exc}"})
            return

        if len(text) > MAX_CHARS:
            self._json(413, {"error": f"text must be at most {MAX_CHARS} characters"})
            return

        try:
            self._json(200, score_text(text))
        except Exception as exc:  # noqa: BLE001 - never take the server down for one request
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _serve_static(self, path: str) -> None:
        if not DIST.exists():
            self._send(
                503,
                b"dist/ not found. Run `npm run build`, or use `npm run dev` for the client.",
                "text/plain; charset=utf-8",
            )
            return

        # Decode first so percent-encoded filenames resolve, then let the containment
        # check below reject traversal. Relying on "%2f never decodes" would be accidental
        # safety rather than a guarantee.
        relative = unquote(path).lstrip("/") or "index.html"
        if "\x00" in relative:
            self._json(400, {"error": "bad path"})
            return

        target = (DIST / relative).resolve()
        # Reject anything that escapes dist/, however it was spelled.
        if not target.is_relative_to(DIST.resolve()):
            self._json(403, {"error": "forbidden"})
            return
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            target = DIST / "index.html"  # SPA fallback
        if not target.is_file():
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return

        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype == "application/javascript":
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)


def main() -> None:
    global _model  # noqa: PLW0603

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"{args.checkpoint} not found — run: python3 -m salience.train.run")

    print(f"loading {args.checkpoint}")
    _model = load_model(args.checkpoint)
    print(f"  {parameter_count(_model):,} parameters, spec {SPEC_VERSION}")
    print(f"  client: {'dist/' if DIST.exists() else 'not built — run npm run build'}")
    print(f"\nlistening on http://{args.host}:{args.port}\n")

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
