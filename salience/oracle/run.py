"""Run the LLM oracle over a corpus and write soft salience targets.

Samples the oracle N times per paragraph at non-zero temperature and averages the results
(see prompt.to_soft_targets). Every response is cached on disk keyed by model, prompt
version, paragraph, and sample index, so re-runs cost nothing and interruptions are free.

Secrets come from secretspec (see secretspec.toml), so the key never lands in a shell
history or a dotenv file:

    secretspec set MISTRAL_API_KEY
    secretspec run --reason "label salience corpus" -- \\
        python3 -m salience.oracle.run --limit 1000 --samples 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import urllib.error
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from salience.oracle.prompt import (
    PROMPT_VERSION,
    SYSTEM,
    build_user_message,
    parse_response,
    to_soft_targets,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "raw" / "paragraphs.jsonl"
OUT_DIR = ROOT / "data" / "labeled"
CACHE = OUT_DIR / "cache.jsonl"

API_URL = "https://api.mistral.ai/v1/chat/completions"
MODELS_URL = "https://api.mistral.ai/v1/models"

# Small model for bulk volume, large for the gold set we measure against.
BULK_MODEL = "mistral-small-latest"
GOLD_MODEL = "mistral-large-latest"

MAX_TOKENS = 1024
# Non-zero on purpose: we sample the oracle N times and average, so the spread between
# samples is signal about how borderline each word is.
TEMPERATURE = 0.7


def cache_key(model: str, paragraph_id: str, sample: int) -> str:
    raw = f"{model}|{PROMPT_VERSION}|{paragraph_id}|{sample}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


class Cache:
    """Append-only JSONL cache. Cheap to write, trivially recoverable, no db."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, str] = {}
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    self._data[rec["k"]] = rec["v"]
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        with self._lock:
            if key in self._data:
                return
            self._data[key] = value
            self._fh.write(json.dumps({"k": key, "v": value}, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __len__(self) -> int:
        return len(self._data)


class Throttle:
    """Shared back-off across all workers.

    Per-thread exponential back-off does not work against a rate limit: the other 39
    threads keep hammering the endpoint while one sleeps, so the limit never clears and
    calls fail permanently. A 40-worker run lost 33 paragraphs that way. This makes a 429
    on any thread pause every thread.
    """

    def __init__(self) -> None:
        self._until = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                remaining = self._until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 5.0))

    def penalize(self, seconds: float) -> None:
        with self._lock:
            self._until = max(self._until, time.monotonic() + seconds)


THROTTLE = Throttle()


def api_key() -> str:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise SystemExit(
            "MISTRAL_API_KEY is not set. Run:\n"
            "  secretspec set MISTRAL_API_KEY\n"
            '  secretspec run --reason "label salience corpus" -- python3 -m salience.oracle.run'
        )
    return key


def list_models(key: str) -> list[str]:
    req = urllib.request.Request(MODELS_URL, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return sorted(m["id"] for m in json.loads(resp.read().decode("utf-8"))["data"])


def call_with_retry(key: str, model: str, user: str, attempts: int = 8) -> str:
    """One oracle call. Retries on rate limits and transient server errors."""
    body = json.dumps(
        {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
        }
    ).encode("utf-8")

    last: Exception | None = None
    for attempt in range(attempts):
        THROTTLE.wait()
        req = urllib.request.Request(
            API_URL,
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            last = exc
            # 4xx other than rate limiting will not fix themselves.
            if exc.code not in (408, 429) and exc.code < 500:
                raise RuntimeError(
                    f"HTTP {exc.code}: {exc.read()[:200].decode('utf-8', 'replace')}"
                ) from exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() else None
                THROTTLE.penalize(delay if delay is not None else min(45.0, 3.0 * 2**attempt))
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last = exc
        if attempt < attempts - 1:
            time.sleep(min(30, 2**attempt) * (1 + random.random() * 0.3))
    raise RuntimeError(f"oracle call failed after {attempts} attempts: {last}")


def label_one(client: str, cache: Cache, model: str, para: dict, samples: int) -> dict | None:
    user, word_count = build_user_message(para["text"])
    if word_count == 0:
        return None

    parsed: list[list[int]] = []
    for s in range(samples):
        key = cache_key(model, para["id"], s)
        raw = cache.get(key)
        if raw is None:
            raw = call_with_retry(client, model, user)
            cache.put(key, raw)
        try:
            parsed.append(parse_response(raw, word_count))
        except ValueError:
            continue

    if not parsed:
        return None

    return {
        "id": para["id"],
        "domain": para["domain"],
        "text": para["text"],
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "word_count": word_count,
        "samples": parsed,
        "targets": [round(v, 4) for v in to_soft_targets(parsed, word_count)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=1000, help="paragraphs to label")
    ap.add_argument("--samples", type=int, default=5, help="oracle samples per paragraph")
    ap.add_argument("--model", default=BULK_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "labels.jsonl")
    ap.add_argument("--offset", type=int, default=0, help="skip this many paragraphs")
    ap.add_argument("--domain", help="restrict to one domain")
    ap.add_argument("--dry-run", action="store_true", help="estimate size and exit")
    ap.add_argument("--list-models", action="store_true", help="print available models and exit")
    args = ap.parse_args()

    if args.list_models:
        print("\n".join(list_models(api_key())))
        return

    if not args.corpus.exists():
        raise SystemExit(f"{args.corpus} not found — run: python3 -m salience.data.corpus")

    with args.corpus.open(encoding="utf-8") as fh:
        paragraphs = [json.loads(line) for line in fh if line.strip()]
    if args.domain:
        paragraphs = [p for p in paragraphs if p["domain"] == args.domain]
    paragraphs = paragraphs[args.offset : args.offset + args.limit]

    calls = len(paragraphs) * args.samples
    approx_in = sum(len(p["text"]) // 3 + 700 for p in paragraphs) * args.samples
    approx_out = calls * 90
    print(
        f"{len(paragraphs)} paragraphs x {args.samples} samples = {calls:,} calls "
        f"(~{approx_in / 1e6:.2f}M in, ~{approx_out / 1e6:.2f}M out) on {args.model}"
    )
    if args.dry_run:
        return

    client = api_key()
    cache = Cache(CACHE)
    print(f"cache holds {len(cache):,} responses")

    results: list[dict] = []
    failures = 0
    done = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(label_one, client, cache, args.model, p, args.samples): p
            for p in paragraphs
        }
        for fut in as_completed(futures):
            done += 1
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001 - one bad paragraph must not kill the run
                failures += 1
                print(f"\n  ! {futures[fut]['id']}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            if rec is None:
                failures += 1
                continue
            results.append(rec)
            if done % 10 == 0 or done == len(paragraphs):
                rate = done / max(1e-9, time.time() - start)
                eta = (len(paragraphs) - done) / max(1e-9, rate)
                print(f"  {done}/{len(paragraphs)}  {rate:.1f}/s  eta {eta / 60:.1f}m", end="\r")

    cache.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in sorted(results, key=lambda r: r["id"]):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    marked = sum(sum(1 for t in r["targets"] if t >= 0.5) for r in results)
    words = sum(r["word_count"] for r in results)
    print(f"\nwrote {len(results)} labeled paragraphs to {args.out} ({failures} failed)")
    print(f"highlight rate at threshold 0.5: {marked / max(1, words):.1%}")


if __name__ == "__main__":
    main()
