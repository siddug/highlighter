"""Fetch a mixed-domain corpus of English paragraphs.

Five domains, deliberately different in register, so the model does not overfit to
encyclopedic prose:

    wikipedia   reference prose, dense with named entities
    news        reported prose, inverted pyramid
    essays      argumentative long-form
    email       work communication, the register this tool is really for
    technical   research abstracts, jargon-heavy

Everything comes from the HuggingFace datasets-server over plain HTTP. No auth, no
`datasets` dependency, no local dataset downloads.

Run:  python3 -m salience.data.corpus --per-domain 600
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw"

SERVER = "https://datasets-server.huggingface.co"
MAX_ROWS_PER_REQUEST = 100


@dataclass(frozen=True)
class Source:
    domain: str
    dataset: str
    config: str
    field: str
    split: str = "train"
    # Articles arrive as one blob; some use blank lines between paragraphs, others
    # a single newline.
    split_on: str = r"\n\s*\n"


SOURCES = (
    Source("wikipedia", "wikimedia/wikipedia", "20231101.en", "text"),
    Source("news", "abisee/cnn_dailymail", "3.0.0", "article", split_on=r"\n"),
    Source("essays", "qwedsacf/ivypanda-essays", "default", "TEXT"),
    Source("email", "Yale-LILY/aeslc", "default", "email_body"),
    Source("technical", "CShorten/ML-ArXiv-Papers", "default", "abstract", split_on=r"\n\s*\n"),
)


@dataclass
class Paragraph:
    id: str
    domain: str
    text: str
    words: int


# --- HTTP -------------------------------------------------------------------


def _get(url: str, retries: int = 4) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "message-highlighter/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"GET failed after {retries} tries: {url}") from last


def num_rows(src: Source) -> int:
    q = urllib.parse.urlencode({"dataset": src.dataset, "config": src.config, "split": src.split})
    info = _get(f"{SERVER}/size?{q}")
    return int(info["size"]["config"]["num_rows"])


def fetch_rows(src: Source, offset: int, length: int) -> list[dict]:
    q = urllib.parse.urlencode(
        {
            "dataset": src.dataset,
            "config": src.config,
            "split": src.split,
            "offset": offset,
            "length": min(length, MAX_ROWS_PER_REQUEST),
        }
    )
    return [r["row"] for r in _get(f"{SERVER}/rows?{q}")["rows"]]


# --- Paragraph extraction and filtering -------------------------------------

MIN_WORDS = 40
MAX_WORDS = 130

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SENTENCE_END_RE = re.compile(r"[.!?]")
_URL_RE = re.compile(r"https?://|www\.")
_LIST_LINE_RE = re.compile(r"^\s*([-*+•]|\d+[.)])\s", re.MULTILINE)
# Wikipedia section headers, citation markers, and template residue.
_MARKUP_RE = re.compile(r"==+|\[\d+\]|\{\{|\}\}|\|\s*\w+\s*=")
# Stripped infobox values leave holes: "The area of the district is ." or "Kormilovsky ()".
_EMPTY_SLOT_RE = re.compile(r"\(\s*\)|\s+[.,;]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ")
    # Collapse runs of spaces/tabs but keep newlines, which carry paragraph structure.
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def is_prose(text: str) -> tuple[bool, str]:
    """Reject anything that is not a well-formed paragraph of running prose."""
    words = _WORD_RE.findall(text)
    n = len(words)
    if n < MIN_WORDS:
        return False, "too short"
    if n > MAX_WORDS:
        return False, "too long"
    if len(_SENTENCE_END_RE.findall(text)) < 2:
        return False, "fewer than two sentences"
    if len(_LIST_LINE_RE.findall(text)) >= 2:
        return False, "looks like a list"
    if len(_URL_RE.findall(text)) > 1:
        return False, "link-heavy"
    if _MARKUP_RE.search(text):
        return False, "markup residue"
    if _EMPTY_SLOT_RE.search(text):
        return False, "empty template slot"
    # Running prose is mostly letters and spaces. Tables and code are not.
    letters = sum(c.isalpha() or c.isspace() for c in text)
    if letters / max(1, len(text)) < 0.82:
        return False, "low letter ratio"
    # A paragraph with no lowercase is a headline or a banner.
    if sum(c.islower() for c in text) / max(1, len(text)) < 0.45:
        return False, "shouting"
    return True, ""


def chunk_long(text: str) -> list[str]:
    """Split an over-long block into sentence-aligned chunks under the word cap.

    News articles and research abstracts frequently arrive as single blocks well past
    MAX_WORDS. Discarding them wastes most of the corpus; splitting on sentence
    boundaries keeps the prose coherent and lifts yield by an order of magnitude.
    """
    if len(_WORD_RE.findall(text)) <= MAX_WORDS:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        n = len(_WORD_RE.findall(sentence))
        if current and current_words + n > MAX_WORDS:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def paragraphs_from(src: Source, row: dict) -> list[str]:
    raw = row.get(src.field)
    if not isinstance(raw, str):
        return []
    blocks = (clean(p) for p in re.split(src.split_on, clean(raw)))
    return [chunk for block in blocks for chunk in chunk_long(block)]


def paragraph_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _dedup_key(text: str) -> str:
    """Normalized key so near-identical boilerplate collapses to one entry."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())[:400]


# --- Collection -------------------------------------------------------------


def collect(src: Source, target: int, seed: int, verbose: bool) -> list[Paragraph]:
    rng = random.Random(seed)
    total = num_rows(src)
    out: list[Paragraph] = []
    seen: set[str] = set()
    rejects: dict[str, int] = {}
    attempts = 0

    # Random offsets rather than a prefix scan: the head of these datasets is often
    # unrepresentative (alphabetical, or oldest-first).
    while len(out) < target and attempts < 400:
        attempts += 1
        offset = rng.randrange(0, max(1, total - MAX_ROWS_PER_REQUEST))
        try:
            rows = fetch_rows(src, offset, MAX_ROWS_PER_REQUEST)
        except RuntimeError as exc:
            print(f"  ! {exc}", file=sys.stderr)
            continue

        for row in rows:
            for para in paragraphs_from(src, row):
                ok, why = is_prose(para)
                if not ok:
                    rejects[why] = rejects.get(why, 0) + 1
                    continue
                key = _dedup_key(para)
                if key in seen:
                    rejects["duplicate"] = rejects.get("duplicate", 0) + 1
                    continue
                seen.add(key)
                out.append(
                    Paragraph(
                        id=paragraph_id(para),
                        domain=src.domain,
                        text=para,
                        words=len(_WORD_RE.findall(para)),
                    )
                )
                if len(out) >= target:
                    break
            if len(out) >= target:
                break

        if verbose:
            print(f"  {src.domain}: {len(out)}/{target} after {attempts} requests", end="\r")

    if verbose:
        top = sorted(rejects.items(), key=lambda kv: -kv[1])[:4]
        reasons = ", ".join(f"{k} {v}" for k, v in top)
        print(f"  {src.domain}: {len(out)} kept in {attempts} requests (rejected: {reasons})")

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-domain", type=int, default=600, help="paragraphs per domain")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "paragraphs.jsonl")
    ap.add_argument("--only", help="restrict to one domain, for quick iteration")
    args = ap.parse_args()

    sources = [s for s in SOURCES if args.only in (None, s.domain)]
    if not sources:
        raise SystemExit(f"no domain matching {args.only!r}")

    everything: list[Paragraph] = []
    for src in sources:
        print(f"{src.domain} <- {src.dataset}")
        everything.extend(collect(src, args.per_domain, args.seed, verbose=True))

    rng = random.Random(args.seed)
    rng.shuffle(everything)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for p in everything:
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    by_domain: dict[str, int] = {}
    words = 0
    for p in everything:
        by_domain[p.domain] = by_domain.get(p.domain, 0) + 1
        words += p.words
    print(f"\nwrote {len(everything)} paragraphs ({words:,} words) to {args.out}")
    for d, c in sorted(by_domain.items()):
        print(f"  {d:<10} {c}")


if __name__ == "__main__":
    main()
