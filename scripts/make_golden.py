#!/usr/bin/env python3
"""Generate the cross-language golden fixtures.

Writes two files:

  shared/golden/inputs.jsonl    language-neutral input strings, one JSON string per line
  shared/golden/features.jsonl  the Python implementation's output for each input

pytest regenerates and compares against the committed features.jsonl, catching Python-side
regressions. vitest computes the TypeScript output for the same inputs and compares against
the same file, catching divergence between the two implementations.

Run:  python scripts/make_golden.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from salience.features import extract_features  # noqa: E402

GOLDEN_DIR = ROOT / "shared" / "golden"

# Hand-picked cases, each targeting a specific rule or a place the two languages
# are likely to disagree.
CURATED = [
    "",
    " ",
    "\n",
    "\n\n\n",
    "a",
    "A",
    "1",
    "hello world",
    "Hello, World!",
    "The quick brown fox jumps over the lazy dog.",
    # Internal punctuation
    "it's",
    "don't stop",
    "state-of-the-art",
    "well- dangling",
    "-leading",
    "a--b",
    "a-b-c-d",
    # Numeric connectors (spec v2): digits keep . and , ; letters do not
    "grew 1.6 kilometers per year",
    "about 10,000 people attended",
    "pi is 3.14159 exactly",
    "Section 1. The next part begins.",
    "U.S. policy shifted, e.g. on tariffs.",
    "version 2.0.1 shipped",
    "1,234,567 rows",
    "ends with 5. Then more.",
    "1.a and a.1 and a.b",
    "trailing 7, then text",
    "rock'n'roll",
    "''",
    "--",
    # Dash lookalikes: only U+002D is internal
    "a—b",
    "a–b",
    "a−b",
    # Line endings
    "windows\r\nline",
    "old\rmac",
    "mixed\r\na\rb\nc",
    "trailing\n",
    "\nleading",
    # Paragraphs
    "First para.\n\nSecond para.",
    "One\n\n\n\nFour newlines.",
    "No break\nsingle newline.",
    # Sentence segmentation, including the cases it gets wrong on purpose
    "Dr. Smith went home. He slept.",
    "Wait... what?",
    "He said \"stop.\" Then he left.",
    "Ends with ellipsis…",
    "Really?! Yes.",
    "3.14 is pi. 42 is not.",
    "one. two. three. four.",
    # Casing
    "NASA Apple lower MiXeD",
    "A B C",
    "ALL CAPS HERE",
    "i",
    # Digits
    "1234567890",
    "v2 release 3rd",
    "3.14159",
    "2026-09-15",
    "COVID-19",
    # Unicode
    "CJK 日本語 text",
    "中文分词测试",
    "Ελληνικά κείμενο",
    "Русский текст",
    "العربية نص",
    "עברית טקסט",
    "emoji 👨‍👩‍👧‍👦 family",
    "single 🎉 emoji",
    "flag 🇯🇵 here",
    "skin 👍🏽 tone",
    "zero​width",
    "combining é accent",
    "precomposed é accent",
    "Turkish İstanbul",
    "German STRASSE straße",
    "math 𝕬𝖑𝖕𝖍𝖆 letters",
    "roman Ⅷ numeral",
    "fullwidth Ａ Ｂ Ｃ",
    " nbsp here",
    # Whitespace
    "tabs\tand\tspaces",
    "  leading spaces",
    "trailing spaces  ",
    "many      spaces",
    "\t\t\t",
    "mixed \t \t mix",
    # Programming-ish text, since work comms contain it
    "__dunder__ name",
    "snake_case_word",
    "camelCaseWord",
    "call(arg, other)",
    "a.b.c.d",
    "https://example.com/path?q=1",
    "user@example.com",
    "#hashtag @mention",
    "$100 and 50%",
    "C++ and C#",
    # Repetition, exercising the document-frequency buckets
    "cat cat cat cat cat",
    "the " * 20,
    "word " * 70,
    # Long single token
    "a" * 60,
    "x" * 200,
    # Realistic prose
    "The James Webb Space Telescope has detected carbon dioxide in the atmosphere of a "
    "planet orbiting another star, the first unambiguous detection of the gas outside our "
    "solar system.",
    "Hey team, quick update: the migration is done, staging looks clean, and I'll cut the "
    "release branch tomorrow morning unless anyone objects.",
    "To install, run `npm install` and then `npm run dev`. The server listens on port 5173 "
    "by default.",
]

# Fragments recombined to broaden coverage beyond the curated cases.
FRAGMENTS = [
    "the", "Cat", "NASA", "42", "3.14", "it's", "co-op", "日本", "👍", "é",
    " ", "  ", "\t", "\n", "\n\n", ".", "!", "?", "…", ",", "-", "'", '"', ")", "(",
    "word", "Word", "WORD", "w0rd", "_x_", "a-b", "x" * 25, "​", "—",
]


def generate(count: int) -> list[str]:
    rng = random.Random(1234)
    out = list(CURATED)
    seen = set(out)
    while len(out) < count:
        parts = [rng.choice(FRAGMENTS) for _ in range(rng.randint(1, 12))]
        s = "".join(parts)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def encode(text: str) -> dict:
    return {
        "t": [
            {
                "c": f.cls,
                "s": f.start,
                "e": f.end,
                "r": f.rows,
                "p": [f.sentence_first_word, f.prev_sentence_last_word],
            }
            for f in extract_features(text)
        ]
    }


def main() -> None:
    inputs = generate(2000)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    with (GOLDEN_DIR / "inputs.jsonl").open("w", encoding="utf-8") as fh:
        for s in inputs:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    with (GOLDEN_DIR / "features.jsonl").open("w", encoding="utf-8") as fh:
        for s in inputs:
            fh.write(json.dumps(encode(s), ensure_ascii=False, separators=(",", ":")) + "\n")

    total = sum(len(encode(s)["t"]) for s in inputs)
    print(f"wrote {len(inputs)} inputs, {total} tokens to {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
