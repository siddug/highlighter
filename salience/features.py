"""Implementation of shared/spec/features.md v2.

ts/src/features.ts is the other implementation. shared/golden/ proves they agree.
If you change anything here, change the spec first, then change both implementations.

Every line in this file has a counterpart in features.ts. Keep them structurally parallel
even where Python would prefer something more idiomatic — divergence here is expensive and
invisible.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

SPEC_VERSION = "v2"

WORD = 0
SPACE = 1
NEWLINE = 2
PUNCT = 3

SPEC_DIR = Path(__file__).resolve().parents[1] / "shared" / "spec"

# Row map from spec section 4.
ROW_CLASS = 0
ROW_HASH_A = 8
ROW_HASH_B = 264
ROW_SUFFIX = 520
ROW_PREFIX = 584
ROW_STOPWORD = 616
ROW_FLAGS = 632
ROW_LENGTH = 641
ROW_SENT_POS = 657
ROW_DOC_POS = 689
ROW_DOC_FREQ = 721
ROW_TOTAL = 737

FLAG_CAPITALIZED = 0
FLAG_ALL_CAPS = 1
FLAG_HAS_DIGIT = 2
FLAG_ALL_DIGITS = 3
FLAG_HAS_HYPHEN = 4
FLAG_HAS_APOSTROPHE = 5
FLAG_SENTENCE_START = 6
FLAG_PARAGRAPH_START = 7
FLAG_BEFORE_SENTENCE_END = 8

MASK32 = 0xFFFFFFFF

SENTENCE_END_CHARS = {".", "!", "?", "…"}
CLOSING_CHARS = {'"', "'", ")", "]", "}", "»", "”", "’"}

LENGTH_EDGES = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 19, 24, 31, 47)
FREQ_EDGES = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48, 64)


@dataclass
class TokenFeatures:
    cls: int
    start: int
    end: int
    text: str
    rows: list[int] = field(default_factory=list)
    sentence_first_word: int = -1
    prev_sentence_last_word: int = -1


# --- Determinism primitives (spec section 1) --------------------------------


def _is_letter(cp: str) -> bool:
    return unicodedata.category(cp)[0] in ("L", "M")


def _is_digit(cp: str) -> bool:
    return unicodedata.category(cp)[0] == "N"


def _is_alnum(cp: str) -> bool:
    return _is_letter(cp) or _is_digit(cp)


def _is_upper(cp: str) -> bool:
    return unicodedata.category(cp) == "Lu"


def ascii_lower(s: str) -> str:
    """ASCII-only lowercasing. Never use str.lower() — it diverges from JS on some locales."""
    return "".join(chr(o + 32) if 0x41 <= (o := ord(c)) <= 0x5A else c for c in s)


def _fnv(data: bytes, basis: int, prime: int) -> int:
    h = basis & MASK32
    for b in data:
        h = (h ^ b) & MASK32
        h = (h * prime) & MASK32
    return h


def _mix(h: int) -> int:
    h &= MASK32
    h = (h ^ (h >> 16)) & MASK32
    h = (h * 2246822507) & MASK32
    h = (h ^ (h >> 13)) & MASK32
    h = (h * 3266489909) & MASK32
    h = (h ^ (h >> 16)) & MASK32
    return h


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


# --- Stopwords (spec section 3.2) -------------------------------------------


@lru_cache(maxsize=1)
def _stopword_ranks() -> dict[str, int]:
    ranks: dict[str, int] = {}
    rank = 0
    for line in (SPEC_DIR / "stopwords.txt").read_text(encoding="utf-8").split("\n"):
        w = line.strip()
        if w == "" or w.startswith("#"):
            continue
        if w not in ranks:
            ranks[w] = rank
        rank += 1
    return ranks


def _stopword_bucket(lower: str) -> int:
    rank = _stopword_ranks().get(lower)
    return 0 if rank is None else min(15, 1 + rank // 16)


def _bucketize(value: int, edges: tuple[int, ...]) -> int:
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


# --- Tokenization (spec section 2) ------------------------------------------


@dataclass
class RawToken:
    cls: int
    start: int
    end: int
    text: str


def tokenize(normalized: str) -> list[RawToken]:
    """Split into tokens. Concatenating all token texts reproduces the normalized input."""
    cps = list(normalized)
    n = len(cps)
    tokens: list[RawToken] = []
    i = 0

    while i < n:
        ch = cps[i]

        if ch == "\n":
            tokens.append(RawToken(NEWLINE, i, i + 1, "\n"))
            i += 1
        elif ch in (" ", "\t"):
            start = i
            while i < n and cps[i] in (" ", "\t"):
                i += 1
            tokens.append(RawToken(SPACE, start, i, "".join(cps[start:i])))
        elif _is_alnum(ch):
            start = i
            i += 1
            while i < n:
                c = cps[i]
                if _is_alnum(c):
                    i += 1
                elif c in ("'", "-") and i + 1 < n and _is_alnum(cps[i + 1]):
                    # Internal only: the previous code point is alphanumeric by construction.
                    i += 1
                elif c in (".", ",") and i + 1 < n and _is_digit(cps[i + 1]) and _is_digit(cps[i - 1]):
                    # Keep numbers whole: 1.6 and 10,000 are single facts.
                    i += 1
                else:
                    break
            tokens.append(RawToken(WORD, start, i, "".join(cps[start:i])))
        else:
            tokens.append(RawToken(PUNCT, i, i + 1, ch))
            i += 1

    return tokens


def _mark_sentence_ends(tokens: list[RawToken]) -> list[bool]:
    n = len(tokens)
    ends = [False] * n

    for t in range(n):
        tok = tokens[t]

        if tok.cls == NEWLINE:
            ends[t] = True
            continue
        if tok.cls != PUNCT or tok.text not in SENTENCE_END_CHARS:
            continue

        j = t + 1
        while j < n and tokens[j].cls == PUNCT and tokens[j].text in CLOSING_CHARS:
            j += 1
        last_idx = j - 1

        k = j
        while k < n and tokens[k].cls == SPACE:
            k += 1

        if k >= n:
            ends[last_idx] = True
        else:
            nxt = tokens[k]
            if nxt.cls == WORD:
                first = nxt.text[0]
                if _is_upper(first) or _is_digit(first):
                    ends[last_idx] = True

    return ends


# --- Feature extraction -----------------------------------------------------


def extract_features(text: str) -> list[TokenFeatures]:
    """Tokenize `text` and compute the embedding row indices for every token."""
    normalized = normalize_newlines(text)
    tokens = tokenize(normalized)
    n = len(tokens)
    if n == 0:
        return []

    lower = [ascii_lower(t.text) if t.cls == WORD else t.text for t in tokens]

    ends = _mark_sentence_ends(tokens)
    sid = [0] * n
    cur = 0
    for t in range(n):
        sid[t] = cur
        if ends[t]:
            cur += 1

    sent_count: dict[int, int] = {}
    sent_index = [0] * n
    for t in range(n):
        s = sid[t]
        c = sent_count.get(s, 0)
        sent_index[t] = c
        sent_count[s] = c + 1

    first_word_of_sent: dict[int, int] = {}
    last_word_of_sent: dict[int, int] = {}
    for t in range(n):
        if tokens[t].cls != WORD:
            continue
        s = sid[t]
        first_word_of_sent.setdefault(s, t)
        last_word_of_sent[s] = t

    doc_freq: dict[str, int] = {}
    for t in range(n):
        if tokens[t].cls != WORD:
            continue
        doc_freq[lower[t]] = doc_freq.get(lower[t], 0) + 1

    paragraph_start = [False] * n
    pending = True
    consecutive_newlines = 0
    for t in range(n):
        cls = tokens[t].cls
        if cls == NEWLINE:
            consecutive_newlines += 1
            if consecutive_newlines >= 2:
                pending = True
        elif cls == WORD:
            if pending:
                paragraph_start[t] = True
                pending = False
            consecutive_newlines = 0
        elif cls == PUNCT:
            consecutive_newlines = 0

    out: list[TokenFeatures] = []

    for t in range(n):
        tok = tokens[t]
        s = sid[t]
        rows: list[int] = [ROW_CLASS + tok.cls]

        sent_len = sent_count.get(s, 1)
        rows.append(ROW_SENT_POS + min(31, (32 * sent_index[t]) // max(1, sent_len)))
        rows.append(ROW_DOC_POS + min(31, (32 * t) // max(1, n)))

        flags = 0

        if tok.cls == WORD:
            lw = lower[t]
            data = lw.encode("utf-8")
            suffix = data[max(0, len(data) - 3) :]
            prefix = data[: min(3, len(data))]

            rows.append(ROW_HASH_A + (_mix(_fnv(data, 2166136261, 16777619)) & 0xFF))
            rows.append(ROW_HASH_B + (_mix(_fnv(data, 1166136321, 2246822519)) & 0xFF))
            rows.append(ROW_SUFFIX + (_mix(_fnv(suffix, 2166136261, 16777619)) & 0x3F))
            rows.append(ROW_PREFIX + (_mix(_fnv(prefix, 2166136261, 16777619)) & 0x1F))
            rows.append(ROW_STOPWORD + _stopword_bucket(lw))

            cps = list(tok.text)
            rows.append(ROW_LENGTH + _bucketize(len(cps), LENGTH_EDGES))
            rows.append(ROW_DOC_FREQ + _bucketize(doc_freq.get(lw, 1), FREQ_EDGES))

            letters = uppers = digits = non_digits = 0
            for c in cps:
                if _is_letter(c):
                    letters += 1
                    if _is_upper(c):
                        uppers += 1
                if _is_digit(c):
                    digits += 1
                else:
                    non_digits += 1

            all_caps = letters >= 2 and uppers == letters
            if all_caps:
                flags |= 1 << FLAG_ALL_CAPS
            elif _is_upper(cps[0]):
                flags |= 1 << FLAG_CAPITALIZED

            if digits > 0:
                flags |= 1 << FLAG_HAS_DIGIT
            if non_digits == 0:
                flags |= 1 << FLAG_ALL_DIGITS
            if "-" in tok.text:
                flags |= 1 << FLAG_HAS_HYPHEN
            if "'" in tok.text:
                flags |= 1 << FLAG_HAS_APOSTROPHE
            if first_word_of_sent.get(s) == t:
                flags |= 1 << FLAG_SENTENCE_START
            if paragraph_start[t]:
                flags |= 1 << FLAG_PARAGRAPH_START

            k = t + 1
            while k < n and tokens[k].cls == SPACE:
                k += 1
            if k < n and tokens[k].cls == PUNCT and tokens[k].text in SENTENCE_END_CHARS:
                flags |= 1 << FLAG_BEFORE_SENTENCE_END

        for b in range(9):
            if flags & (1 << b):
                rows.append(ROW_FLAGS + b)

        rows.sort()

        out.append(
            TokenFeatures(
                cls=tok.cls,
                start=tok.start,
                end=tok.end,
                text=tok.text,
                rows=rows,
                sentence_first_word=first_word_of_sent.get(s, -1),
                prev_sentence_last_word=last_word_of_sent.get(s - 1, -1) if s > 0 else -1,
            )
        )

    return out
