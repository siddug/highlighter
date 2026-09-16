"""The labeling prompt and its response parser.

gpu-lexer had Shiki: a free, deterministic, infinite oracle that could be asked the type
of any token forever. Prose salience has no such thing, so we manufacture one. This file
is that oracle, and it is the single highest-leverage artifact in the project — the model
can never be better than the labels it learns from.

Words are identified by index into the WORD tokens produced by salience.features, so the
oracle's output aligns exactly with the model's own tokenization. No fuzzy string matching,
no ambiguity when a word repeats.
"""

from __future__ import annotations

import json
import re

from salience.features import WORD, extract_features

# Bump when the prompt text changes; it is part of the cache key, so old responses
# are ignored rather than silently mixed with new ones.
#
# p1 -> p2: p1 stated a "15-30%" target in prose and the oracle ignored it, marking a
# median 40% (max 54%) with huge sample-to-sample spread. p2 computes an explicit integer
# cap and states it in the user turn, and names the function words outright.
#
# p3 was tried and reverted. It said outright that the cap was a ceiling rather than a
# target and lowered the fraction to 0.22; the median rate went *up* (26% -> 28%, max
# 38% -> 50%). The oracle has a natural rate near 26-30% that prompt wording does not
# move, so p2's numbers stand and rate tuning stops here.
#
# This is the right place to stop for a structural reason, not just a pragmatic one. What
# we need from the oracle is the *ranking* of words, not the absolute count: the soft
# targets already separate a unanimous core (16% of words) from padding that each sample
# chooses differently (25% partial), and the model's own highlight rate is set at training
# time by the budget term in the loss. Spending more calls to shave the oracle's rate would
# buy nothing the loss function does not already control.
PROMPT_VERSION = "p2"

# Cap as a fraction of the paragraph's words. Slightly above the ~20% we actually want,
# so the cap binds the runaway cases without forcing the oracle to pad short paragraphs.
BUDGET_FRACTION = 0.25
MIN_BUDGET = 5

SYSTEM = """\
You mark the words in a paragraph that carry its meaning.

A reader will see only the words you mark, in their original order, with everything else \
hidden. From that telegraphic residue they must be able to say what the paragraph claimed: \
who did what, to what, when, and whether it was asserted or denied.

Mark a word when removing it would lose or change a fact:
- the actors and the things acted upon — nouns, named entities, specific terms
- what happened — the main verbs
- quantities, dates, units, measurements, proportions
- negations and reversals — not, no, never, without, failed, denied, rejected, unlike. \
Dropping one of these inverts the claim, which is the most damaging mistake you can make.
- logical hinges where they change the argument — but, because, despite, therefore, however
- modifiers doing real work: a "sharp decline" is a different fact from a "decline"

Never mark these, no matter where they appear: the, a, an, of, to, in, on, at, for, from, \
by, with, is, are, was, were, be, been, being, has, have, had, do, does, did, that, this, \
these, those, it, its, as, and, or.

Also leave unmarked:
- auxiliaries and copulas — "would", "could", "will", "can" — unless the whole claim is \
about possibility or obligation
- hedges and filler — "it is worth noting that", "in order to", "a number of"
- adjectives and adverbs that merely decorate
- any word repeating something you already marked earlier in the same paragraph

You are writing a telegram, and you are paying by the word. Prefer cutting. A reader who \
can still answer "what happened?" from a shorter list did not need the longer one.

Reply with nothing but a JSON array of the indices you mark, in ascending order."""

USER_TEMPLATE = """\
Paragraph:
{text}

Indexed words:
{indexed}

This paragraph has {count} words. Mark AT MOST {budget} of them — fewer if you can. \
Returning more than {budget} indices is a failed answer.

JSON array of indices to mark:"""


def word_tokens(text: str) -> list[tuple[int, str]]:
    """The WORD tokens of `text` as (token_index, surface). Word index is position in this list."""
    return [(i, f.text) for i, f in enumerate(extract_features(text)) if f.cls == WORD]


def budget_for(word_count: int) -> int:
    """How many words the oracle may mark. An explicit integer, because a stated
    percentage range did not bind — see the p1 -> p2 note above."""
    return max(MIN_BUDGET, round(BUDGET_FRACTION * word_count))


def build_user_message(text: str) -> tuple[str, int]:
    """Render the user turn and return it with the number of indexable words."""
    words = word_tokens(text)
    indexed = " ".join(f"{w}:{surface}" for w, (_, surface) in enumerate(words))
    message = USER_TEMPLATE.format(
        text=text,
        indexed=indexed,
        count=len(words),
        budget=budget_for(len(words)),
    )
    return message, len(words)


_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def parse_response(raw: str, word_count: int) -> list[int]:
    """Extract the marked word indices from a model reply.

    Tolerates code fences, leading prose, and trailing commentary. Out-of-range and
    duplicate indices are dropped rather than raising: a slightly lossy label beats
    discarding an otherwise good annotation.

    Raises ValueError only when no array can be found at all, so the caller can retry.
    """
    matches = _ARRAY_RE.findall(raw)
    if not matches:
        raise ValueError(f"no JSON array in response: {raw[:200]!r}")

    # Prefer the longest array; a chatty model sometimes emits "[0]" style examples first.
    for candidate in sorted(matches, key=len, reverse=True):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list):
            continue
        out = sorted({v for v in parsed if isinstance(v, int) and 0 <= v < word_count})
        return out

    raise ValueError(f"no parsable JSON array in response: {raw[:200]!r}")


def to_soft_targets(samples: list[list[int]], word_count: int) -> list[float]:
    """Average several independent annotations into a per-word target in [0,1].

    Sampling the oracle N times at non-zero temperature and averaging does two things:
    it damps the considerable annotation variance on this task, and it turns a binary
    judgement into a graded one. Words every sample marks are unambiguous; words half
    the samples mark are genuinely borderline, and the model should learn that.
    """
    if not samples:
        return [0.0] * word_count
    counts = [0] * word_count
    for sample in samples:
        for i in sample:
            if 0 <= i < word_count:
                counts[i] += 1
    return [c / len(samples) for c in counts]


def render_highlights(text: str, marked: list[int]) -> str:
    """The telegraphic residue a reader would actually see. Used for eyeballing labels."""
    words = word_tokens(text)
    chosen = set(marked)
    return " ".join(surface for w, (_, surface) in enumerate(words) if w in chosen)
