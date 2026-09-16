"""Python-side conformance against shared/spec/features.md.

The golden test here guards against Python regressions; ts/src/golden.test.ts guards
against divergence between the two implementations. Both read the same fixtures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from salience.features import (  # noqa: E402
    FLAG_ALL_CAPS,
    FLAG_BEFORE_SENTENCE_END,
    FLAG_CAPITALIZED,
    FLAG_PARAGRAPH_START,
    FLAG_SENTENCE_START,
    NEWLINE,
    PUNCT,
    ROW_DOC_FREQ,
    ROW_FLAGS,
    ROW_LENGTH,
    ROW_SENT_POS,
    ROW_STOPWORD,
    ROW_TOTAL,
    SPACE,
    WORD,
    ascii_lower,
    extract_features,
    normalize_newlines,
    tokenize,
)

GOLDEN = ROOT / "shared" / "golden"


def _read_jsonl(name: str) -> list:
    with (GOLDEN / name).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module")
def golden() -> tuple[list[str], list[dict]]:
    return _read_jsonl("inputs.jsonl"), _read_jsonl("features.jsonl")


def test_golden_fixtures_are_current(golden) -> None:
    """Regenerating the fixtures must be a no-op. If this fails, run scripts/make_golden.py."""
    inputs, expected = golden
    assert len(inputs) == len(expected)

    for i, (text, want) in enumerate(zip(inputs, expected)):
        got = [
            {
                "c": f.cls,
                "s": f.start,
                "e": f.end,
                "r": f.rows,
                "p": [f.sentence_first_word, f.prev_sentence_last_word],
            }
            for f in extract_features(text)
        ]
        assert got == want["t"], f"input[{i}] {text!r}"


def test_every_code_point_belongs_to_exactly_one_token(golden) -> None:
    inputs, _ = golden
    for text in inputs:
        normalized = normalize_newlines(text)
        tokens = tokenize(normalized)
        assert "".join(t.text for t in tokens) == normalized
        for prev, cur in zip(tokens, tokens[1:]):
            assert cur.start == prev.end


def test_rows_are_sorted_and_in_range(golden) -> None:
    inputs, _ = golden
    for text in inputs:
        for f in extract_features(text):
            assert f.rows == sorted(f.rows)
            assert all(0 <= r < ROW_TOTAL for r in f.rows)


def test_ascii_lower_leaves_non_ascii_alone() -> None:
    assert ascii_lower("HeLLo") == "hello"
    assert ascii_lower("İSTANBUL") == "İstanbul"
    assert ascii_lower("STRASSE") == "strasse"
    assert ascii_lower("ÉCOLE") == "École"


def test_internal_punctuation_stays_inside_words() -> None:
    words = [t.text for t in tokenize("it's state-of-the-art") if t.cls == WORD]
    assert words == ["it's", "state-of-the-art"]


def test_dangling_hyphen_splits_off() -> None:
    got = [(t.cls, t.text) for t in tokenize("well- done")]
    assert got == [(WORD, "well"), (PUNCT, "-"), (SPACE, " "), (WORD, "done")]


def test_only_hyphen_minus_is_internal() -> None:
    assert [t.text for t in tokenize("a-b") if t.cls == WORD] == ["a-b"]
    assert [t.text for t in tokenize("a—b") if t.cls == WORD] == ["a", "b"]


def test_one_token_per_newline() -> None:
    assert sum(1 for t in tokenize("a\n\nb") if t.cls == NEWLINE) == 2


def test_non_word_tokens_get_exactly_three_rows() -> None:
    for f in extract_features("hi there."):
        if f.cls != WORD:
            assert len(f.rows) == 3


def test_sentence_and_paragraph_start_flags() -> None:
    words = [f for f in extract_features("One two.\n\nThree four.") if f.cls == WORD]
    has = lambda i, bit: (ROW_FLAGS + bit) in words[i].rows  # noqa: E731

    assert has(0, FLAG_SENTENCE_START)
    assert has(0, FLAG_PARAGRAPH_START)
    assert not has(1, FLAG_SENTENCE_START)
    assert has(2, FLAG_SENTENCE_START)
    assert has(2, FLAG_PARAGRAPH_START)
    assert not has(3, FLAG_PARAGRAPH_START)


def test_before_sentence_end_flag() -> None:
    words = [f for f in extract_features("Alpha beta.") if f.cls == WORD]
    assert (ROW_FLAGS + FLAG_BEFORE_SENTENCE_END) not in words[0].rows
    assert (ROW_FLAGS + FLAG_BEFORE_SENTENCE_END) in words[1].rows


def test_capitalized_and_all_caps_are_exclusive() -> None:
    words = [f for f in extract_features("Apple NASA lower") if f.cls == WORD]
    assert (ROW_FLAGS + FLAG_CAPITALIZED) in words[0].rows
    assert (ROW_FLAGS + FLAG_ALL_CAPS) not in words[0].rows
    assert (ROW_FLAGS + FLAG_ALL_CAPS) in words[1].rows
    assert (ROW_FLAGS + FLAG_CAPITALIZED) not in words[1].rows
    assert (ROW_FLAGS + FLAG_CAPITALIZED) not in words[2].rows


def test_stopword_bucket() -> None:
    words = [f for f in extract_features("the photosynthesis") if f.cls == WORD]
    bucket = lambda f: next(r for r in f.rows if ROW_STOPWORD <= r < ROW_FLAGS) - ROW_STOPWORD  # noqa: E731
    assert bucket(words[0]) == 1
    assert bucket(words[1]) == 0


def test_document_frequency_is_case_insensitive() -> None:
    words = [f for f in extract_features("cat dog cat Cat") if f.cls == WORD]
    freq = lambda f: next(r for r in f.rows if ROW_DOC_FREQ <= r < ROW_TOTAL) - ROW_DOC_FREQ  # noqa: E731
    assert freq(words[0]) == 2
    assert freq(words[1]) == 0
    assert freq(words[3]) == 2


def test_length_bucket_edges() -> None:
    def bucket(word: str) -> int:
        rows = extract_features(word)[0].rows
        return next(r for r in rows if ROW_LENGTH <= r < ROW_SENT_POS) - ROW_LENGTH

    assert bucket("a") == 0
    assert bucket("abcdefgh") == 7
    assert bucket("a" * 9) == 8
    assert bucket("a" * 11) == 9
    assert bucket("a" * 48) == 15


def test_pointers() -> None:
    feats = extract_features("Alpha beta. Gamma delta.")
    assert feats[0].sentence_first_word == 0
    assert feats[0].prev_sentence_last_word == -1
    assert feats[5].sentence_first_word == 5
    assert feats[5].prev_sentence_last_word == 2


def test_empty_input() -> None:
    assert extract_features("") == []


def test_newline_normalization() -> None:
    assert normalize_newlines("a\r\nb\rc\nd") == "a\nb\nc\nd"
