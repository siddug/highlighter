"""Tests for the oracle's prompt construction and response parsing.

No network and no API key: these cover the parts that fail silently and corrupt labels.
"""

from __future__ import annotations

import pytest

from salience.oracle.prompt import (
    build_user_message,
    parse_response,
    render_highlights,
    to_soft_targets,
    word_tokens,
)

PARAGRAPH = "The reactor achieved criticality on Tuesday. Engineers did not report any fault."


def test_word_tokens_skips_punctuation_and_space() -> None:
    surfaces = [s for _, s in word_tokens(PARAGRAPH)]
    assert surfaces == [
        "The", "reactor", "achieved", "criticality", "on", "Tuesday",
        "Engineers", "did", "not", "report", "any", "fault",
    ]


def test_build_user_message_indexes_every_word() -> None:
    user, count = build_user_message(PARAGRAPH)
    assert count == 12
    assert "0:The" in user
    assert "11:fault" in user
    assert PARAGRAPH in user


def test_build_user_message_on_empty_input() -> None:
    user, count = build_user_message("")
    assert count == 0
    assert isinstance(user, str)


class TestParseResponse:
    def test_plain_array(self) -> None:
        assert parse_response("[1, 3, 5]", 12) == [1, 3, 5]

    def test_code_fence(self) -> None:
        assert parse_response("```json\n[0, 2]\n```", 12) == [0, 2]

    def test_leading_prose(self) -> None:
        assert parse_response("Here are the indices:\n[4, 6, 8]", 12) == [4, 6, 8]

    def test_sorts_and_deduplicates(self) -> None:
        assert parse_response("[5, 1, 5, 3, 1]", 12) == [1, 3, 5]

    def test_drops_out_of_range(self) -> None:
        # A lossy label beats discarding an otherwise good annotation.
        assert parse_response("[0, 11, 12, 99, -1]", 12) == [0, 11]

    def test_ignores_non_integers(self) -> None:
        assert parse_response('[1, "two", 3.5, null, 4]', 12) == [1, 4]

    def test_empty_array_is_valid(self) -> None:
        assert parse_response("[]", 12) == []

    def test_prefers_the_longest_array(self) -> None:
        chatty = "For example [0]. My answer: [1, 2, 3, 4]"
        assert parse_response(chatty, 12) == [1, 2, 3, 4]

    def test_raises_when_no_array(self) -> None:
        with pytest.raises(ValueError):
            parse_response("I cannot help with that.", 12)

    def test_raises_on_unparsable_array(self) -> None:
        with pytest.raises(ValueError):
            parse_response("[this is not json}", 12)


class TestSoftTargets:
    def test_unanimous_words_reach_one(self) -> None:
        assert to_soft_targets([[0, 2], [0, 2], [0, 2]], 3) == [1.0, 0.0, 1.0]

    def test_partial_agreement_is_fractional(self) -> None:
        targets = to_soft_targets([[0], [0], [1], [0]], 2)
        assert targets == [0.75, 0.25]

    def test_no_samples_yields_zeros(self) -> None:
        assert to_soft_targets([], 3) == [0.0, 0.0, 0.0]

    def test_ignores_indices_past_the_end(self) -> None:
        assert to_soft_targets([[0, 9]], 2) == [1.0, 0.0]

    def test_targets_are_within_unit_interval(self) -> None:
        targets = to_soft_targets([[0, 1], [1], [1, 0], [0]], 2)
        assert all(0.0 <= t <= 1.0 for t in targets)


def test_render_highlights_preserves_order() -> None:
    assert render_highlights(PARAGRAPH, [1, 3, 8, 11]) == "reactor criticality. not fault."


def test_render_highlights_groups_by_sentence() -> None:
    # A full stop marks the boundary so "criticality" and "not" do not read as one claim.
    assert render_highlights(PARAGRAPH, [3, 8]) == "criticality. not."


def test_render_highlights_within_one_sentence_has_no_internal_stop() -> None:
    assert render_highlights(PARAGRAPH, [1, 3]) == "reactor criticality."


def test_render_highlights_with_nothing_marked() -> None:
    assert render_highlights(PARAGRAPH, []) == ""
