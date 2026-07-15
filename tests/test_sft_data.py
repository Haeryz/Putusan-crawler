from __future__ import annotations

import pytest

from trainer.sft.data import (
    choose_max_length,
    format_messages,
    measure_token_lengths,
)


class FakeTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt
    ):
        assert tokenize is False
        assert add_generation_prompt is False
        return "|".join(message["content"] for message in messages)

    def __call__(self, texts, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [text.split() for text in texts]}


def test_format_messages_renders_each_conversation() -> None:
    examples = {
        "messages": [
            [{"role": "user", "content": "one"}],
            [
                {"role": "system", "content": "extract"},
                {"role": "user", "content": "two"},
            ],
        ]
    }

    assert format_messages(examples, FakeTokenizer()) == {
        "text": ["one", "extract|two"]
    }


def test_measure_token_lengths_batches_without_special_tokens() -> None:
    lengths = measure_token_lengths(
        ["one two", "three", "four five six"],
        FakeTokenizer(),
        batch_size=2,
    )

    assert lengths == [2, 1, 3]


def test_choose_max_length_rounds_up_and_reports_coverage() -> None:
    profile = choose_max_length(
        [10, 100, 200, 300], context_cap=512, multiple=64
    )

    assert profile.max_length == 320
    assert profile.maximum == 300
    assert profile.coverage == 1.0


def test_choose_max_length_rejects_context_cap_below_percentile() -> None:
    with pytest.raises(ValueError, match="exceeds context cap"):
        choose_max_length([100, 200, 300], context_cap=128, multiple=64)

