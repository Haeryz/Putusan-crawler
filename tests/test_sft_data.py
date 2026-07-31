from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from trainer.sft.config import DataConfig, MODEL_PROFILES
from trainer.sft.data import (
    choose_max_length,
    format_messages,
    load_splits,
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


def test_load_splits_uses_hugging_face_sft_config(monkeypatch) -> None:
    calls = []

    def fake_load_dataset(repository, subset, *, split):
        calls.append((repository, subset, split))
        return [f"{split}-row"]

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    train, validation = load_splits(DataConfig())

    assert train == ["train-row"]
    assert validation == ["validation-row"]
    assert calls == [
        ("Haeryz/putusan-structured-extraction", "sft", "train"),
        ("Haeryz/putusan-structured-extraction", "sft", "validation"),
    ]


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

    assert format_messages(examples, FakeTokenizer(), MODEL_PROFILES["qwen"]) == {
        "text": ["one", "extract|two"]
    }


def test_format_messages_rejects_multimodal_content_blocks() -> None:
    examples = {
        "messages": [[{
            "role": "user",
            "content": [{"type": "image", "url": "example.png"}],
        }]]
    }

    with pytest.raises(ValueError, match="text-only"):
        format_messages(examples, FakeTokenizer(), MODEL_PROFILES["gemma"])


def test_gemma_formatting_removes_processor_bos() -> None:
    class BosTokenizer(FakeTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return "<bos><|turn>user\nhello<|turn|>\n"

    result = format_messages(
        {"messages": [[{"role": "user", "content": "hello"}]]},
        BosTokenizer(),
        MODEL_PROFILES["gemma"],
    )

    assert result["text"] == ["<|turn>user\nhello<|turn|>\n"]


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


def test_choose_max_length_caps_and_reports_truncated_rows() -> None:
    profile = choose_max_length(
        [100, 200, 300], context_cap=128, multiple=64
    )

    assert profile.max_length == 128
    assert profile.coverage == pytest.approx(1 / 3)
