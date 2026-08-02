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
    tokenize_response_only_text,
    truncate_text_preserving_response,
)
from trainer.sft.section_slicing import SECTION_GUIDANCE, slice_row_by_section


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

    train, validation = load_splits(DataConfig(slice_by_section=False))

    assert train == ["train-row"]
    assert validation == ["validation-row"]
    assert calls == [
        ("Haeryz/putusan-structured-extraction", "sft", "train"),
        ("Haeryz/putusan-structured-extraction", "sft", "validation"),
    ]


def test_notebook_section_slicing_emits_one_gold_span_example_per_section() -> None:
    sections = {section: [] for section in SECTION_GUIDANCE}
    sections["judul"] = ["P U T U S A N", "PUTUSAN"]
    row = {
        "id": "case-1",
        "corpus": "Anak",
        "target_json": {"sections": sections},
    }

    children = slice_row_by_section(row, source_row_no=7)

    assert len(children) == 31
    judul = children[0]
    assert judul["id"] == "row-000007::case-1::judul"
    assert judul["sliced_input"] == (
        "<span>\nP U T U S A N\n</span>\n<span>\nPUTUSAN\n</span>"
    )
    assert judul["messages"][1]["content"].endswith(judul["sliced_input"])
    assert judul["messages"][2]["content"] == judul["answer"]
    assert '"judul": ["P U T U S A N", "PUTUSAN"]' in judul["answer"]
    assert children[1]["is_empty"] is True
    assert '"empty_sections": ["nomor_putusan"]' in children[1]["answer"]


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


class CharacterTokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


def test_middle_truncation_preserves_markers_prompt_edges_and_response() -> None:
    config = MODEL_PROFILES["qwen"]
    text = (
        config.instruction_part
        + "BEGIN-"
        + ("x" * 300)
        + "-END"
        + config.response_part
        + "TARGET-ANSWER"
    )

    truncated = truncate_text_preserving_response(
        text, CharacterTokenizer(), config, max_length=160
    )

    assert len(truncated) <= 160
    assert config.instruction_part in truncated
    assert config.response_part in truncated
    assert "BEGIN-" in truncated
    assert "-END" in truncated
    assert truncated.endswith("TARGET-ANSWER")


def test_short_marker_complete_text_is_not_modified() -> None:
    config = MODEL_PROFILES["qwen"]
    text = config.instruction_part + "source" + config.response_part + "answer"

    assert truncate_text_preserving_response(
        text, CharacterTokenizer(), config, max_length=100
    ) == text


def test_precomputed_labels_mask_prompt_and_assistant_marker() -> None:
    config = MODEL_PROFILES["qwen"]
    text = (
        config.instruction_part
        + "source"
        + config.response_part
        + "ANSWER"
    )

    prepared = tokenize_response_only_text(
        text, CharacterTokenizer(), config, max_length=200
    )

    answer_at = text.index("ANSWER")
    assert prepared["input_ids"] == [ord(character) for character in text]
    assert prepared["labels"][:answer_at] == [-100] * answer_at
    assert prepared["labels"][answer_at:] == [ord(c) for c in "ANSWER"]
