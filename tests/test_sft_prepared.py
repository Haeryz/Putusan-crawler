from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from trainer.sft.config import MODEL_PROFILES, run_config_for_model
from trainer.sft.prepared import prepared_manifest, validate_prepared_manifest
from trainer.sft.config import TrainingConfig
from trainer.sft.training import build_trainer


def test_prepared_manifest_binds_model_dataset_slicing_and_context() -> None:
    config = run_config_for_model(MODEL_PROFILES["gemma"])
    manifest = prepared_manifest(
        config,
        max_length=8_192,
        train_rows=76_508,
        validation_rows=9_176,
        length_profile={
            "p50": 500,
            "p90": 4_000,
            "p95": 8_700,
            "maximum": 100_000,
            "max_length": 8_192,
            "coverage": 0.945,
        },
    )

    validate_prepared_manifest(manifest, config)
    assert manifest["base_model"] == "google/gemma-4-E2B-it"
    assert manifest["section_slicing"] is True
    assert manifest["max_length"] == 8_192


def test_prepared_manifest_rejects_wrong_model() -> None:
    gemma = run_config_for_model(MODEL_PROFILES["gemma"])
    deepseek = run_config_for_model(MODEL_PROFILES["deepseek"])
    manifest = prepared_manifest(
        gemma, 8_192, 1, 1,
        {"p50": 1, "p90": 1, "p95": 1, "maximum": 1,
         "max_length": 8_192, "coverage": 1.0},
    )

    with pytest.raises(RuntimeError, match="incompatible"):
        validate_prepared_manifest(manifest, deepseek)


def test_trainer_skips_tokenization_and_masking_for_prepared_ids(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class PreparedDataset:
        column_names = ["input_ids", "labels"]

        def __len__(self):
            return 2

    class Tokenizer:
        truncation_side = "left"

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs
            self.train_dataset = kwargs["train_dataset"]
            self.eval_dataset = kwargs["eval_dataset"]

    class FakeCollator:
        def __init__(self, **kwargs):
            captured["collator"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "trl",
        SimpleNamespace(
            SFTConfig=lambda **kwargs: captured.setdefault("config", kwargs),
            SFTTrainer=FakeTrainer,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(DataCollatorForSeq2Seq=FakeCollator),
    )
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(
            FastLanguageModel=SimpleNamespace(for_training=lambda model: None)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "unsloth.chat_templates",
        SimpleNamespace(
            train_on_responses_only=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("prepared labels must not be remasked")
            )
        ),
    )

    build_trainer(
        object(),
        Tokenizer(),
        PreparedDataset(),
        PreparedDataset(),
        max_length=8_192,
        config=TrainingConfig(eval_steps=1),
        model_config=MODEL_PROFILES["gemma"],
    )

    assert captured["config"]["dataset_kwargs"] == {
        "skip_prepare_dataset": True
    }
    assert "data_collator" in captured["trainer"]
