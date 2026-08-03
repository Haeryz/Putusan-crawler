from __future__ import annotations

from argparse import Namespace
import sys
from types import SimpleNamespace

import pytest

from trainer.sft.config import MODEL_PROFILES
from trainer.sft.preflight import infer_modalities
from trainer.sft.run_all import build_training_command
from trainer.sft.transformer import load_base_model, verify_non_text_modules_frozen


def runner_args(**overrides) -> Namespace:
    values = {
        "dataset": "Haeryz/putusan-structured-extraction",
        "dataset_config": "sft",
        "num_train_epochs": 1.0,
        "half_epoch": False,
        "max_steps": None,
        "eval_steps": 38,
        "evaluations_per_epoch": None,
        "no_eval": False,
        "save_steps": 5,
        "gpu_count": 1,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "wandb_project": "Sinergi-training",
        "wandb_entity": None,
        "wandb_run_prefix": "trial",
        "allow_non_a100": False,
        "no_wandb_upload": False,
        "no_wandb_resume": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_sequential_commands_select_each_profile_and_unique_run_name() -> None:
    commands = {
        key: build_training_command(key, runner_args())
        for key in ("qwen", "gemma", "deepseek")
    }

    for key, command in commands.items():
        assert command[command.index("--model") + 1] == key
        assert command[command.index("--num-train-epochs") + 1] == "1.0"
        assert "--max-steps" not in command
    assert commands["qwen"][commands["qwen"].index("--wandb-run-name") + 1] == (
        "trial-qwen3-5-4b"
    )
    assert commands["gemma"][commands["gemma"].index("--wandb-run-name") + 1] == (
        "trial-gemma-4-e2b"
    )
    assert (
        commands["deepseek"][
            commands["deepseek"].index("--wandb-run-name") + 1
        ]
        == "trial-deepseek-r1-distill-qwen-1-5b"
    )


def test_sequential_command_passes_half_epoch_preset() -> None:
    command = build_training_command("qwen", runner_args(half_epoch=True))

    assert "--half-epoch" in command
    assert "--num-train-epochs" not in command


def test_sequential_command_can_disable_evaluation() -> None:
    command = build_training_command(
        "deepseek", runner_args(max_steps=300, no_eval=True)
    )

    assert command[command.index("--max-steps") + 1] == "300"
    assert "--no-eval" in command
    assert "--eval-steps" not in command


def test_hub_config_modality_inference_distinguishes_three_architectures() -> None:
    assert infer_modalities({"text_config": {}, "vision_config": {}}) == {
        "text", "image", "video"
    }
    assert infer_modalities(
        {"text_config": {}, "vision_config": {}, "audio_config": {}}
    ) == {"text", "image", "audio", "video"}
    assert infer_modalities({"model_type": "qwen2"}) == {"text"}


def test_gemma_non_text_trainable_parameter_is_rejected() -> None:
    model = SimpleNamespace(
        named_parameters=lambda: iter([
            (
                "model.vision_tower.encoder.layers.0.lora_A.weight",
                SimpleNamespace(requires_grad=True),
            )
        ])
    )

    with pytest.raises(RuntimeError, match="trainable non-text"):
        verify_non_text_modules_frozen(model, MODEL_PROFILES["gemma"])


def test_language_only_trainable_parameters_are_allowed() -> None:
    model = SimpleNamespace(
        named_parameters=lambda: iter([
            (
                "model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
                SimpleNamespace(requires_grad=True),
            ),
            (
                "model.audio_tower.encoder.weight",
                SimpleNamespace(requires_grad=False),
            ),
        ])
    )

    verify_non_text_modules_frozen(model, MODEL_PROFILES["gemma"])


def test_gemma_uses_fast_model_without_qwen_text_only_flag(monkeypatch) -> None:
    calls: dict[str, dict[str, object]] = {}

    class FakeFastModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["fast_model"] = kwargs
            return object(), object()

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["language_model"] = kwargs
            return object(), object()

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(bfloat16="bf16")
    )
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(
            FastModel=FakeFastModel,
            FastLanguageModel=FakeFastLanguageModel,
        ),
    )

    load_base_model(MODEL_PROFILES["gemma"])

    assert calls["fast_model"]["model_name"] == "google/gemma-4-E2B-it"
    assert "text_only" not in calls["fast_model"]
    assert "language_model" not in calls


def test_qwen_loader_explicitly_enables_text_only(monkeypatch) -> None:
    calls: dict[str, dict[str, object]] = {}

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["language_model"] = kwargs
            return object(), object()

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(bfloat16="bf16")
    )
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(
            FastModel=object(),
            FastLanguageModel=FakeFastLanguageModel,
        ),
    )

    load_base_model(MODEL_PROFILES["qwen"])

    assert calls["language_model"]["text_only"] is True
