import sys
from types import SimpleNamespace

import pytest

from trainer.sft.cli import build_parser, config_from_args
from trainer.sft.config import RunConfig
from trainer.sft.transformer import (
    validate_distributed_launch,
    validate_hardware,
)


def test_two_a100_80gb_defaults_are_explicit() -> None:
    config = RunConfig()

    assert config.model.model_name == "Qwen/Qwen3.5-4B"
    assert config.model.max_seq_length == 49_152
    assert config.model.required_gpu_count == 2
    assert config.model.minimum_vram_gb == 78.0
    assert config.data.subset == "sft_sections"
    assert config.training.max_steps == 100
    assert config.training.per_device_train_batch_size == 2
    assert config.training.gradient_accumulation_steps == 2
    assert config.training.effective_batch_size(world_size=2) == 8
    assert config.tracking.project == "putusan-sft"
    assert config.tracking.upload_adapter is True


def test_cli_accepts_training_overrides() -> None:
    args = build_parser().parse_args(
        [
            "--max-steps",
            "7",
            "--per-device-batch-size",
            "1",
            "--gradient-accumulation-steps",
            "4",
            "--allow-non-a100",
            "--wandb-project",
            "court-extractor",
            "--wandb-artifact-name",
            "stage-1-lora",
        ]
    )

    assert args.max_steps == 7
    assert args.per_device_batch_size == 1
    assert args.gradient_accumulation_steps == 4
    assert args.allow_non_a100 is True
    assert args.wandb_project == "court-extractor"
    assert args.wandb_artifact_name == "stage-1-lora"

    config = config_from_args(args)
    assert config.training.max_steps == 7
    assert config.tracking.project == "court-extractor"
    assert config.tracking.artifact_name == "stage-1-lora"
    assert config.tracking.upload_adapter is True


def test_two_worker_torchrun_environment_is_required() -> None:
    config = RunConfig()

    assert validate_distributed_launch(
        config.model, {"WORLD_SIZE": "2"}
    ) == 2
    with pytest.raises(RuntimeError, match="torchrun"):
        validate_distributed_launch(config.model, {"WORLD_SIZE": "1"})


def test_effective_batch_rejects_invalid_world_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        RunConfig().training.effective_batch_size(0)


def test_hardware_profile_checks_both_gpus_and_selects_local_rank(
    monkeypatch,
) -> None:
    selected: list[int] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def get_device_name(index: int) -> str:
            return f"NVIDIA A100-SXM4-80GB #{index}"

        @staticmethod
        def get_device_properties(index: int):
            return SimpleNamespace(total_memory=80 * 1024**3)

        @staticmethod
        def set_device(index: int) -> None:
            selected.append(index)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda))
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    devices = validate_hardware(RunConfig().model)

    assert len(devices) == 2
    assert all(vram_gb == 80.0 for _, vram_gb in devices)
    assert selected == [1]
