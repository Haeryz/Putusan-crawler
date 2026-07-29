import sys
from types import SimpleNamespace

import pytest

from trainer.sft.cli import build_parser, config_from_args
from trainer.sft.config import RunConfig
from trainer.sft.transformer import (
    validate_distributed_launch,
    validate_hardware,
)


def test_short_single_a100_experiment_defaults_are_explicit() -> None:
    config = RunConfig()

    assert config.model.model_name == "Qwen/Qwen3.5-4B"
    assert config.model.max_seq_length == 49_152
    assert config.model.required_gpu_count == 1
    assert config.model.require_distributed_launch is False
    assert config.model.minimum_vram_gb == 78.0
    assert config.data.subset == "sft"
    assert config.training.max_steps == 30
    assert config.training.eval_steps == 10
    assert config.training.per_device_train_batch_size == 1
    assert config.training.gradient_accumulation_steps == 8
    assert config.training.effective_batch_size(world_size=1) == 8
    assert config.tracking.project == "putusan-sft"
    assert config.tracking.upload_adapter is True


def test_cli_accepts_training_overrides() -> None:
    args = build_parser().parse_args(
        [
            "--max-steps",
            "7",
            "--eval-steps",
            "2",
            "--gpu-count",
            "2",
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
    assert args.eval_steps == 2
    assert args.gpu_count == 2
    assert args.per_device_batch_size == 1
    assert args.gradient_accumulation_steps == 4
    assert args.allow_non_a100 is True
    assert args.dataset_config == "sft"
    assert args.wandb_project == "court-extractor"
    assert args.wandb_artifact_name == "stage-1-lora"

    config = config_from_args(args)
    assert config.training.max_steps == 7
    assert config.training.eval_steps == 2
    assert config.model.required_gpu_count == 2
    assert config.model.require_distributed_launch is True
    assert config.tracking.project == "court-extractor"
    assert config.tracking.artifact_name == "stage-1-lora"
    assert config.tracking.upload_adapter is True


def test_two_worker_torchrun_environment_is_required_for_two_gpu_override() -> None:
    config = config_from_args(build_parser().parse_args(["--gpu-count", "2"]))

    assert validate_distributed_launch(
        config.model, {"WORLD_SIZE": "2"}
    ) == 2
    with pytest.raises(RuntimeError, match="torchrun"):
        validate_distributed_launch(config.model, {"WORLD_SIZE": "1"})


def test_effective_batch_rejects_invalid_world_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        RunConfig().training.effective_batch_size(0)


def test_hardware_profile_checks_single_default_gpu(
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
    monkeypatch.setenv("WORLD_SIZE", "1")

    devices = validate_hardware(RunConfig().model)

    assert len(devices) == 1
    assert all(vram_gb == 80.0 for _, vram_gb in devices)
    assert selected == []
