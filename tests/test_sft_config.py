import sys
from types import SimpleNamespace

import pytest

from trainer.sft.cli import build_parser, config_from_args, resolve_gpu_count
from trainer.sft.config import (
    MODEL_ORDER,
    MODEL_PROFILES,
    ModelConfig,
    RunConfig,
    TrainingConfig,
    model_config_for,
    run_config_for_model,
)
from trainer.sft.training import build_trainer
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
    assert config.training.num_train_epochs == 1.0
    assert config.training.max_steps == -1
    assert config.training.eval_steps == 38
    assert config.training.evaluations_per_epoch == 4
    assert config.training.save_steps == 50
    assert config.training.per_device_train_batch_size == 1
    assert config.training.gradient_accumulation_steps == 8
    assert config.training.effective_batch_size(world_size=1) == 8
    assert config.tracking.project == "Sinergi-training"
    assert config.tracking.upload_adapter is True
    assert config.tracking.upload_checkpoints is True
    assert config.tracking.restore_checkpoints is True


def test_cli_help_documents_gpu_auto_detection_and_batch_formula() -> None:
    help_text = build_parser().format_help()

    assert "[--modelname" in help_text
    assert "all CUDA GPUs visible" in help_text
    assert "Effective batch" in help_text
    assert "automatically left-truncated" in help_text
    assert "Override auto-detection" in help_text
    assert "python -m trainer.sft.run_all" in help_text


def test_gpu_count_auto_detection_uses_all_visible_devices(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(device_count=lambda: 3)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_gpu_count(None) == 3
    assert resolve_gpu_count(2) == 2


def test_cli_accepts_training_overrides() -> None:
    args = build_parser().parse_args(
        [
            "--max-steps",
            "7",
            "--eval-steps",
            "2",
            "--save-steps",
            "5",
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
            "--wandb-checkpoint-artifact-name",
            "stage-1-checkpoint",
        ]
    )

    assert args.max_steps == 7
    assert args.eval_steps == 2
    assert args.save_steps == 5
    assert args.gpu_count == 2
    assert args.per_device_batch_size == 1
    assert args.gradient_accumulation_steps == 4
    assert args.allow_non_a100 is True
    assert args.dataset_config == "sft"
    assert args.wandb_project == "court-extractor"
    assert args.wandb_artifact_name == "stage-1-lora"
    assert args.wandb_checkpoint_artifact_name == "stage-1-checkpoint"

    config = config_from_args(args)
    assert config.training.max_steps == 7
    assert config.training.num_train_epochs == 1.0
    assert config.training.eval_steps == 2
    assert config.training.save_steps == 5
    assert config.model.required_gpu_count == 2
    assert config.model.require_distributed_launch is True
    assert config.tracking.project == "court-extractor"
    assert config.tracking.artifact_name == "stage-1-lora"
    assert config.tracking.checkpoint_artifact_name == "stage-1-checkpoint"
    assert config.tracking.upload_adapter is True
    assert config.tracking.upload_checkpoints is True


def test_modelname_alias_selects_one_full_training_profile() -> None:
    args = build_parser().parse_args(["--modelname", "deepseek"])
    config = config_from_args(args)

    assert args.model == "deepseek"
    assert config.model.model_name == (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    assert config.training.max_steps == -1
    assert config.training.num_train_epochs == 1.0
    assert config.tracking.upload_adapter is True


def test_no_wandb_upload_disables_checkpoint_and_lora_artifacts() -> None:
    config = config_from_args(
        build_parser().parse_args(["--no-wandb-upload"])
    )

    assert config.tracking.upload_adapter is False
    assert config.tracking.upload_checkpoints is False
    assert config.tracking.restore_checkpoints is True


def test_no_wandb_resume_disables_only_remote_checkpoint_restore() -> None:
    config = config_from_args(
        build_parser().parse_args(["--no-wandb-resume"])
    )

    assert config.tracking.restore_checkpoints is False
    assert config.tracking.upload_adapter is True
    assert config.tracking.upload_checkpoints is True


def test_trainer_saves_resumable_state_every_configured_steps(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeFastLanguageModel:
        @staticmethod
        def for_training(model) -> None:
            captured["training_model"] = model

    class FakeSFTTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"] = kwargs
            self.train_dataset = kwargs["train_dataset"]
            self.eval_dataset = kwargs["eval_dataset"]

    def fake_sft_config(**kwargs):
        captured["sft_config"] = kwargs
        return kwargs

    monkeypatch.setitem(
        sys.modules,
        "trl",
        SimpleNamespace(SFTConfig=fake_sft_config, SFTTrainer=FakeSFTTrainer),
    )
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(FastLanguageModel=FakeFastLanguageModel),
    )
    monkeypatch.setitem(
        sys.modules,
        "unsloth.chat_templates",
        SimpleNamespace(train_on_responses_only=lambda trainer, **kwargs: trainer),
    )

    build_trainer(
        object(),
        SimpleNamespace(),
        [1],
        [2],
        max_length=256,
        config=TrainingConfig(save_steps=5, eval_steps=10),
        model_config=ModelConfig(),
    )

    sft_config = captured["sft_config"]
    assert isinstance(sft_config, dict)
    assert sft_config["save_strategy"] == "steps"
    assert sft_config["save_steps"] == 5
    assert sft_config["save_only_model"] is False
    assert sft_config["num_train_epochs"] == 1.0
    assert sft_config["max_steps"] == -1
    assert sft_config["max_length"] == 256
    assert captured["trainer_kwargs"]["processing_class"].truncation_side == "left"


def test_supported_model_profiles_match_hub_architectures_and_modalities() -> None:
    assert MODEL_ORDER == ("qwen", "gemma", "deepseek")
    assert MODEL_PROFILES["qwen"].input_modalities == (
        "text", "image", "video"
    )
    gemma = MODEL_PROFILES["gemma"]
    assert gemma.architecture == "Gemma4ForConditionalGeneration"
    assert gemma.input_modalities == ("text", "image", "audio", "video")
    assert gemma.lora_kind == "multimodal_language_only"
    assert "vision_tower" in gemma.non_text_module_fragments
    assert "audio_tower" in gemma.non_text_module_fragments
    deepseek = model_config_for(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    assert deepseek.architecture == "Qwen2ForCausalLM"
    assert deepseek.input_modalities == ("text",)
    assert deepseek.instruction_part == "<｜User｜>"
    assert deepseek.response_part == "<｜Assistant｜>"


def test_each_model_uses_isolated_local_and_wandb_names() -> None:
    configs = [
        run_config_for_model(MODEL_PROFILES[key]) for key in MODEL_ORDER
    ]

    assert len({config.training.output_dir for config in configs}) == 3
    assert len({config.training.adapter_dir for config in configs}) == 3
    assert len({config.data.cache_dir for config in configs}) == 3
    assert len({config.tracking.artifact_name for config in configs}) == 3


def test_cli_selects_gemma_profile_and_derived_paths() -> None:
    config = config_from_args(
        build_parser().parse_args(["--model", "gemma"])
    )

    assert config.model.model_name == "google/gemma-4-E2B-it"
    assert config.model.loader_kind == "fast_model"
    assert config.training.output_dir.as_posix().endswith(
        "gemma-4-e2b/checkpoints"
    )
    assert config.tracking.artifact_name == "gemma-4-e2b-lora"


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


def test_automatic_eval_interval_targets_four_evaluations_per_epoch() -> None:
    training = TrainingConfig(eval_steps=None)

    assert training.optimizer_steps_per_epoch(2_468, world_size=2) == 155
    assert training.resolved_eval_steps(2_468, world_size=2) == 38
    assert training.optimizer_steps_per_epoch(14_766, world_size=2) == 923
    assert training.resolved_eval_steps(14_766, world_size=2) == 230
    assert TrainingConfig().resolved_eval_steps(14_766, world_size=2) == 38
    assert TrainingConfig(eval_steps=75).resolved_eval_steps(14_766, 2) == 75


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
