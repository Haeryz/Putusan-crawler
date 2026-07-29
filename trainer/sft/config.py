"""Typed configuration for the whole-document SFT workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Qwen/Unsloth model, LoRA, and GPU hardware settings."""

    model_name: str = "Qwen/Qwen3.5-4B"
    max_seq_length: int = 49_152
    load_in_4bit: bool = True
    lora_rank: int = 32
    lora_alpha: int = 32
    require_a100: bool = True
    required_gpu_count: int = 1
    minimum_vram_gb: float = 78.0
    require_distributed_launch: bool = False


@dataclass(frozen=True)
class DataConfig:
    """Dataset and sequence-length measurement settings."""

    repository: str = "Haeryz/putusan-structured-extraction"
    subset: str = "sft"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    length_percentile: int = 95
    length_multiple: int = 256
    tokenization_batch_size: int = 256
    minimum_context_coverage: float = 0.94
    cache_dir: Path = Path("outputs/sft/cache")
    distributed_cache_timeout_seconds: int = 3_600


@dataclass(frozen=True)
class TrainingConfig:
    """TRL settings for the short single-A100 validation experiment."""

    output_dir: Path = Path("outputs/sft/checkpoints")
    adapter_dir: Path = Path("qwen_extractor_sft_lora")
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 5
    max_steps: int = 30
    learning_rate: float = 2e-4
    logging_steps: int = 1
    eval_steps: int = 10
    weight_decay: float = 0.001
    seed: int = 3407
    report_to: str = "wandb"
    minimum_response_retention: float = 0.94
    dataloader_num_workers: int = 4
    dataloader_prefetch_factor: int = 2

    def effective_batch_size(self, world_size: int) -> int:
        """Return the number of examples contributing to each optimizer step."""

        if world_size < 1:
            raise ValueError("world_size must be positive")
        return (
            self.per_device_train_batch_size
            * world_size
            * self.gradient_accumulation_steps
        )


@dataclass(frozen=True)
class TrackingConfig:
    """Weights & Biases run and model-artifact settings."""

    project: str = "putusan-sft"
    entity: str | None = None
    run_name: str | None = None
    artifact_name: str = "qwen-extractor-sft-lora"
    artifact_type: str = "model"
    artifact_aliases: tuple[str, ...] = ("latest",)
    upload_adapter: bool = True
    upload_timeout_seconds: int = 3_600
    length_cache_artifact_name: str = "sft-token-lengths"
    length_cache_artifact_type: str = "dataset"
    reuse_length_cache: bool = True


@dataclass(frozen=True)
class RunConfig:
    """Complete end-to-end run configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
