"""Typed configuration and supported-model profiles for SFT."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal


LoaderKind = Literal["fast_language_model", "fast_model"]
LoraKind = Literal["language_peft", "multimodal_language_only"]


@dataclass(frozen=True)
class ModelConfig:
    """One model's architecture, text-only policy, LoRA, and hardware settings."""

    profile_name: str = "qwen"
    model_name: str = "Qwen/Qwen3.5-4B"
    architecture: str = "Qwen3_5ForConditionalGeneration"
    input_modalities: tuple[str, ...] = ("text", "image", "video")
    loader_kind: LoaderKind = "fast_language_model"
    lora_kind: LoraKind = "language_peft"
    instruction_part: str = "<|im_start|>user\n"
    response_part: str = "<|im_start|>assistant\n"
    strip_bos_from_formatted_text: bool = False
    require_linear_attention_lora: bool = True
    require_tiled_mlp: bool = True
    non_text_module_fragments: tuple[str, ...] = ("visual", "vision")
    max_seq_length: int = 49_152
    parameter_count_billions: float = 4.0
    non_quantized_parameter_count_billions: float = 0.64
    hidden_size: int = 2_560
    hidden_layers: int = 32
    vocabulary_size: int = 248_320
    load_in_4bit: bool = True
    lora_rank: int = 32
    lora_alpha: int = 32
    require_a100: bool = True
    required_gpu_count: int = 1
    minimum_vram_gb: float = 78.0
    require_distributed_launch: bool = False

    @property
    def slug(self) -> str:
        return {
            "qwen": "qwen3-5-4b",
            "gemma": "gemma-4-e2b",
            "deepseek": "deepseek-r1-distill-qwen-1-5b",
        }[self.profile_name]


MODEL_ORDER: tuple[str, ...] = ("qwen", "gemma", "deepseek")
# Qwen is retained for merge/inference and reproducibility, but its SFT is
# already complete. New sequential jobs train only the two outstanding models.
TRAINING_ORDER: tuple[str, ...] = ("gemma", "deepseek")

MODEL_PROFILES: dict[str, ModelConfig] = {
    "qwen": ModelConfig(),
    "gemma": ModelConfig(
        profile_name="gemma",
        model_name="google/gemma-4-E2B-it",
        architecture="Gemma4ForConditionalGeneration",
        input_modalities=("text", "image", "audio", "video"),
        loader_kind="fast_model",
        lora_kind="multimodal_language_only",
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
        strip_bos_from_formatted_text=True,
        require_linear_attention_lora=False,
        require_tiled_mlp=False,
        non_text_module_fragments=(
            "vision_tower",
            "audio_tower",
            "multi_modal_projector",
            "multimodal_projector",
        ),
        max_seq_length=8_192,
        parameter_count_billions=5.1,
        # PLE tables + shared token embeddings + frozen vision/audio towers.
        non_quantized_parameter_count_billions=3.20,
        hidden_size=1_536,
        hidden_layers=35,
        vocabulary_size=262_144,
    ),
    "deepseek": ModelConfig(
        profile_name="deepseek",
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        architecture="Qwen2ForCausalLM",
        input_modalities=("text",),
        instruction_part="<｜User｜>",
        response_part="<｜Assistant｜>",
        require_linear_attention_lora=False,
        require_tiled_mlp=False,
        non_text_module_fragments=(),
        max_seq_length=8_192,
        parameter_count_billions=1.8,
        # Untied input embedding and LM head are normally kept in BF16.
        non_quantized_parameter_count_billions=0.467,
        hidden_size=1_536,
        hidden_layers=28,
        vocabulary_size=151_936,
    ),
}

_MODEL_ALIASES = {
    key: key for key in MODEL_ORDER
} | {
    config.model_name: key for key, config in MODEL_PROFILES.items()
}


def model_config_for(name_or_repository: str) -> ModelConfig:
    """Resolve a supported profile key or exact Hugging Face repository."""

    try:
        return MODEL_PROFILES[_MODEL_ALIASES[name_or_repository]]
    except KeyError as error:
        supported = ", ".join(
            config.model_name for config in MODEL_PROFILES.values()
        )
        raise ValueError(
            f"Unsupported model {name_or_repository!r}; choose one of: {supported}"
        ) from error


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
    cache_dir: Path = Path("outputs/sft/qwen3-5-4b/cache")
    prepared_dir: Path = Path("outputs/sft/qwen3-5-4b/prepared-dataset")
    distributed_cache_timeout_seconds: int = 3_600
    slice_by_section: bool = False
    section_slicing_version: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    """TRL settings for one complete fine-tuning epoch."""

    output_dir: Path = Path("outputs/sft/qwen3-5-4b/checkpoints")
    adapter_dir: Path = Path("outputs/sft/qwen3-5-4b/lora")
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 5
    num_train_epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    logging_steps: int = 1
    eval_steps: int | None = 38
    evaluations_per_epoch: int = 4
    save_steps: int = 50
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

    def optimizer_steps_per_epoch(
        self, train_rows: int, world_size: int
    ) -> int:
        """Return DDP optimizer updates needed to consume one epoch."""

        if train_rows < 1:
            raise ValueError("train_rows must be positive")
        if world_size < 1:
            raise ValueError("world_size must be positive")
        rows_per_worker = math.ceil(train_rows / world_size)
        micro_batches = math.ceil(
            rows_per_worker / self.per_device_train_batch_size
        )
        return math.ceil(micro_batches / self.gradient_accumulation_steps)

    def resolved_eval_steps(self, train_rows: int, world_size: int) -> int:
        """Use an explicit interval or spread evaluations across each epoch."""

        if self.eval_steps is not None:
            return self.eval_steps
        if self.evaluations_per_epoch < 1:
            raise ValueError("evaluations_per_epoch must be positive")
        steps = self.optimizer_steps_per_epoch(train_rows, world_size)
        # Floor division ensures the Kth evaluation occurs before the epoch's
        # final step when the step count is not exactly divisible by K.
        return max(1, steps // self.evaluations_per_epoch)


@dataclass(frozen=True)
class TrackingConfig:
    """Weights & Biases run and model-artifact settings."""

    project: str = "Sinergi-training"
    entity: str | None = None
    run_name: str | None = None
    artifact_name: str = "qwen3-5-4b-lora"
    artifact_type: str = "model"
    artifact_aliases: tuple[str, ...] = ("latest",)
    upload_adapter: bool = True
    checkpoint_artifact_name: str = "qwen3-5-4b-checkpoint"
    checkpoint_artifact_type: str = "model-checkpoint"
    upload_checkpoints: bool = True
    upload_timeout_seconds: int = 3_600
    restore_checkpoints: bool = True
    restore_timeout_seconds: int = 3_600
    length_cache_artifact_name: str = "qwen3-5-4b-token-lengths"
    length_cache_artifact_type: str = "dataset"
    reuse_length_cache: bool = True
    prepared_dataset_artifact_name: str = "qwen3-5-4b-prepared-sft"
    prepared_dataset_artifact_type: str = "tokenized-dataset"
    reuse_prepared_dataset: bool = True


@dataclass(frozen=True)
class RunConfig:
    """Complete end-to-end run configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


def run_config_for_model(model: ModelConfig) -> RunConfig:
    """Build isolated local and W&B paths for one supported model."""

    slug = model.slug
    section_slicing = model.profile_name in TRAINING_ORDER
    local_root = (
        f"outputs/sft/{slug}/section-sliced"
        if section_slicing
        else f"outputs/sft/{slug}"
    )
    artifact_slug = f"{slug}-section-sliced" if section_slicing else slug
    batch_size, accumulation = {
        "qwen": (1, 8),
        # Largest integer micro-batches fitting the conservative
        # 8,192-token / 78-GiB budget. The next integer fails memory.py.
        "gemma": (17, 1),
        "deepseek": (24, 1),
    }[model.profile_name]
    return RunConfig(
        model=model,
        data=DataConfig(
            cache_dir=Path(f"{local_root}/cache"),
            prepared_dir=Path(f"{local_root}/prepared-dataset"),
            slice_by_section=section_slicing,
        ),
        training=TrainingConfig(
            output_dir=Path(f"{local_root}/checkpoints"),
            adapter_dir=Path(f"{local_root}/lora"),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
        ),
        tracking=TrackingConfig(
            artifact_name=f"{artifact_slug}-lora",
            checkpoint_artifact_name=f"{artifact_slug}-checkpoint",
            length_cache_artifact_name=f"{artifact_slug}-token-lengths",
            prepared_dataset_artifact_name=f"{artifact_slug}-prepared-sft",
        ),
    )
