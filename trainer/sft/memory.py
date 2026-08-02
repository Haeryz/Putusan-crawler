"""Conservative, auditable VRAM budgeting for section-sliced QLoRA SFT."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelConfig, TrainingConfig


GIB = 2**30


@dataclass(frozen=True)
class TrainingMemoryEstimate:
    model_storage_gib: float
    adapter_optimizer_gib: float
    checkpointed_activations_gib: float
    fused_loss_workspace_gib: float
    runtime_reserve_gib: float
    fragmentation_headroom_gib: float
    total_gib: float
    usable_vram_gib: float


def estimate_training_memory(
    model: ModelConfig,
    training: TrainingConfig,
    max_length: int,
    available_vram_gib: float,
) -> TrainingMemoryEstimate:
    """Upper-budget one GPU with 4-bit weights and checkpointed BF16 states.

    The activation term stores every layer boundary in BF16 and applies a 2.5x
    forward/backward/workspace factor. Unsloth's fused loss is enabled by
    ``UNSLOTH_RETURN_LOGITS=0``; four GiB is still reserved for its tiled logits
    and kernel workspaces. A final 25% headroom covers allocator fragmentation
    and implementation variance.
    """

    if max_length < 1 or available_vram_gib <= 0:
        raise ValueError("max_length and available_vram_gib must be positive")
    batch = training.per_device_train_batch_size
    # Embeddings and frozen modality towers are commonly excluded from 4-bit
    # linear quantization. Budget those explicitly at BF16 and the remaining
    # parameters at 0.625 byte (NF4 payload + block metadata).
    non_quantized = model.non_quantized_parameter_count_billions * 1e9
    quantized = (
        model.parameter_count_billions
        - model.non_quantized_parameter_count_billions
    ) * 1e9
    quantized_model = (non_quantized * 2 + quantized * 0.625) / GIB
    # Rank-32 LoRA BF16 weights + BF16 gradients + 8-bit Adam states and scales.
    # Two GiB is deliberately larger than these adapters for either target.
    adapter_optimizer = 2.0
    layer_boundaries = (
        batch
        * max_length
        * model.hidden_size
        * model.hidden_layers
        * 2
        / GIB
    )
    checkpointed_activations = layer_boundaries * 2.5
    fused_loss_workspace = 4.0
    runtime_reserve = 8.0
    subtotal = (
        quantized_model
        + adapter_optimizer
        + checkpointed_activations
        + fused_loss_workspace
        + runtime_reserve
    )
    fragmentation = subtotal * 0.25
    total = subtotal + fragmentation
    # Keep 10% of physical VRAM completely unused for CUDA context and spikes.
    usable = available_vram_gib * 0.90
    return TrainingMemoryEstimate(
        model_storage_gib=quantized_model,
        adapter_optimizer_gib=adapter_optimizer,
        checkpointed_activations_gib=checkpointed_activations,
        fused_loss_workspace_gib=fused_loss_workspace,
        runtime_reserve_gib=runtime_reserve,
        fragmentation_headroom_gib=fragmentation,
        total_gib=total,
        usable_vram_gib=usable,
    )


def assert_training_memory_fits(
    model: ModelConfig,
    training: TrainingConfig,
    max_length: int,
    available_vram_gib: float,
) -> TrainingMemoryEstimate:
    """Fail before trainer construction if the conservative budget does not fit."""

    estimate = estimate_training_memory(
        model, training, max_length, available_vram_gib
    )
    if estimate.total_gib > estimate.usable_vram_gib:
        raise RuntimeError(
            f"Estimated peak {estimate.total_gib:.1f} GiB exceeds the safe "
            f"{estimate.usable_vram_gib:.1f} GiB budget; lower "
            "--per-device-batch-size or --max-seq-length"
        )
    return estimate
