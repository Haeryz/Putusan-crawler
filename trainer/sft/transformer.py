"""Unsloth model loading, LoRA attachment, and long-context checks."""

from __future__ import annotations

from collections import Counter
import os
from typing import Any

from .config import ModelConfig


def distributed_world_size(environ: dict[str, str] | None = None) -> int:
    """Read the process count supplied by torchrun."""

    environment = os.environ if environ is None else environ
    return int(environment.get("WORLD_SIZE", "1"))


def validate_distributed_launch(
    config: ModelConfig, environ: dict[str, str] | None = None
) -> int:
    """Require one DDP worker per GPU for a multi-GPU profile."""

    world_size = distributed_world_size(environ)
    if (
        config.require_distributed_launch
        and world_size != config.required_gpu_count
    ):
        raise RuntimeError(
            f"Expected {config.required_gpu_count} distributed workers, found "
            f"WORLD_SIZE={world_size}. Launch with: torchrun --standalone "
            f"--nproc_per_node={config.required_gpu_count} -m trainer.sft"
        )
    return world_size


def validate_hardware(config: ModelConfig) -> list[tuple[str, float]]:
    """Validate the configured A100-80GB hardware profile."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA runtime is required for this SFT profile")
    world_size = validate_distributed_launch(config)
    visible_count = torch.cuda.device_count()
    if visible_count < config.required_gpu_count:
        raise RuntimeError(
            f"Expected {config.required_gpu_count} visible GPUs, found "
            f"{visible_count}"
        )

    devices: list[tuple[str, float]] = []
    for index in range(config.required_gpu_count):
        name = torch.cuda.get_device_name(index)
        vram_gb = (
            torch.cuda.get_device_properties(index).total_memory / 1024**3
        )
        devices.append((name, vram_gb))
        if config.require_a100 and (
            "A100" not in name or vram_gb < config.minimum_vram_gb
        ):
            raise RuntimeError(
                f"GPU {index}: expected an A100 80GB-class device with at "
                f"least {config.minimum_vram_gb:g} GiB; found {name} "
                f"({vram_gb:.2f} GiB)"
            )
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not 0 <= local_rank < visible_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is invalid for {visible_count} "
                "visible GPUs"
            )
        # The model is loaded before Trainer creates Accelerator, so select the
        # rank's GPU now instead of letting every worker allocate on cuda:0.
        torch.cuda.set_device(local_rank)
    return devices


def _progress(message: str) -> None:
    """Report a long silent step on rank zero."""

    if int(os.environ.get("RANK", "0")) == 0:
        print(message, flush=True)


def load_base_model(config: ModelConfig) -> tuple[Any, Any]:
    """Load one supported model with the matching Unsloth model wrapper."""

    # Both steps below can run for minutes without printing anything, so name
    # them before they start rather than leaving a bare cursor.
    _progress("  Importing Unsloth (patches kernels on first import)...")
    import torch
    from unsloth import FastLanguageModel, FastModel

    _progress(f"  Loading {config.model_name} in 4-bit (cached by setup)...")
    if config.loader_kind == "fast_model":
        return FastModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=config.load_in_4bit,
            full_finetuning=False,
        )
    return FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
        text_only=True,
        use_gradient_checkpointing="unsloth",
        unsloth_tiled_mlp=config.require_tiled_mlp,
    )


def lora_target_names(model: Any) -> Counter[str]:
    """Count LoRA parameters by projection name."""

    return Counter(
        name.split(".")[-4]
        for name, _ in model.named_parameters()
        if "lora_A" in name
    )


def covers_linear_attention(names: Counter[str]) -> bool:
    """Check for Qwen 3.5 linear-attention adapter coverage."""

    return any(
        name.startswith("in_proj") or name == "out_proj" for name in names
    )


def discover_lora_targets(model: Any, config: ModelConfig) -> list[str]:
    """Find language-block linear projections, excluding heads and vision."""

    import torch.nn as nn

    return sorted(
        {
            name.split(".")[-1]
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
            and not any(
                excluded in name
                for excluded in (
                    "lm_head",
                    "embed",
                    *config.non_text_module_fragments,
                )
            )
        }
    )


def _expected_language_layers(model: Any) -> int:
    text_config = getattr(model.config, "text_config", model.config)
    return int(text_config.num_hidden_layers)


def verify_long_context_stack(model: Any, config: ModelConfig) -> None:
    """Ensure tiled MLP and gradient checkpointing survive PEFT wrapping."""

    if config.require_tiled_mlp:
        expected = _expected_language_layers(model)
        tiled = [
            name
            for name, module in model.named_modules()
            if name.lower().endswith(".mlp")
            and hasattr(module, "_original_forward")
        ]
        if len(tiled) != expected:
            raise RuntimeError(
                f"Tiled MLP patched {len(tiled)}/{expected} language blocks"
            )

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    checkpointing = getattr(
        base_model, "is_gradient_checkpointing", False
    ) or getattr(base_model, "gradient_checkpointing", False)
    if not checkpointing:
        raise RuntimeError("Gradient checkpointing is not enabled on the base model")


def verify_non_text_modules_frozen(model: Any, config: ModelConfig) -> None:
    """Reject trainable adapters or weights in vision/audio components."""

    violations = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and any(
            fragment in name.lower()
            for fragment in config.non_text_module_fragments
        )
    ]
    if violations:
        preview = ", ".join(violations[:5])
        raise RuntimeError(
            f"{config.profile_name} has trainable non-text parameters: {preview}"
        )


def attach_lora(model: Any, config: ModelConfig) -> Any:
    """Attach language-only LoRA and prove all non-text towers remain frozen."""

    from peft import LoraConfig, PeftModel, get_peft_model

    if (
        config.require_linear_attention_lora
        and isinstance(model, PeftModel)
        and not covers_linear_attention(lora_target_names(model))
    ):
        model = model.unload()

    if not isinstance(model, PeftModel):
        if config.lora_kind == "multimodal_language_only":
            from unsloth import FastModel

            model = FastModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=0,
                bias="none",
                random_state=3407,
            )
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=0,
                    bias="none",
                    target_modules=discover_lora_targets(model, config),
                    task_type="CAUSAL_LM",
                ),
            )
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    peft_config = model.peft_config["default"]
    if (
        peft_config.r != config.lora_rank
        or peft_config.lora_alpha != config.lora_alpha
    ):
        raise RuntimeError(
            "Attached adapter does not match the requested LoRA profile"
        )
    if (
        config.require_linear_attention_lora
        and not covers_linear_attention(lora_target_names(model))
    ):
        raise RuntimeError("Qwen 3.5 linear-attention layers lack LoRA adapters")

    verify_non_text_modules_frozen(model, config)
    verify_long_context_stack(model, config)
    backfill_architectures(model)
    return model


def backfill_architectures(model: Any) -> int:
    """Repair configs used by Unsloth's patched generate implementation."""

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    architecture = [type(base_model).__name__]
    patched = 0
    modules = [model, *(module for _, module in model.named_modules())]
    for module in modules:
        module_config = getattr(module, "config", None)
        if (
            module_config is not None
            and getattr(module_config, "architectures", None) is None
        ):
            try:
                module_config.architectures = architecture
                patched += 1
            except Exception:
                pass
    return patched


def prepare_model(config: ModelConfig) -> tuple[Any, Any]:
    """Validate hardware, load the selected model, and attach language LoRA."""

    validate_hardware(config)
    model, tokenizer = load_base_model(config)
    return attach_lora(model, config), tokenizer
