"""Adapter and merged-model persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ModelConfig


def save_adapter(model: Any, tokenizer: Any, destination: Path) -> None:
    """Save Stage-1 LoRA adapters and tokenizer locally."""

    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)


def load_adapter(
    destination: Path, config: ModelConfig
) -> tuple[Any, Any]:
    """Load locally saved adapters through Unsloth for inference."""

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(destination),
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        text_only=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def save_merged(model: Any, tokenizer: Any, destination: Path) -> None:
    """Save a serving-ready 16-bit merged model locally."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(destination), tokenizer)


def push_adapter(model: Any, tokenizer: Any, repository: str, token: str) -> None:
    """Publish adapters explicitly; callers must supply their token at runtime."""

    model.push_to_hub(repository, token=token)
    tokenizer.push_to_hub(repository, token=token)


def push_merged(model: Any, tokenizer: Any, repository: str, token: str) -> None:
    """Publish a serving-ready merged model explicitly."""

    model.push_to_hub_merged(repository, tokenizer, token=token)

