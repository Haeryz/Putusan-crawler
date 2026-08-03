"""Adapter and merged-model persistence helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from .config import ModelConfig


CHECKPOINT_METADATA_NAME = "sinergi_checkpoint.json"


def _checkpoint_metadata_matches(
    path: Path, required_metadata: Mapping[str, Any] | None
) -> bool:
    if not required_metadata:
        return True
    metadata_path = path / CHECKPOINT_METADATA_NAME
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(metadata.get(key) == value for key, value in required_metadata.items())


def write_checkpoint_metadata(
    checkpoint_dir: Path, metadata: Mapping[str, Any]
) -> None:
    """Bind a local Trainer checkpoint to its dataset/sampling identity."""

    (checkpoint_dir / CHECKPOINT_METADATA_NAME).write_text(
        json.dumps(dict(metadata), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def latest_checkpoint(
    output_dir: Path,
    required_metadata: Mapping[str, Any] | None = None,
) -> Path | None:
    """Return the highest-numbered complete Trainer checkpoint, if any."""

    if not output_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        # Trainer writes this after the model, optimizer, scheduler, and RNG
        # state. Ignoring directories without it avoids resuming a save that
        # was interrupted halfway through.
        if (
            (path / "trainer_state.json").is_file()
            and _checkpoint_metadata_matches(path, required_metadata)
        ):
            candidates.append((step, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def checkpoint_step(path: Path | None) -> int:
    """Return a checkpoint directory's numeric step, or zero when absent."""

    if path is None:
        return 0
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError as error:
        raise ValueError(f"Invalid checkpoint directory name: {path}") from error


def artifact_checkpoint_step(artifact: Any) -> int | None:
    """Read a Trainer step from W&B metadata or a step-N alias."""

    metadata = getattr(artifact, "metadata", {}) or {}
    try:
        step = int(metadata["global_step"])
    except (KeyError, TypeError, ValueError):
        step = 0
    if step > 0:
        return step
    for alias in getattr(artifact, "aliases", ()) or ():
        match = re.fullmatch(r"step-(\d+)", str(alias))
        if match and int(match.group(1)) > 0:
            return int(match.group(1))
    return None


def restore_checkpoint_artifact(
    artifact: Any,
    output_dir: Path,
    expected_step: int,
) -> Path:
    """Atomically restore one W&B artifact as a Trainer checkpoint."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"checkpoint-{expected_step}"
    if destination.exists():
        if (destination / "trainer_state.json").is_file():
            return destination
        quarantine = output_dir / f".incomplete-checkpoint-{expected_step}"
        suffix = 1
        while quarantine.exists():
            quarantine = output_dir / (
                f".incomplete-checkpoint-{expected_step}-{suffix}"
            )
            suffix += 1
        os.replace(destination, quarantine)
        print(
            f"Moved incomplete local checkpoint to {quarantine}.",
            flush=True,
        )

    with tempfile.TemporaryDirectory(
        prefix=".wandb-checkpoint-", dir=output_dir
    ) as temporary:
        downloaded = Path(artifact.download(root=temporary))
        source = downloaded / "checkpoint"
        state_path = source / "trainer_state.json"
        if not state_path.is_file():
            raise RuntimeError(
                "W&B checkpoint artifact is missing "
                "checkpoint/trainer_state.json"
            )
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            saved_step = int(state["global_step"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Invalid Trainer state in W&B checkpoint: {state_path}"
            ) from error
        if saved_step != expected_step:
            raise RuntimeError(
                f"W&B artifact metadata says step {expected_step}, but "
                f"trainer_state.json says step {saved_step}"
            )
        # The temporary directory is under output_dir, so this rename is an
        # atomic operation on one filesystem. A failed download never creates
        # a checkpoint-N directory that latest_checkpoint could select.
        os.replace(source, destination)
    return destination


def save_adapter(model: Any, tokenizer: Any, destination: Path) -> None:
    """Save Stage-1 LoRA adapters and tokenizer locally."""

    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)


def load_adapter(
    destination: Path, config: ModelConfig
) -> tuple[Any, Any]:
    """Load locally saved adapters through Unsloth for inference."""

    from unsloth import FastLanguageModel, FastModel

    loader = (
        FastModel
        if config.loader_kind == "fast_model"
        else FastLanguageModel
    )
    arguments: dict[str, Any] = {
        "model_name": str(destination),
        "max_seq_length": config.max_seq_length,
        "load_in_4bit": config.load_in_4bit,
    }
    if config.loader_kind == "fast_language_model":
        arguments["text_only"] = True
    model, tokenizer = loader.from_pretrained(**arguments)
    loader.for_inference(model)
    return model, tokenizer


def save_merged(
    model: Any,
    tokenizer: Any,
    destination: Path,
    maximum_memory_usage: float = 0.75,
) -> None:
    """Save a serving-ready 16-bit merged model locally."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(destination),
        tokenizer,
        save_method="merged_16bit",
        maximum_memory_usage=maximum_memory_usage,
    )


def push_adapter(model: Any, tokenizer: Any, repository: str, token: str) -> None:
    """Publish adapters explicitly; callers must supply their token at runtime."""

    model.push_to_hub(repository, token=token)
    tokenizer.push_to_hub(repository, token=token)


def push_merged(model: Any, tokenizer: Any, repository: str, token: str) -> None:
    """Publish a serving-ready merged model explicitly."""

    model.push_to_hub_merged(repository, tokenizer, token=token)
