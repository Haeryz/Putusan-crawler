"""Weights & Biases run lifecycle and model-artifact uploads."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import RunConfig, TrackingConfig


def _serializable_config(config: RunConfig) -> dict[str, Any]:
    """Convert dataclasses, paths, and tuples to W&B-safe values."""

    return json.loads(json.dumps(asdict(config), default=str))


def initialize_wandb(config: RunConfig) -> Any:
    """Start the run reused by Trainer's W&B callback."""

    import wandb

    return wandb.init(
        project=config.tracking.project,
        entity=config.tracking.entity,
        name=config.tracking.run_name,
        job_type="sft",
        config=_serializable_config(config),
    )


def log_model_artifact(
    run: Any,
    adapter_dir: Path,
    config: TrackingConfig,
    metadata: dict[str, Any],
) -> Any:
    """Upload the complete adapter directory and wait for W&B to commit it."""

    if not adapter_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot upload missing adapter directory: {adapter_dir}"
        )
    import wandb

    artifact = wandb.Artifact(
        name=config.artifact_name,
        type=config.artifact_type,
        description="Qwen3.5 putusan structured-extraction LoRA adapters",
        metadata=metadata,
    )
    artifact.add_dir(local_path=str(adapter_dir), name="adapter")
    logged_artifact = run.log_artifact(
        artifact, aliases=list(config.artifact_aliases)
    )
    return logged_artifact.wait(timeout=config.upload_timeout_seconds)


def finish_wandb(run: Any | None, exit_code: int) -> None:
    """Flush metrics/artifacts and close a run if this process owns one."""

    if run is not None:
        run.finish(exit_code=exit_code)
