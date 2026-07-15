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


def fetch_length_cache(
    run: Any, cache_path: Path, config: TrackingConfig
) -> bool:
    """Download a previous run's token-length cache onto this machine."""

    reference = f"{config.length_cache_artifact_name}:latest"
    try:
        artifact = run.use_artifact(
            reference, type=config.length_cache_artifact_type
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        artifact.download(root=str(cache_path.parent))
    except Exception as error:
        # No artifact exists on the first run ever, and a cache miss must cost
        # measurement time rather than the whole training job.
        print(
            f"No reusable token-length cache ({type(error).__name__}); "
            "measuring instead.",
            flush=True,
        )
        return False
    if not cache_path.exists():
        print(
            f"Artifact {reference} did not contain {cache_path.name}; "
            "measuring instead.",
            flush=True,
        )
        return False
    print(f"Reusing token lengths from {reference}.", flush=True)
    return True


def upload_length_cache(
    run: Any,
    cache_path: Path,
    config: TrackingConfig,
    metadata: dict[str, Any],
) -> None:
    """Publish freshly measured token lengths for later runs to reuse."""

    import wandb

    artifact = wandb.Artifact(
        name=config.length_cache_artifact_name,
        type=config.length_cache_artifact_type,
        description="Per-row token counts used to select the SFT context window",
        metadata=metadata,
    )
    artifact.add_file(str(cache_path), name=cache_path.name)
    run.log_artifact(artifact, aliases=["latest"])
    print(
        f"Token-length cache queued for upload as "
        f"{config.length_cache_artifact_name}:latest.",
        flush=True,
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
