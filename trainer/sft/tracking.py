"""Weights & Biases run lifecycle and model-artifact uploads."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .checkpoint import (
    artifact_checkpoint_step,
    checkpoint_step,
    latest_checkpoint,
    restore_checkpoint_artifact,
)
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


def _checkpoint_matches_run(artifact: Any, config: RunConfig) -> bool:
    """Require identity metadata before resuming executable training state."""

    metadata = getattr(artifact, "metadata", {}) or {}
    expected = {
        "base_model": config.model.model_name,
        "dataset": config.data.repository,
        "dataset_config": config.data.subset,
    }
    return all(metadata.get(key) == value for key, value in expected.items())


def find_newest_wandb_checkpoint(
    run: Any, config: RunConfig
) -> tuple[Any, int] | None:
    """Scan the model's W&B collection for its highest compatible step."""

    import wandb

    entity = getattr(run, "entity", None) or config.tracking.entity
    project = getattr(run, "project", None) or config.tracking.project
    if not entity:
        raise RuntimeError(
            "W&B did not resolve an entity for automatic checkpoint restore"
        )
    collection_name = (
        f"{entity}/{project}/{config.tracking.checkpoint_artifact_name}"
    )
    api = wandb.Api()
    if not api.artifact_collection_exists(
        collection_name, config.tracking.checkpoint_artifact_type
    ):
        print(
            f"No W&B checkpoint collection found at {collection_name}.",
            flush=True,
        )
        return None
    collection = api.artifact_collection(
        config.tracking.checkpoint_artifact_type,
        collection_name,
    )
    candidates: list[tuple[int, Any]] = []
    for artifact in collection.artifacts():
        step = artifact_checkpoint_step(artifact)
        if step is not None and _checkpoint_matches_run(artifact, config):
            candidates.append((step, artifact))
    if not candidates:
        print(
            f"No compatible checkpoints found in {collection_name}.",
            flush=True,
        )
        return None
    step, artifact = max(candidates, key=lambda item: item[0])
    return artifact, step


def restore_newest_wandb_checkpoint(
    run: Any, config: RunConfig
) -> Path | None:
    """Restore W&B only when its newest compatible step beats local state."""

    local = latest_checkpoint(config.training.output_dir)
    local_step = checkpoint_step(local)
    remote = find_newest_wandb_checkpoint(run, config)
    if remote is None:
        if local is not None:
            print(f"Using local checkpoint {local}.", flush=True)
        return local
    artifact, remote_step = remote
    if local_step >= remote_step:
        print(
            f"Local checkpoint step {local_step} is at least as new as W&B "
            f"step {remote_step}; no download needed.",
            flush=True,
        )
        return local

    version = getattr(artifact, "version", "unknown")
    print(
        f"Restoring W&B checkpoint "
        f"{config.tracking.checkpoint_artifact_name}:{version} "
        f"(step {remote_step})...",
        flush=True,
    )
    restored = restore_checkpoint_artifact(
        artifact,
        config.training.output_dir,
        remote_step,
    )
    print(f"W&B checkpoint restored to {restored}.", flush=True)
    return restored


def fetch_length_cache(
    run: Any, cache_path: Path, config: TrackingConfig
) -> bool:
    """Download a previous run's token-length cache onto this machine."""

    project = getattr(run, "project", None) or config.project
    entity = getattr(run, "entity", None) or config.entity
    prefix = f"{entity}/{project}" if entity else project
    reference = f"{prefix}/{config.length_cache_artifact_name}:latest"
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
    logged = run.log_artifact(artifact, aliases=["latest"])
    committed = logged.wait(timeout=config.upload_timeout_seconds)
    print(
        f"Token-length cache uploaded as "
        f"{committed.name} with alias latest.",
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
        description="Putusan structured-extraction LoRA adapters",
        metadata=metadata,
    )
    artifact.add_dir(local_path=str(adapter_dir), name="adapter")
    logged_artifact = run.log_artifact(
        artifact, aliases=list(config.artifact_aliases)
    )
    return logged_artifact.wait(timeout=config.upload_timeout_seconds)


def log_checkpoint_artifact(
    run: Any,
    checkpoint_dir: Path,
    global_step: int,
    config: TrackingConfig,
    metadata: dict[str, Any],
) -> Any:
    """Upload one resumable Trainer checkpoint and wait for its commit."""

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot upload missing checkpoint directory: {checkpoint_dir}"
        )
    import wandb

    artifact = wandb.Artifact(
        name=config.checkpoint_artifact_name,
        type=config.checkpoint_artifact_type,
        description="Resumable SFT Trainer checkpoint",
        metadata={**metadata, "global_step": global_step},
    )
    artifact.add_dir(local_path=str(checkpoint_dir), name="checkpoint")
    logged_artifact = run.log_artifact(
        artifact,
        aliases=["latest", f"step-{global_step}"],
    )
    return logged_artifact.wait(timeout=config.upload_timeout_seconds)


def checkpoint_upload_callback(
    run: Any,
    config: TrackingConfig,
    metadata: dict[str, Any],
) -> Any:
    """Build a Trainer callback that commits every saved checkpoint to W&B."""

    from transformers import TrainerCallback

    class WandbCheckpointUploadCallback(TrainerCallback):
        def on_save(
            self,
            args: Any,
            state: Any,
            control: Any,
            **kwargs: Any,
        ) -> Any:
            checkpoint_dir = (
                Path(args.output_dir) / f"checkpoint-{state.global_step}"
            )
            logged = log_checkpoint_artifact(
                run,
                checkpoint_dir,
                state.global_step,
                config,
                metadata,
            )
            print(
                f"W&B checkpoint uploaded: {logged.name} "
                f"(step {state.global_step})",
                flush=True,
            )
            return control

    return WandbCheckpointUploadCallback()


def finish_wandb(run: Any | None, exit_code: int) -> None:
    """Flush metrics/artifacts and close a run if this process owns one."""

    if run is not None:
        run.finish(exit_code=exit_code)
