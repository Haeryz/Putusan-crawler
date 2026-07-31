"""Complete model-profile SFT job, from validation through W&B upload.

This file supports both package execution and the convenient RunPod command:

    cd trainer/sft
    python main.py

When started outside torchrun it relaunches itself with the configured number
of local DDP workers.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence
from dataclasses import replace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from trainer.sft.checkpoint import latest_checkpoint, save_adapter
    from trainer.sft.config import RunConfig
    from trainer.sft.data import (
        choose_max_length,
        format_dataset,
        length_cache_path,
        load_cached_lengths,
        load_or_measure_lengths,
        load_splits,
        truncate_dataset_preserving_responses,
    )
    from trainer.sft.tracking import (
        checkpoint_upload_callback,
        fetch_length_cache,
        finish_wandb,
        initialize_wandb,
        log_model_artifact,
        restore_newest_wandb_checkpoint,
        upload_length_cache,
    )
    from trainer.sft.training import build_trainer
    from trainer.sft.transformer import (
        attach_lora,
        distributed_world_size,
        load_base_model,
        validate_hardware,
    )
else:
    from .checkpoint import latest_checkpoint, save_adapter
    from .config import RunConfig
    from .data import (
        choose_max_length,
        format_dataset,
        length_cache_path,
        load_cached_lengths,
        load_or_measure_lengths,
        load_splits,
        truncate_dataset_preserving_responses,
    )
    from .tracking import (
        checkpoint_upload_callback,
        fetch_length_cache,
        finish_wandb,
        initialize_wandb,
        log_model_artifact,
        restore_newest_wandb_checkpoint,
        upload_length_cache,
    )
    from .training import build_trainer
    from .transformer import (
        attach_lora,
        distributed_world_size,
        load_base_model,
        validate_hardware,
    )


def _is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


TRAINING_ENV_KEYS = {
    "HF_TOKEN",
    "HF_HOME",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
}


def load_training_env(path: Path | None = None) -> set[str]:
    """Load the SFT .env without overriding explicitly exported variables."""

    env_path = path or Path(__file__).with_name(".env")
    loaded: set[str] = set()
    if not env_path.is_file():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in TRAINING_ENV_KEYS:
            continue
        value = value.strip().strip('"').strip("'")
        if value and key not in os.environ:
            os.environ[key] = value
            loaded.add(key)
    return loaded


def _stage(number: int, total: int, message: str) -> None:
    if _is_rank_zero():
        print(f"\n[{number}/{total}] {message}", flush=True)


def _restore_sync_marker(config: RunConfig) -> Path:
    """Return a per-torchrun marker shared by every local worker."""

    identity = os.environ.get("TORCHELASTIC_RUN_ID") or (
        f"{os.environ.get('MASTER_ADDR', 'local')}:"
        f"{os.environ.get('MASTER_PORT', 'single')}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return (
        config.training.output_dir
        / f".wandb-checkpoint-restore-{digest}.json"
    )


def synchronize_wandb_checkpoint_restore(
    config: RunConfig, wandb_run: Any | None
) -> None:
    """Let rank zero restore W&B state before any worker selects a checkpoint."""

    if not config.tracking.restore_checkpoints:
        return
    world_size = distributed_world_size()
    if world_size == 1:
        if wandb_run is None:
            raise RuntimeError("Rank 0 has no active W&B run for checkpoint scan")
        restore_newest_wandb_checkpoint(wandb_run, config)
        return

    marker = _restore_sync_marker(config)
    if _is_rank_zero():
        if wandb_run is None:
            raise RuntimeError("Rank 0 has no active W&B run for checkpoint scan")
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_suffix(".tmp")
        try:
            restored = restore_newest_wandb_checkpoint(wandb_run, config)
            payload = {
                "ok": True,
                "checkpoint": str(restored) if restored is not None else None,
            }
        except BaseException as error:
            payload = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(temporary, marker)
            raise
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, marker)
        return

    deadline = time.monotonic() + config.tracking.restore_timeout_seconds
    while time.monotonic() < deadline:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            time.sleep(1)
            continue
        if not payload.get("ok"):
            raise RuntimeError(
                "Rank 0 failed automatic W&B checkpoint restore: "
                f"{payload.get('error', 'unknown error')}"
            )
        return
    raise TimeoutError(
        f"Timed out waiting for rank 0 W&B restore marker {marker}"
    )


def distributed_launch_command(
    script: Path, argv: Sequence[str], gpu_count: int
) -> list[str]:
    """Build the torchrun-equivalent command used by direct Python execution."""

    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpu_count}",
        str(script),
        *argv,
    ]


def launch_distributed(argv: Sequence[str], gpu_count: int) -> int:
    """Relaunch this file with one process per local GPU."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is not installed. Run setup_runpod.sh before training."
        ) from error
    visible_gpus = torch.cuda.device_count()
    if visible_gpus < gpu_count:
        raise RuntimeError(
            f"This profile needs {gpu_count} visible GPUs, but Python sees "
            f"{visible_gpus}. Run this command in the VS Code window connected "
            "to the RunPod host, not in a local terminal."
        )
    command = distributed_launch_command(
        Path(__file__).resolve(), argv, gpu_count
    )
    print("Launching distributed training:", " ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


def run_training(config: RunConfig) -> tuple[Any, Any, Any]:
    """Run every modular stage and upload the final adapter to W&B."""

    total_stages = 12
    wandb_run: Any | None = None
    try:
        _stage(
            1,
            total_stages,
            f"Validate {config.model.required_gpu_count}x A100 80GB environment",
        )
        validate_hardware(config.model)

        _stage(2, total_stages, "Start Weights & Biases run")
        if _is_rank_zero():
            wandb_run = initialize_wandb(config)

        _stage(
            3,
            total_stages,
            "Scan W&B and restore the newest compatible checkpoint",
        )
        synchronize_wandb_checkpoint_restore(config, wandb_run)

        _stage(
            4,
            total_stages,
            f"Load {config.model.model_name} in 4-bit",
        )
        model, tokenizer = load_base_model(config.model)

        _stage(5, total_stages, "Verify long-context patches and attach LoRA")
        model = attach_lora(model, config.model)

        _stage(6, total_stages, "Load train and validation datasets")
        train_dataset, eval_dataset = load_splits(config.data)

        _stage(
            7,
            total_stages,
            f"Apply {config.model.profile_name} chat template",
        )
        train_dataset = format_dataset(
            train_dataset, tokenizer, config.model
        )
        eval_dataset = format_dataset(eval_dataset, tokenizer, config.model)

        _stage(8, total_stages, "Measure token lengths and select context")
        cache_path = length_cache_path(config.data)
        reuse_cache = (
            _is_rank_zero()
            and wandb_run is not None
            and config.tracking.reuse_length_cache
        )
        if reuse_cache and load_cached_lengths(train_dataset, config.data) is None:
            fetch_length_cache(wandb_run, cache_path, config.tracking)
        # Checked before measuring so a stale download still triggers an upload.
        measured = load_cached_lengths(train_dataset, config.data) is None
        lengths = load_or_measure_lengths(
            train_dataset, tokenizer, config.data
        )
        if reuse_cache and measured:
            upload_length_cache(
                wandb_run,
                cache_path,
                config.tracking,
                {
                    "repository": config.data.repository,
                    "subset": config.data.subset,
                    "split": config.data.train_split,
                    "rows": len(lengths),
                },
            )
        profile = choose_max_length(
            lengths,
            config.model.max_seq_length,
            config.data.length_percentile,
            config.data.length_multiple,
        )
        world_size = distributed_world_size()
        automatic_eval_cadence = config.training.eval_steps is None
        eval_steps = config.training.resolved_eval_steps(
            len(train_dataset), world_size
        )
        config = replace(
            config,
            training=replace(config.training, eval_steps=eval_steps),
        )
        train_dataset = truncate_dataset_preserving_responses(
            train_dataset, tokenizer, config.model, profile.max_length
        )
        eval_dataset = truncate_dataset_preserving_responses(
            eval_dataset, tokenizer, config.model, profile.max_length
        )
        if _is_rank_zero():
            print(
                f"Context: p50={profile.p50}, p90={profile.p90}, "
                f"p95={profile.p95}, max={profile.maximum}, "
                f"selected={profile.max_length}, coverage={profile.coverage:.2%}"
            )
            print(
                "Truncation: middle of user content, preserving instruction "
                "and response markers; "
                f"{1 - profile.coverage:.2%} of training rows exceed the "
                f"{profile.max_length}-token limit"
            )
            print(
                "Batch: "
                f"{config.training.per_device_train_batch_size}/GPU x "
                f"{world_size} GPUs x "
                f"{config.training.gradient_accumulation_steps} accumulation "
                f"= {config.training.effective_batch_size(world_size)}"
            )
            print(
                f"Evaluation: every {eval_steps} optimizer steps"
                + (
                    f" (target {config.training.evaluations_per_epoch}/epoch)"
                    if automatic_eval_cadence
                    else " (fixed interval)"
                )
            )

        _stage(9, total_stages, "Build response-only SFT trainer")
        trainer = build_trainer(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            profile.max_length,
            config.training,
            config.model,
        )

        artifact_metadata = {
            "base_model": config.model.model_name,
            "dataset": config.data.repository,
            "dataset_config": config.data.subset,
            "max_length": profile.max_length,
            "truncation_strategy": "middle-preserve-chat-markers-and-response",
            "fraction_rows_truncated": 1 - profile.coverage,
            "effective_batch_size": config.training.effective_batch_size(
                world_size
            ),
            "num_train_epochs": config.training.num_train_epochs,
            "max_steps": config.training.max_steps,
        }
        if (
            trainer.is_world_process_zero()
            and config.tracking.upload_checkpoints
        ):
            if wandb_run is None:
                raise RuntimeError("Rank 0 has no active W&B run")
            trainer.add_callback(
                checkpoint_upload_callback(
                    wandb_run,
                    config.tracking,
                    artifact_metadata,
                )
            )

        resume_checkpoint = latest_checkpoint(config.training.output_dir)
        if trainer.is_world_process_zero() and resume_checkpoint is not None:
            print(f"Resuming from {resume_checkpoint}.", flush=True)

        _stage(10, total_stages, "Train and evaluate")
        trainer_stats = trainer.train(
            resume_from_checkpoint=(
                str(resume_checkpoint) if resume_checkpoint is not None else None
            )
        )

        _stage(11, total_stages, "Save LoRA adapter locally")
        if trainer.is_world_process_zero():
            save_adapter(model, tokenizer, config.training.adapter_dir)

        _stage(12, total_stages, "Upload LoRA adapter as W&B model artifact")
        if trainer.is_world_process_zero() and config.tracking.upload_adapter:
            if wandb_run is None:
                raise RuntimeError("Rank 0 has no active W&B run")
            metrics = dict(getattr(trainer_stats, "metrics", {}))
            metadata = {
                **artifact_metadata,
                **metrics,
            }
            logged = log_model_artifact(
                wandb_run,
                config.training.adapter_dir,
                config.tracking,
                metadata,
            )
            print(f"W&B artifact uploaded: {logged.name}", flush=True)

        if trainer.is_world_process_zero():
            print(
                "Full single-model SFT complete.\n"
                f"Local LoRA adapter: {config.training.adapter_dir}\n"
                f"W&B project: {config.tracking.project}\n"
                "Merge step: not run (adapter remains separate).",
                flush=True,
            )

        finish_wandb(wandb_run, exit_code=0)
        return model, tokenizer, trainer
    except BaseException:
        finish_wandb(wandb_run, exit_code=1)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI options, auto-launch DDP, then run all training stages."""

    load_training_env()
    if __package__ in {None, ""}:
        from trainer.sft.cli import main as cli_main
    else:
        from .cli import main as cli_main
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
