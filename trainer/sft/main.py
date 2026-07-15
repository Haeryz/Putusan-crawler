"""Complete Qwen SFT job, from hardware validation through W&B upload.

This file supports both package execution and the convenient RunPod command:

    cd trainer/sft
    python main.py

When started outside torchrun it relaunches itself with the configured number
of local DDP workers.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from trainer.sft.checkpoint import save_adapter
    from trainer.sft.config import RunConfig
    from trainer.sft.data import (
        choose_max_length,
        format_dataset,
        load_or_measure_lengths,
        load_splits,
    )
    from trainer.sft.tracking import (
        finish_wandb,
        initialize_wandb,
        log_model_artifact,
    )
    from trainer.sft.training import build_trainer
    from trainer.sft.transformer import (
        attach_lora,
        distributed_world_size,
        load_base_model,
        validate_hardware,
        verify_long_context_stack,
    )
else:
    from .checkpoint import save_adapter
    from .config import RunConfig
    from .data import (
        choose_max_length,
        format_dataset,
        load_or_measure_lengths,
        load_splits,
    )
    from .tracking import finish_wandb, initialize_wandb, log_model_artifact
    from .training import build_trainer
    from .transformer import (
        attach_lora,
        distributed_world_size,
        load_base_model,
        validate_hardware,
        verify_long_context_stack,
    )


def _is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _stage(number: int, total: int, message: str) -> None:
    if _is_rank_zero():
        print(f"\n[{number}/{total}] {message}", flush=True)


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

    total_stages = 11
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

        _stage(3, total_stages, "Load text-only 4-bit Qwen base model")
        model, tokenizer = load_base_model(config.model)

        _stage(4, total_stages, "Verify long-context patches and attach LoRA")
        verify_long_context_stack(model)
        model = attach_lora(model, config.model)

        _stage(5, total_stages, "Load train and validation datasets")
        train_dataset, eval_dataset = load_splits(config.data)

        _stage(6, total_stages, "Apply Qwen chat template")
        train_dataset = format_dataset(train_dataset, tokenizer)
        eval_dataset = format_dataset(eval_dataset, tokenizer)

        _stage(7, total_stages, "Measure token lengths and select context")
        lengths = load_or_measure_lengths(
            train_dataset, tokenizer, config.data
        )
        profile = choose_max_length(
            lengths,
            config.model.max_seq_length,
            config.data.length_percentile,
            config.data.length_multiple,
        )
        if profile.coverage < config.data.minimum_context_coverage:
            raise RuntimeError(
                f"Context coverage {profile.coverage:.2%} is below "
                f"{config.data.minimum_context_coverage:.2%}"
            )
        world_size = distributed_world_size()
        if _is_rank_zero():
            print(
                f"Context: p50={profile.p50}, p90={profile.p90}, "
                f"p95={profile.p95}, max={profile.maximum}, "
                f"selected={profile.max_length}, coverage={profile.coverage:.2%}"
            )
            print(
                "Batch: "
                f"{config.training.per_device_train_batch_size}/GPU x "
                f"{world_size} GPUs x "
                f"{config.training.gradient_accumulation_steps} accumulation "
                f"= {config.training.effective_batch_size(world_size)}"
            )

        _stage(8, total_stages, "Build response-only SFT trainer")
        trainer = build_trainer(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            profile.max_length,
            config.training,
        )

        _stage(9, total_stages, "Train and evaluate")
        trainer_stats = trainer.train()

        _stage(10, total_stages, "Save LoRA adapter locally")
        if trainer.is_world_process_zero():
            save_adapter(model, tokenizer, config.training.adapter_dir)

        _stage(11, total_stages, "Upload LoRA adapter as W&B model artifact")
        if trainer.is_world_process_zero() and config.tracking.upload_adapter:
            if wandb_run is None:
                raise RuntimeError("Rank 0 has no active W&B run")
            metrics = dict(getattr(trainer_stats, "metrics", {}))
            metadata = {
                "base_model": config.model.model_name,
                "dataset": config.data.repository,
                "dataset_config": config.data.subset,
                "max_length": profile.max_length,
                "effective_batch_size": config.training.effective_batch_size(
                    world_size
                ),
                "max_steps": config.training.max_steps,
                **metrics,
            }
            logged = log_model_artifact(
                wandb_run,
                config.training.adapter_dir,
                config.tracking,
                metadata,
            )
            print(f"W&B artifact uploaded: {logged.name}", flush=True)

        finish_wandb(wandb_run, exit_code=0)
        return model, tokenizer, trainer
    except BaseException:
        finish_wandb(wandb_run, exit_code=1)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI options, auto-launch DDP, then run all training stages."""

    if __package__ in {None, ""}:
        from trainer.sft.cli import main as cli_main
    else:
        from .cli import main as cli_main
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
