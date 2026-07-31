"""Command-line entry point for modular SFT training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
from typing import Sequence

from .config import MODEL_PROFILES, RunConfig, model_config_for, run_config_for_model


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    # Hyperparameters are defined in config.py; the flags below only override
    # those defaults, so they must read from it rather than restate values.
    defaults = RunConfig()
    parser = argparse.ArgumentParser(
        description="Fine-tune a supported text model for putusan extraction"
    )
    parser.add_argument(
        "--model",
        default="qwen",
        choices=(
            *MODEL_PROFILES,
            *(profile.model_name for profile in MODEL_PROFILES.values()),
        ),
    )
    parser.add_argument(
        "--dataset", default="Haeryz/putusan-structured-extraction"
    )
    parser.add_argument("--dataset-config", default=defaults.data.subset)
    parser.add_argument(
        "--num-train-epochs",
        type=positive_float,
        default=defaults.training.num_train_epochs,
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Optional step limit for smoke/debug runs; overrides epoch count",
    )
    parser.add_argument(
        "--eval-steps", type=positive_int, default=defaults.training.eval_steps
    )
    parser.add_argument(
        "--save-steps", type=positive_int, default=defaults.training.save_steps
    )
    parser.add_argument(
        "--gpu-count", type=positive_int, default=defaults.model.required_gpu_count
    )
    parser.add_argument(
        "--max-seq-length",
        type=positive_int,
        default=None,
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=positive_int,
        default=defaults.training.per_device_train_batch_size,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=positive_int,
        default=defaults.training.gradient_accumulation_steps,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None
    )
    parser.add_argument(
        "--adapter-dir", type=Path, default=None
    )
    parser.add_argument(
        "--allow-non-a100",
        action="store_true",
        help="Skip the model-name/VRAM profile check (CUDA is still required)",
    )
    parser.add_argument("--wandb-project", default="putusan-sft")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-artifact-name", default=None
    )
    parser.add_argument(
        "--wandb-checkpoint-artifact-name",
        default=None,
    )
    parser.add_argument(
        "--no-wandb-upload",
        action="store_true",
        help="Track metrics but do not upload checkpoint or LoRA artifacts",
    )
    parser.add_argument(
        "--no-wandb-resume",
        action="store_true",
        help="Do not scan W&B for a newer resumable checkpoint",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Build immutable workflow configuration from parsed arguments."""

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    profile = model_config_for(args.model)
    config = run_config_for_model(profile)
    config = replace(
        config,
        model=replace(
            config.model,
            max_seq_length=args.max_seq_length or profile.max_seq_length,
            require_a100=not args.allow_non_a100,
            required_gpu_count=args.gpu_count,
            require_distributed_launch=args.gpu_count > 1,
        ),
        data=replace(
            config.data, repository=args.dataset, subset=args.dataset_config
        ),
        training=replace(
            config.training,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps if args.max_steps is not None else -1,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            output_dir=args.output_dir or config.training.output_dir,
            adapter_dir=args.adapter_dir or config.training.adapter_dir,
        ),
        tracking=replace(
            config.tracking,
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name,
            artifact_name=(
                args.wandb_artifact_name or config.tracking.artifact_name
            ),
            upload_adapter=not args.no_wandb_upload,
            checkpoint_artifact_name=(
                args.wandb_checkpoint_artifact_name
                or config.tracking.checkpoint_artifact_name
            ),
            upload_checkpoints=not args.no_wandb_upload,
            restore_checkpoints=not args.no_wandb_resume,
        ),
    )
    return config


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    config = config_from_args(args)

    from .main import launch_distributed, run_training
    from .transformer import distributed_world_size

    if (
        config.model.require_distributed_launch
        and distributed_world_size() == 1
    ):
        return launch_distributed(raw_argv, config.model.required_gpu_count)

    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
