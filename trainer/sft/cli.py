"""Command-line entry point for modular SFT training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
from typing import Sequence

from .config import MODEL_PROFILES, RunConfig, model_config_for, run_config_for_model


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Preserve example line breaks while also displaying defaults."""


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


def resolve_gpu_count(requested: int | None) -> int:
    """Use an explicit count or every CUDA GPU visible to this process."""

    if requested is not None:
        return requested
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to auto-detect GPUs; run setup_runpod.sh first"
        ) from error
    # Keep a one-GPU profile on CPU-only developer machines; the hardware
    # preflight later reports the more useful "CUDA runtime required" error.
    return max(int(torch.cuda.device_count()), 1)


def build_parser() -> argparse.ArgumentParser:
    # Hyperparameters are defined in config.py; the flags below only override
    # those defaults, so they must read from it rather than restate values.
    defaults = RunConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune one supported model for putusan extraction. By default "
            "all CUDA GPUs visible inside the machine are used. Effective batch "
            "= per-device batch x GPU count x gradient accumulation. Oversized "
            "conversations are automatically middle-truncated to the context "
            "limit while preserving chat markers and the assistant response."
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "examples:\n"
            "  python trainer/sft/main.py --modelname qwen\n"
            "  python trainer/sft/main.py --modelname qwen --half-epoch\n"
            "  python -m trainer.sft --modelname qwen\n"
            "  python -m trainer.sft --model gemma --gpu-count 2 "
            "--dataset-config sft_sections\n"
            "  python -m trainer.sft --model deepseek --max-steps 10\n\n"
            "Run all three models sequentially with:\n"
            "  python -m trainer.sft.run_all"
        ),
    )
    parser.add_argument(
        "--modelname",
        "--model",
        "--model-name",
        dest="model",
        default="qwen",
        choices=(
            *MODEL_PROFILES,
            *(profile.model_name for profile in MODEL_PROFILES.values()),
        ),
        help="Model profile key or exact supported Hugging Face repository",
    )
    parser.add_argument(
        "--dataset",
        default="Haeryz/putusan-structured-extraction",
        help="Hugging Face dataset repository",
    )
    parser.add_argument(
        "--dataset-config",
        default=defaults.data.subset,
        help="Dataset subset/config, such as sft or sft_sections",
    )
    epoch_group = parser.add_mutually_exclusive_group()
    epoch_group.add_argument(
        "--num-train-epochs",
        type=positive_float,
        default=defaults.training.num_train_epochs,
        help="Number of complete passes over the training split",
    )
    epoch_group.add_argument(
        "--half-epoch",
        action="store_true",
        help="Train for half a pass over the training split (0.5 epoch)",
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Optional step limit for smoke/debug runs; overrides epoch count",
    )
    parser.add_argument(
        "--eval-steps",
        type=positive_int,
        default=defaults.training.eval_steps,
        help="Explicitly run validation every N optimizer steps",
    )
    parser.add_argument(
        "--evaluations-per-epoch",
        type=positive_int,
        default=None,
        help=(
            "Use automatic validation cadence instead of the fixed "
            "--eval-steps interval"
        ),
    )
    parser.add_argument(
        "--save-steps",
        type=positive_int,
        default=defaults.training.save_steps,
        help="Save and optionally upload every N optimizer steps",
    )
    parser.add_argument(
        "--gpu-count",
        type=positive_int,
        default=None,
        help="Override auto-detection and use exactly N visible GPUs",
    )
    parser.add_argument(
        "--max-seq-length",
        type=positive_int,
        default=None,
        help=(
            "Override the 49,152-token context limit; oversized conversations "
            "are automatically middle-truncated with chat markers preserved"
        ),
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=positive_int,
        default=defaults.training.per_device_train_batch_size,
        help="Micro-batch size processed by each GPU",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=positive_int,
        default=defaults.training.gradient_accumulation_steps,
        help="Micro-batches accumulated before each optimizer update",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override the profile-specific checkpoint directory",
    )
    parser.add_argument(
        "--adapter-dir", type=Path, default=None,
        help="Override the profile-specific final LoRA directory",
    )
    parser.add_argument(
        "--allow-non-a100",
        action="store_true",
        help="Skip the model-name/VRAM profile check (CUDA is still required)",
    )
    parser.add_argument(
        "--wandb-project",
        default=defaults.tracking.project,
        help="W&B project name",
    )
    parser.add_argument("--wandb-entity", help="Optional W&B team/entity")
    parser.add_argument("--wandb-run-name", help="Optional W&B run name")
    parser.add_argument(
        "--wandb-artifact-name", default=None,
        help="Override the profile-specific adapter artifact name",
    )
    parser.add_argument(
        "--wandb-checkpoint-artifact-name",
        default=None,
        help="Override the profile-specific checkpoint artifact name",
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
    # main() resolves the automatic value before training. Retain the profile
    # default here so configuration inspection and unit tests do not need
    # PyTorch installed.
    gpu_count = (
        args.gpu_count
        if args.gpu_count is not None
        else config.model.required_gpu_count
    )
    config = replace(
        config,
        model=replace(
            config.model,
            max_seq_length=args.max_seq_length or profile.max_seq_length,
            require_a100=not args.allow_non_a100,
            required_gpu_count=gpu_count,
            require_distributed_launch=gpu_count > 1,
        ),
        data=replace(
            config.data, repository=args.dataset, subset=args.dataset_config
        ),
        training=replace(
            config.training,
            num_train_epochs=(0.5 if args.half_epoch else args.num_train_epochs),
            max_steps=args.max_steps if args.max_steps is not None else -1,
            eval_steps=(
                None
                if args.evaluations_per_epoch is not None
                else args.eval_steps
            ),
            evaluations_per_epoch=(
                args.evaluations_per_epoch
                if args.evaluations_per_epoch is not None
                else config.training.evaluations_per_epoch
            ),
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
    if args.gpu_count is None:
        args.gpu_count = resolve_gpu_count(None)
        print(
            f"Auto-detected {args.gpu_count} visible CUDA GPU(s); using all of "
            "them. Pass --gpu-count N to override.",
            flush=True,
        )
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
