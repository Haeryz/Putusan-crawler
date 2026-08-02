"""Run the outstanding Gemma and DeepSeek SFT jobs as child processes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Sequence

from .config import MODEL_PROFILES, TRAINING_ORDER, TrackingConfig
from .cli import HelpFormatter, positive_float, positive_int, resolve_gpu_count


def build_training_command(
    model_key: str, args: argparse.Namespace
) -> list[str]:
    """Build one model command while leaving its output paths isolated."""

    profile = MODEL_PROFILES[model_key]
    command = [
        sys.executable,
        "-m",
        "trainer.sft",
        "--model",
        model_key,
        "--dataset",
        args.dataset,
        "--dataset-config",
        args.dataset_config,
        "--save-steps",
        str(args.save_steps),
        "--gpu-count",
        str(args.gpu_count),
        "--wandb-project",
        args.wandb_project,
        "--wandb-run-name",
        (
            f"{args.wandb_run_prefix}-{profile.slug}"
            if args.wandb_run_prefix
            else profile.slug
        ),
    ]
    if args.per_device_batch_size is not None:
        command.extend(
            ("--per-device-batch-size", str(args.per_device_batch_size))
        )
    if args.gradient_accumulation_steps is not None:
        command.extend((
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
        ))
    if args.half_epoch:
        command.append("--half-epoch")
    else:
        command.extend(("--num-train-epochs", str(args.num_train_epochs)))
    if args.max_steps is not None:
        command.extend(("--max-steps", str(args.max_steps)))
    if args.evaluations_per_epoch is not None:
        command.extend(
            ("--evaluations-per-epoch", str(args.evaluations_per_epoch))
        )
    elif args.eval_steps is not None:
        command.extend(("--eval-steps", str(args.eval_steps)))
    if args.wandb_entity:
        command.extend(("--wandb-entity", args.wandb_entity))
    if args.allow_non_a100:
        command.append("--allow-non-a100")
    if args.no_wandb_upload:
        command.append("--no-wandb-upload")
    if args.no_wandb_resume:
        command.append("--no-wandb-resume")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gemma then DeepSeek SFT sequentially. By default every "
            "CUDA GPU visible inside the machine is used for each model."
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "example:\n"
            "  python -m trainer.sft.run_all --evaluations-per-epoch 4\n\n"
            "For single-model options:\n"
            "  python -m trainer.sft --help"
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the default fast metadata/environment preflight",
    )
    parser.add_argument(
        "--quick-preflight",
        action="store_true",
        help="Deprecated alias; fast preflight is now the default",
    )
    parser.add_argument(
        "--deep-preflight",
        action="store_true",
        help="Load weights and run a forward smoke test before training",
    )
    parser.add_argument(
        "--dataset", default="Haeryz/putusan-structured-extraction",
        help="Hugging Face dataset repository passed to every model",
    )
    parser.add_argument(
        "--dataset-config", default="sft",
        help="Dataset subset/config, such as sft or sft_sections",
    )
    epoch_group = parser.add_mutually_exclusive_group()
    epoch_group.add_argument(
        "--num-train-epochs", type=positive_float, default=1.0,
        help="Number of complete passes for each model",
    )
    epoch_group.add_argument(
        "--half-epoch", action="store_true",
        help="Train every model for half a pass over its training split",
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        help="Optional step limit for smoke/debug runs",
    )
    parser.add_argument(
        "--eval-steps", type=positive_int, default=38,
        help="Explicitly run validation every N optimizer steps",
    )
    parser.add_argument(
        "--evaluations-per-epoch", type=positive_int, default=None,
        help="Use automatic cadence instead of the fixed --eval-steps interval",
    )
    parser.add_argument(
        "--save-steps", type=positive_int, default=50,
        help="Save a checkpoint every N optimizer steps",
    )
    parser.add_argument(
        "--gpu-count", type=positive_int, default=None,
        help="Override auto-detection and use exactly N visible GPUs",
    )
    parser.add_argument(
        "--per-device-batch-size", type=positive_int, default=None,
        help="Override profile micro-batch (Gemma 17, DeepSeek 24)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=positive_int, default=None,
        help="Override profile accumulation (default 1 for both)",
    )
    parser.add_argument(
        "--allow-non-a100", action="store_true",
        help="Skip the A100 name/VRAM guard (CUDA remains required)",
    )
    parser.add_argument(
        "--wandb-project",
        default=TrackingConfig().project,
        help="W&B project name",
    )
    parser.add_argument("--wandb-entity", help="Optional W&B team/entity")
    parser.add_argument(
        "--wandb-run-prefix", help="Prefix for profile-specific W&B run names"
    )
    parser.add_argument(
        "--no-wandb-upload", action="store_true",
        help="Track metrics without uploading checkpoints or final adapters",
    )
    parser.add_argument(
        "--no-wandb-resume", action="store_true",
        help="Do not scan W&B for a newer resumable checkpoint",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpu_count is None:
        args.gpu_count = resolve_gpu_count(None)
        print(
            f"Auto-detected {args.gpu_count} visible CUDA GPU(s); using all for "
            "each model. Pass --gpu-count N to override.",
            flush=True,
        )
    child_environment = os.environ.copy()
    for variable in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        child_environment.pop(variable, None)

    if not args.skip_preflight:
        preflight = [
            sys.executable,
            "-m",
            "trainer.sft.preflight",
            "--dataset",
            args.dataset,
            "--dataset-config",
            args.dataset_config,
        ]
        if args.deep_preflight:
            preflight.append("--deep")
        subprocess.run(preflight, check=True, env=child_environment)

    for position, model_key in enumerate(TRAINING_ORDER, start=1):
        profile = MODEL_PROFILES[model_key]
        print(
            f"\n=== [{position}/{len(TRAINING_ORDER)}] "
            f"{profile.model_name} ===",
            flush=True,
        )
        subprocess.run(
            build_training_command(model_key, args),
            check=True,
            env=child_environment,
        )
    print("\nGemma and DeepSeek SFT jobs completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
