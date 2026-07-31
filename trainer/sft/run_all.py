"""Run Qwen, Gemma, then DeepSeek SFT as isolated child processes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Sequence

from .config import MODEL_ORDER, MODEL_PROFILES


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
        "--num-train-epochs",
        str(args.num_train_epochs),
        "--eval-steps",
        str(args.eval_steps),
        "--save-steps",
        str(args.save_steps),
        "--gpu-count",
        str(args.gpu_count),
        "--per-device-batch-size",
        str(args.per_device_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--wandb-project",
        args.wandb_project,
        "--wandb-run-name",
        (
            f"{args.wandb_run_prefix}-{profile.slug}"
            if args.wandb_run_prefix
            else profile.slug
        ),
    ]
    if args.max_steps is not None:
        command.extend(("--max-steps", str(args.max_steps)))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the default deep model/environment smoke test",
    )
    parser.add_argument(
        "--quick-preflight",
        action="store_true",
        help="Check metadata and services without loading model weights",
    )
    parser.add_argument(
        "--dataset", default="Haeryz/putusan-structured-extraction"
    )
    parser.add_argument("--dataset-config", default="sft")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optional step limit for smoke/debug runs",
    )
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=5)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--allow-non-a100", action="store_true")
    parser.add_argument("--wandb-project", default="putusan-sft")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-prefix")
    parser.add_argument("--no-wandb-upload", action="store_true")
    parser.add_argument("--no-wandb-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        if not args.quick_preflight:
            preflight.append("--deep")
        subprocess.run(preflight, check=True, env=child_environment)

    for position, model_key in enumerate(MODEL_ORDER, start=1):
        profile = MODEL_PROFILES[model_key]
        print(
            f"\n=== [{position}/{len(MODEL_ORDER)}] "
            f"{profile.model_name} ===",
            flush=True,
        )
        subprocess.run(
            build_training_command(model_key, args),
            check=True,
            env=child_environment,
        )
    print("\nAll three SFT jobs completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
