"""Command-line entry point for modular SFT training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
from typing import Sequence

from .config import RunConfig


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3.5-4B for per-section putusan extraction"
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--dataset", default="Haeryz/putusan-structured-extraction"
    )
    parser.add_argument("--dataset-config", default="sft_sections")
    parser.add_argument("--max-steps", type=positive_int, default=100)
    parser.add_argument("--max-seq-length", type=positive_int, default=49_152)
    parser.add_argument("--per-device-batch-size", type=positive_int, default=2)
    parser.add_argument(
        "--gradient-accumulation-steps", type=positive_int, default=2
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/sft/checkpoints")
    )
    parser.add_argument(
        "--adapter-dir", type=Path, default=Path("qwen_extractor_sft_lora")
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
        "--wandb-artifact-name", default="qwen-extractor-sft-lora"
    )
    parser.add_argument(
        "--no-wandb-upload",
        action="store_true",
        help="Track metrics but do not upload the final adapter artifact",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Build immutable workflow configuration from parsed arguments."""

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    config = RunConfig()
    config = replace(
        config,
        model=replace(
            config.model,
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            require_a100=not args.allow_non_a100,
        ),
        data=replace(
            config.data, repository=args.dataset, subset=args.dataset_config
        ),
        training=replace(
            config.training,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            output_dir=args.output_dir,
            adapter_dir=args.adapter_dir,
        ),
        tracking=replace(
            config.tracking,
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name,
            artifact_name=args.wandb_artifact_name,
            upload_adapter=not args.no_wandb_upload,
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
