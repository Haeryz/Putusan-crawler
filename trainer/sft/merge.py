"""Restore a completed LoRA adapter and merge it into 16-bit base weights."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

from .checkpoint import load_adapter, save_merged
from .cli import HelpFormatter, positive_float
from .config import (
    MODEL_PROFILES,
    TrackingConfig,
    model_config_for,
    run_config_for_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge one completed LoRA adapter into serving-ready 16-bit base "
            "weights. A local adapter is preferred; when it is absent, the "
            "profile's W&B artifact is downloaded automatically."
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m trainer.sft.merge --model qwen\n"
            "  python -m trainer.sft.merge --model gemma "
            "--wandb-entity MY_TEAM\n"
            "  python -m trainer.sft.merge --model deepseek "
            "--adapter-dir /workspace/lora --output-dir /workspace/merged"
        ),
    )
    parser.add_argument(
        "--model",
        default="qwen",
        choices=(
            *MODEL_PROFILES,
            *(profile.model_name for profile in MODEL_PROFILES.values()),
        ),
        help="Model profile whose base weights and adapter should be merged",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        help="Local adapter directory; defaults to the profile's LoRA path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination for the merged Hugging Face model directory",
    )
    parser.add_argument(
        "--wandb-project",
        default=TrackingConfig().project,
        help="W&B project containing the final adapter artifact",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity/team; WANDB_ENTITY is used when set",
    )
    parser.add_argument(
        "--wandb-artifact",
        help=(
            "Explicit artifact reference, for example "
            "entity/project/qwen3-5-4b-lora:latest"
        ),
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Fail instead of downloading from W&B when no local adapter exists",
    )
    parser.add_argument(
        "--maximum-memory-usage",
        type=positive_float,
        default=0.75,
        help="Maximum fraction of GPU memory Unsloth may use while merging",
    )
    return parser


def adapter_artifact_reference(
    artifact_name: str,
    project: str,
    entity: str | None,
) -> str:
    prefix = f"{entity}/{project}" if entity else project
    return f"{prefix}/{artifact_name}:latest"


def download_adapter_artifact(reference: str, destination: Path) -> Path:
    """Download W&B's adapter/ payload and return its actual directory."""

    import wandb

    destination.mkdir(parents=True, exist_ok=True)
    artifact = wandb.Api().artifact(reference, type="model")
    downloaded = Path(artifact.download(root=str(destination)))
    adapter = downloaded / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        raise RuntimeError(
            f"W&B artifact {reference} has no adapter/adapter_config.json"
        )
    return adapter


def resolve_adapter(args: argparse.Namespace, profile_config: Any) -> Path:
    local = args.adapter_dir or profile_config.training.adapter_dir
    if (local / "adapter_config.json").is_file():
        print(f"Using local LoRA adapter: {local}", flush=True)
        return local
    if args.local_only:
        raise FileNotFoundError(f"No LoRA adapter found at {local}")

    reference = args.wandb_artifact or adapter_artifact_reference(
        profile_config.tracking.artifact_name,
        args.wandb_project,
        args.wandb_entity,
    )
    download_root = Path("outputs/sft") / profile_config.model.slug / "wandb-lora"
    print(f"Downloading LoRA adapter from W&B: {reference}", flush=True)
    return download_adapter_artifact(reference, download_root)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_config = model_config_for(args.model)
    profile_config = run_config_for_model(model_config)
    adapter = resolve_adapter(args, profile_config)
    output = args.output_dir or (
        Path("outputs/sft") / model_config.slug / "merged-16bit"
    )

    print(f"Loading {model_config.model_name} with adapter {adapter}...", flush=True)
    model, tokenizer = load_adapter(adapter, model_config)
    print(f"Merging and saving 16-bit model to {output}...", flush=True)
    save_merged(
        model,
        tokenizer,
        output,
        maximum_memory_usage=args.maximum_memory_usage,
    )
    print(f"Merged model saved to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
