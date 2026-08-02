"""Build fully tokenized per-model SFT datasets locally and publish to W&B."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
from typing import Any, Sequence

from .cli import HelpFormatter
from .config import (
    MODEL_PROFILES,
    TRAINING_ORDER,
    TrackingConfig,
    run_config_for_model,
)
from .data import (
    LengthProfile,
    choose_max_length,
    format_dataset,
    length_cache_path,
    load_or_measure_lengths,
    prepare_tokenized_dataset,
    slice_dataset_by_section,
)
from .prepared import prepared_manifest, save_prepared_splits
from .tracking import upload_length_cache, upload_prepared_dataset


SECRET_ENV_KEYS = {"HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY"}


def load_secret_env(path: Path) -> set[str]:
    """Load only authentication variables without overriding the shell."""

    loaded: set[str] = set()
    if not path.is_file():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in SECRET_ENV_KEYS:
            continue
        value = value.strip().strip('"').strip("'")
        if value and key not in os.environ:
            os.environ[key] = value
            loaded.add(key)
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Slice, format, truncate, tokenize, and response-mask complete "
            "Gemma/DeepSeek datasets locally, then upload trainer-ready W&B "
            "artifacts. No base-model weights or training GPU are used."
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "example:\n"
            "  python -m trainer.sft.precompute_dataset\n"
            "  python -m trainer.sft.precompute_dataset --model gemma "
            "--model deepseek"
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(MODEL_PROFILES),
        help="Profile to measure; repeat it (default: Gemma and DeepSeek)",
    )
    parser.add_argument(
        "--dataset",
        default="Haeryz/putusan-structured-extraction",
        help="Hugging Face dataset repository",
    )
    parser.add_argument(
        "--dataset-config",
        default="sft",
        help="Dataset subset/config",
    )
    parser.add_argument(
        "--wandb-project",
        default=TrackingConfig().project,
        help="W&B project receiving the cache artifacts",
    )
    parser.add_argument("--wandb-entity", help="Optional W&B entity/team")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("trainer/sft/.env"),
        help="File supplying HF_TOKEN and WANDB_API_KEY",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remeasure and rebuild prepared datasets",
    )
    parser.add_argument(
        "--upload-existing-only",
        action="store_true",
        help="Upload existing caches and prepared datasets without tokenization",
    )
    return parser


def load_source_splits(repository: str, subset: str) -> tuple[Any, Any]:
    from datasets import load_dataset

    return (
        load_dataset(repository, subset, split="train"),
        load_dataset(repository, subset, split="validation"),
    )


def load_tokenizer(model_key: str, token: str | None) -> Any:
    from transformers import AutoTokenizer

    model_config = MODEL_PROFILES[model_key]
    arguments = {"token": token} if token else {}
    return AutoTokenizer.from_pretrained(model_config.model_name, **arguments)


def remove_local_cache(cache_path: Path) -> None:
    """Remove exactly one known cache file for an explicit --force run."""

    if cache_path.is_file():
        cache_path.unlink()


def summarize_lengths(
    lengths: Sequence[int], context_cap: int, percentile: int, multiple: int
) -> tuple[LengthProfile, bool]:
    """Summarize the selected cap and whether every row fits inside it."""

    profile = choose_max_length(lengths, context_cap, percentile, multiple)
    return profile, profile.coverage == 1.0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loaded = load_secret_env(args.env_file)
    print(
        "Credentials available: "
        f"HF_TOKEN={'yes' if os.environ.get('HF_TOKEN') else 'no'}, "
        f"WANDB_API_KEY={'yes' if os.environ.get('WANDB_API_KEY') else 'no'}"
        + (f" (loaded {', '.join(sorted(loaded))} from .env)" if loaded else ""),
        flush=True,
    )
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required to upload length caches")

    model_keys = tuple(args.model or TRAINING_ORDER)
    source_splits = None
    if not args.upload_existing_only:
        print(
            f"Loading {args.dataset}/{args.dataset_config} source splits once...",
            flush=True,
        )
        source_splits = load_source_splits(args.dataset, args.dataset_config)

    import wandb

    for index, model_key in enumerate(model_keys, start=1):
        model_config = MODEL_PROFILES[model_key]
        config = run_config_for_model(model_config)
        config = replace(
            config,
            data=replace(
                config.data,
                repository=args.dataset,
                subset=args.dataset_config,
            ),
            tracking=replace(
                config.tracking,
                project=args.wandb_project,
                entity=args.wandb_entity,
            ),
        )
        cache_path = length_cache_path(config.data)
        if args.upload_existing_only:
            import numpy as np
            import json

            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"No existing cache for {model_key}: {cache_path}"
                )
            lengths = [int(value) for value in np.load(cache_path)]
            print(
                f"\n[{index}/{len(model_keys)}] Reusing {cache_path} "
                f"({len(lengths)} rows)...",
                flush=True,
            )
            manifest_path = config.data.prepared_dir / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"No prepared dataset for {model_key}: "
                    f"{config.data.prepared_dir}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            profile_data = manifest["length_profile"]
            profile = LengthProfile(**profile_data)
        else:
            print(
                f"\n[{index}/{len(model_keys)}] Loading tokenizer for "
                f"{model_config.model_name}...",
                flush=True,
            )
            tokenizer_or_processor = load_tokenizer(
                model_key, os.environ.get("HF_TOKEN")
            )
            if source_splits is None:
                raise RuntimeError("Source splits were not loaded")
            train_source, validation_source = source_splits
            train_dataset = train_source
            validation_dataset = validation_source
            if config.data.slice_by_section:
                train_dataset = slice_dataset_by_section(train_dataset)
                validation_dataset = slice_dataset_by_section(validation_dataset)
            train_dataset = format_dataset(
                train_dataset, tokenizer_or_processor, model_config
            )
            validation_dataset = format_dataset(
                validation_dataset, tokenizer_or_processor, model_config
            )
            if args.force:
                remove_local_cache(cache_path)
            lengths = load_or_measure_lengths(
                train_dataset, tokenizer_or_processor, config.data
            )
            profile, _ = summarize_lengths(
                lengths,
                model_config.max_seq_length,
                config.data.length_percentile,
                config.data.length_multiple,
            )
            prepared_train = prepare_tokenized_dataset(
                train_dataset,
                tokenizer_or_processor,
                model_config,
                profile.max_length,
            )
            prepared_validation = prepare_tokenized_dataset(
                validation_dataset,
                tokenizer_or_processor,
                model_config,
                profile.max_length,
            )
            manifest = prepared_manifest(
                config,
                profile.max_length,
                len(prepared_train),
                len(prepared_validation),
                {
                    "p50": profile.p50,
                    "p90": profile.p90,
                    "p95": profile.p95,
                    "maximum": profile.maximum,
                    "max_length": profile.max_length,
                    "coverage": profile.coverage,
                },
            )
            save_prepared_splits(
                prepared_train,
                prepared_validation,
                config.data.prepared_dir,
                manifest,
            )
        all_rows_fit = profile.coverage == 1.0
        print(
            f"{model_key}: rows={len(lengths)}, p50={profile.p50}, "
            f"p90={profile.p90}, p95={profile.p95}, max={profile.maximum}, "
            f"selected={profile.max_length}, coverage={profile.coverage:.2%}",
            flush=True,
        )
        if not all_rows_fit:
            print(
                f"WARNING: {model_key} p{config.data.length_percentile}="
                f"{profile.p95} exceeds the configured context cap "
                f"{model_config.max_seq_length}; training will middle-truncate "
                f"the user content in the longest "
                f"{1 - profile.coverage:.2%} of rows.",
                flush=True,
            )
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"precompute-token-lengths-{model_config.slug}",
            job_type="dataset-precompute",
            config={
                "base_model": model_config.model_name,
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "rows": len(lengths),
            },
            reinit="finish_previous",
        )
        try:
            upload_length_cache(
                run,
                cache_path,
                config.tracking,
                {
                    "repository": args.dataset,
                    "subset": args.dataset_config,
                    "split": config.data.train_split,
                    "rows": len(lengths),
                    "base_model": model_config.model_name,
                    "p50": profile.p50,
                    "p90": profile.p90,
                    "p95": profile.p95,
                    "maximum": profile.maximum,
                    "selected_max_length": profile.max_length,
                    "coverage": profile.coverage,
                    "context_profile_compatible": all_rows_fit,
                },
            )
            upload_prepared_dataset(
                run,
                config.data.prepared_dir,
                config.tracking,
                manifest,
            )
        finally:
            run.finish()

    print(
        "\nAll requested tokenized datasets and length caches uploaded to W&B.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
