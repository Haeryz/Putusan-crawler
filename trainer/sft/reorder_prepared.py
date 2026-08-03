"""Upgrade an existing DeepSeek prepared dataset to hard-first order.

This changes only row order and the compatibility manifest. Token IDs and
response-only labels are reused byte-for-byte; no tokenizer or model is loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .config import MODEL_PROFILES, run_config_for_model
from .precompute_lengths import load_source_splits
from .prepared import MANIFEST_NAME, save_prepared_splits
from .section_slicing import (
    HARD_FIRST_CURRICULUM,
    hard_first_indices_for_source_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reorder an existing DeepSeek section-sliced prepared dataset into "
            "the deterministic hard-first curriculum without retokenizing."
        )
    )
    parser.add_argument(
        "--dataset",
        default="Haeryz/putusan-structured-extraction",
    )
    parser.add_argument("--dataset-config", default="sft")
    return parser


def reorder_split(prepared: Any, source: Sequence[dict[str, Any]]) -> Any:
    order = hard_first_indices_for_source_rows(source)
    if len(order) != len(prepared):
        raise RuntimeError(
            f"Prepared/source row mismatch: {len(prepared)} != {len(order)}"
        )
    return prepared.select(order)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = run_config_for_model(MODEL_PROFILES["deepseek"])
    destination = config.data.prepared_dir
    manifest_path = destination / MANIFEST_NAME
    dataset_path = destination / "dataset"
    if not manifest_path.is_file() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Prepared DeepSeek dataset missing: {destination}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("base_model") != config.model.model_name:
        raise RuntimeError("Prepared dataset is not the DeepSeek profile")
    if manifest.get("section_slicing_version") != 1:
        raise RuntimeError("Only parent-major section slicing v1 can be upgraded")
    if manifest.get("train_curriculum") == HARD_FIRST_CURRICULUM:
        print("DeepSeek prepared dataset is already hard-first.", flush=True)
        return 0

    from datasets import load_from_disk

    prepared = load_from_disk(str(dataset_path))
    train_source, _ = load_source_splits(args.dataset, args.dataset_config)
    print(
        f"Reordering {len(prepared['train'])} DeepSeek rows without tokenization...",
        flush=True,
    )
    reordered_train = reorder_split(prepared["train"], train_source)
    manifest["train_curriculum"] = HARD_FIRST_CURRICULUM
    save_prepared_splits(
        reordered_train,
        prepared["validation"],
        destination,
        manifest,
    )
    print(f"Saved hard-first prepared dataset to {destination}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
