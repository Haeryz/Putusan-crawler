"""Persistence and compatibility checks for fully prepared SFT datasets."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from .config import RunConfig


MANIFEST_NAME = "manifest.json"
PREPARED_SCHEMA_VERSION = 1


def prepared_manifest(
    config: RunConfig,
    max_length: int,
    train_rows: int,
    validation_rows: int,
    length_profile: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "base_model": config.model.model_name,
        "repository": config.data.repository,
        "subset": config.data.subset,
        "section_slicing": config.data.slice_by_section,
        "section_slicing_version": config.data.section_slicing_version,
        "train_curriculum": config.data.train_curriculum,
        "max_length": max_length,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "length_profile": length_profile,
    }


def validate_prepared_manifest(
    manifest: dict[str, Any], config: RunConfig
) -> None:
    expected = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "base_model": config.model.model_name,
        "repository": config.data.repository,
        "subset": config.data.subset,
        "section_slicing": config.data.slice_by_section,
        "section_slicing_version": config.data.section_slicing_version,
        "train_curriculum": config.data.train_curriculum,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key, "none" if key == "train_curriculum" else None)
        != value
    }
    if mismatches:
        raise RuntimeError(f"Prepared dataset is incompatible: {mismatches}")
    max_length = int(manifest.get("max_length", 0))
    if not 1 <= max_length <= config.model.max_seq_length:
        raise RuntimeError("Prepared dataset max_length exceeds model profile")


def save_prepared_splits(
    train: Any,
    validation: Any,
    destination: Path,
    manifest: dict[str, Any],
) -> None:
    from datasets import DatasetDict

    temporary = destination.with_name(f"{destination.name}.building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    DatasetDict(train=train, validation=validation).save_to_disk(
        str(temporary / "dataset")
    )
    (temporary / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)


def load_prepared_splits(
    destination: Path, config: RunConfig
) -> tuple[Any, Any, dict[str, Any]] | None:
    manifest_path = destination / MANIFEST_NAME
    dataset_path = destination / "dataset"
    if not manifest_path.is_file() or not dataset_path.is_dir():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_prepared_manifest(manifest, config)
    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_path))
    train = dataset["train"]
    validation = dataset["validation"]
    if len(train) != int(manifest["train_rows"]):
        raise RuntimeError("Prepared train row count does not match manifest")
    if len(validation) != int(manifest["validation_rows"]):
        raise RuntimeError("Prepared validation row count does not match manifest")
    return train, validation, manifest
