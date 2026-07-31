from __future__ import annotations

import os
from pathlib import Path

from trainer.sft.cli import build_parser as build_training_parser
from trainer.sft.config import TrackingConfig
from trainer.sft.merge import build_parser as build_merge_parser
from trainer.sft.precompute_lengths import (
    build_parser,
    load_secret_env,
    summarize_lengths,
)
from trainer.sft.run_all import build_parser as build_run_all_parser
from trainer.sft.tracking import fetch_length_cache


def test_precompute_defaults_to_all_models_and_sft_dataset() -> None:
    args = build_parser().parse_args([])

    assert args.model is None
    assert args.dataset_config == "sft"
    assert args.wandb_project == "Sinergi-training"


def test_every_sft_entrypoint_defaults_to_the_same_wandb_project() -> None:
    expected = "Sinergi-training"

    assert build_parser().parse_args([]).wandb_project == expected
    assert build_training_parser().parse_args([]).wandb_project == expected
    assert build_run_all_parser().parse_args([]).wandb_project == expected
    assert build_merge_parser().parse_args([]).wandb_project == expected


def test_cache_download_is_qualified_by_active_run_project(tmp_path: Path) -> None:
    references: list[tuple[str, str]] = []

    class FakeArtifact:
        @staticmethod
        def download(*, root: str) -> None:
            return None

    run = type(
        "Run",
        (),
        {
            "entity": "haeriz42069-universitas-muhammadiyah-malang",
            "project": "Sinergi-training",
            "use_artifact": lambda self, reference, type: (
                references.append((reference, type)) or FakeArtifact()
            ),
        },
    )()
    config = TrackingConfig(length_cache_artifact_name="qwen-token-lengths")

    assert fetch_length_cache(run, tmp_path / "token_lengths.npy", config) is False
    assert references == [
        (
            "haeriz42069-universitas-muhammadiyah-malang/"
            "Sinergi-training/qwen-token-lengths:latest",
            "dataset",
        )
    ]


def test_secret_env_loads_auth_but_not_runpod_cache_path(
    monkeypatch, tmp_path: Path
) -> None:
    for key in ("HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY", "HF_HOME"):
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "HF_TOKEN=hf_test\n"
        "WANDB_API_KEY=wandb_test\n"
        "WANDB_ENTITY=test-team\n"
        "HF_HOME=/workspace/.cache/huggingface\n",
        encoding="utf-8",
    )

    loaded = load_secret_env(env_file)

    assert loaded == {"HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY"}
    assert os.environ["HF_TOKEN"] == "hf_test"
    assert os.environ["WANDB_API_KEY"] == "wandb_test"
    assert os.environ["WANDB_ENTITY"] == "test-team"
    assert "HF_HOME" not in os.environ


def test_summary_preserves_cache_stats_when_percentile_exceeds_cap() -> None:
    profile, compatible = summarize_lengths(
        [100, 200, 300, 100_000],
        context_cap=256,
        percentile=95,
        multiple=256,
    )

    assert compatible is False
    assert profile.max_length == 256
    assert profile.coverage == 0.5
    assert profile.maximum == 100_000
