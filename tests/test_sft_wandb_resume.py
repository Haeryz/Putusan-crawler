from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from trainer.sft.checkpoint import (
    artifact_checkpoint_step,
    latest_checkpoint,
    restore_checkpoint_artifact,
    write_checkpoint_metadata,
)
from trainer.sft.config import MODEL_PROFILES, run_config_for_model
from trainer.sft.tracking import (
    find_newest_wandb_checkpoint,
    restore_newest_wandb_checkpoint,
)


class FakeArtifact:
    def __init__(
        self,
        step: int | None,
        *,
        base_model: str = "Qwen/Qwen3.5-4B",
        dataset: str = "Haeryz/putusan-structured-extraction",
        dataset_config: str = "sft",
        aliases: tuple[str, ...] = (),
    ) -> None:
        self.metadata = {
            "base_model": base_model,
            "dataset": dataset,
            "dataset_config": dataset_config,
        }
        if step is not None:
            self.metadata["global_step"] = step
        self.aliases = aliases
        self.version = f"v{step or 0}"
        self.download_calls = 0

    def download(self, *, root: str) -> str:
        self.download_calls += 1
        checkpoint = Path(root) / "checkpoint"
        checkpoint.mkdir(parents=True)
        step = artifact_checkpoint_step(self)
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step}),
            encoding="utf-8",
        )
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        return root


def config_with_output(tmp_path: Path):
    config = run_config_for_model(MODEL_PROFILES["qwen"])
    return replace(
        config,
        training=replace(config.training, output_dir=tmp_path),
    )


def install_fake_wandb(monkeypatch, artifacts) -> None:
    collection = SimpleNamespace(artifacts=lambda: iter(artifacts))

    class FakeApi:
        @staticmethod
        def artifact_collection_exists(name, artifact_type):
            assert name == "haeryz/putusan-sft/qwen3-5-4b-checkpoint"
            assert artifact_type == "model-checkpoint"
            return True

        @staticmethod
        def artifact_collection(artifact_type, name):
            assert artifact_type == "model-checkpoint"
            assert name == "haeryz/putusan-sft/qwen3-5-4b-checkpoint"
            return collection

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=FakeApi))


def test_artifact_step_falls_back_to_step_alias() -> None:
    artifact = FakeArtifact(None, aliases=("latest", "step-35"))

    assert artifact_checkpoint_step(artifact) == 35


def test_wandb_scan_selects_highest_compatible_step(monkeypatch, tmp_path) -> None:
    incompatible = FakeArtifact(50, base_model="another/model")
    step_5 = FakeArtifact(5)
    step_20 = FakeArtifact(20)
    install_fake_wandb(monkeypatch, [step_5, incompatible, step_20])

    selected = find_newest_wandb_checkpoint(
        SimpleNamespace(entity="haeryz", project="putusan-sft"),
        config_with_output(tmp_path),
    )

    assert selected == (step_20, 20)


def test_missing_wandb_collection_starts_without_remote_checkpoint(
    monkeypatch, tmp_path
) -> None:
    class FakeApi:
        @staticmethod
        def artifact_collection_exists(name, artifact_type):
            return False

    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(Api=FakeApi)
    )

    selected = find_newest_wandb_checkpoint(
        SimpleNamespace(entity="haeryz", project="putusan-sft"),
        config_with_output(tmp_path),
    )

    assert selected is None


def test_restore_checkpoint_artifact_creates_complete_atomic_directory(
    tmp_path,
) -> None:
    artifact = FakeArtifact(15)

    restored = restore_checkpoint_artifact(artifact, tmp_path, 15)

    assert restored == tmp_path / "checkpoint-15"
    assert latest_checkpoint(tmp_path) == restored
    assert (restored / "optimizer.pt").is_file()
    assert not list(tmp_path.glob(".wandb-checkpoint-*"))


def test_curriculum_checkpoint_requires_matching_local_identity(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-20"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 20}), encoding="utf-8"
    )
    required = {"train_curriculum": "hard_sections_first_v1"}

    assert latest_checkpoint(tmp_path, required) is None

    write_checkpoint_metadata(
        checkpoint, {"train_curriculum": "random_legacy_order"}
    )
    assert latest_checkpoint(tmp_path, required) is None

    write_checkpoint_metadata(checkpoint, required)
    assert latest_checkpoint(tmp_path, required) == checkpoint


def test_restore_quarantines_incomplete_same_step_directory(tmp_path) -> None:
    incomplete = tmp_path / "checkpoint-15"
    incomplete.mkdir()
    (incomplete / "partial.bin").write_bytes(b"partial")

    restored = restore_checkpoint_artifact(FakeArtifact(15), tmp_path, 15)

    assert (restored / "trainer_state.json").is_file()
    assert (
        tmp_path / ".incomplete-checkpoint-15" / "partial.bin"
    ).is_file()


def test_restore_rejects_artifact_with_mismatched_trainer_step(tmp_path) -> None:
    artifact = FakeArtifact(15)

    def mismatched_download(*, root: str) -> str:
        checkpoint = Path(root) / "checkpoint"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 14}), encoding="utf-8"
        )
        return root

    artifact.download = mismatched_download

    with pytest.raises(RuntimeError, match="trainer_state.json says step 14"):
        restore_checkpoint_artifact(artifact, tmp_path, 15)

    assert latest_checkpoint(tmp_path) is None


def test_newer_wandb_checkpoint_is_downloaded_automatically(
    monkeypatch, tmp_path
) -> None:
    artifact = FakeArtifact(25)
    install_fake_wandb(monkeypatch, [artifact])
    config = config_with_output(tmp_path)

    restored = restore_newest_wandb_checkpoint(
        SimpleNamespace(entity="haeryz", project="putusan-sft"),
        config,
    )

    assert restored == tmp_path / "checkpoint-25"
    assert artifact.download_calls == 1


def test_newer_local_checkpoint_wins_without_wandb_download(
    monkeypatch, tmp_path
) -> None:
    local = tmp_path / "checkpoint-30"
    local.mkdir()
    (local / "trainer_state.json").write_text(
        json.dumps({"global_step": 30}), encoding="utf-8"
    )
    artifact = FakeArtifact(25)
    install_fake_wandb(monkeypatch, [artifact])

    selected = restore_newest_wandb_checkpoint(
        SimpleNamespace(entity="haeryz", project="putusan-sft"),
        config_with_output(tmp_path),
    )

    assert selected == local
    assert artifact.download_calls == 0
