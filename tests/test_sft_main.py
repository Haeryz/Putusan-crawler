from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import trainer.sft.cli as sft_cli
from trainer.sft.checkpoint import latest_checkpoint
from trainer.sft.config import RunConfig, TrackingConfig
from trainer.sft import main as workflow
from trainer.sft import tracking
from trainer.sft.tracking import (
    checkpoint_upload_callback,
    log_checkpoint_artifact,
    log_model_artifact,
)


def test_distributed_command_relaunches_same_script_and_arguments() -> None:
    command = workflow.distributed_launch_command(
        Path("/workspace/Sinergi/trainer/sft/main.py"),
        ["--max-steps", "7"],
        gpu_count=2,
    )

    assert command[:4] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
    ]
    assert "--nproc_per_node=2" in command
    assert command[-2:] == ["--max-steps", "7"]


def test_cli_auto_detects_visible_gpus_and_relaunches_ddp(monkeypatch) -> None:
    launch_calls: list[tuple[list[str], int]] = []
    monkeypatch.setattr(sft_cli, "resolve_gpu_count", lambda requested: 2)
    monkeypatch.setattr(workflow, "distributed_world_size", lambda: 1)
    monkeypatch.setattr(
        workflow,
        "launch_distributed",
        lambda argv, count: launch_calls.append((list(argv), count)) or 0,
    )

    assert sft_cli.main(["--model", "qwen"]) == 0
    assert launch_calls == [(["--model", "qwen"], 2)]


def test_main_loads_training_env_without_overriding_shell(
    monkeypatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "HF_TOKEN=from-file\n"
        "WANDB_API_KEY=wandb-file\n"
        "HF_HOME=/workspace/.cache/huggingface\n"
        "UNRELATED=ignored\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_TOKEN", "from-shell")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    loaded = workflow.load_training_env(env_file)

    assert loaded == {"WANDB_API_KEY", "HF_HOME"}
    assert os.environ["HF_TOKEN"] == "from-shell"
    assert os.environ["WANDB_API_KEY"] == "wandb-file"
    assert os.environ["HF_HOME"] == "/workspace/.cache/huggingface"
    assert "UNRELATED" not in os.environ


def test_complete_workflow_runs_modules_in_order_and_uploads(
    monkeypatch, tmp_path: Path
) -> None:
    events: list[str] = []
    model = object()
    tokenizer = object()
    wandb_run = object()

    class FakeTrainer:
        def add_callback(self, callback):
            events.append("add_checkpoint_callback")

        def train(self, *, resume_from_checkpoint):
            assert resume_from_checkpoint is None
            events.append("train")
            return SimpleNamespace(metrics={"train_loss": 1.25})

        @staticmethod
        def is_world_process_zero() -> bool:
            return True

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        workflow, "validate_hardware", lambda config: events.append("validate")
    )
    monkeypatch.setattr(
        workflow,
        "initialize_wandb",
        lambda config: events.append("wandb_init") or wandb_run,
    )
    monkeypatch.setattr(
        workflow,
        "synchronize_wandb_checkpoint_restore",
        lambda config, run: events.append("restore_checkpoint"),
    )
    monkeypatch.setattr(
        workflow,
        "load_base_model",
        lambda config: events.append("load_model") or (model, tokenizer),
    )
    monkeypatch.setattr(
        workflow,
        "attach_lora",
        lambda loaded_model, config: events.append("attach") or loaded_model,
    )
    monkeypatch.setattr(
        workflow,
        "load_splits",
        lambda config: events.append("load_splits") or ([1], [2]),
    )
    monkeypatch.setattr(
        workflow,
        "synchronize_prepared_dataset",
        lambda config, run: events.append("restore_prepared") or None,
    )
    monkeypatch.setattr(
        workflow,
        "format_dataset",
        lambda dataset, loaded_tokenizer, model_config: events.append("format")
        or dataset,
    )
    monkeypatch.setattr(
        workflow,
        "load_or_measure_lengths",
        lambda dataset, loaded_tokenizer, config: events.append("lengths")
        or [100],
    )
    monkeypatch.setattr(
        workflow, "load_cached_lengths", lambda dataset, config: [100]
    )
    profile = SimpleNamespace(
        p50=100,
        p90=100,
        p95=100,
        maximum=100,
        max_length=256,
        coverage=1.0,
    )
    monkeypatch.setattr(
        workflow,
        "choose_max_length",
        lambda *args: events.append("choose_context") or profile,
    )
    monkeypatch.setattr(
        workflow,
        "truncate_dataset_preserving_responses",
        lambda dataset, *args: events.append("truncate") or dataset,
    )
    monkeypatch.setattr(workflow, "distributed_world_size", lambda: 2)
    monkeypatch.setattr(
        workflow,
        "build_trainer",
        lambda *args: events.append("build_trainer") or FakeTrainer(),
    )
    monkeypatch.setattr(
        workflow,
        "checkpoint_upload_callback",
        lambda *args: events.append("build_checkpoint_callback") or object(),
    )
    monkeypatch.setattr(
        workflow, "latest_checkpoint", lambda output_dir, metadata=None: None
    )
    monkeypatch.setattr(
        workflow,
        "save_adapter",
        lambda *args: events.append("save_adapter"),
    )
    monkeypatch.setattr(
        workflow,
        "log_model_artifact",
        lambda *args: events.append("upload_artifact")
        or SimpleNamespace(name="qwen-extractor:v0"),
    )
    monkeypatch.setattr(
        workflow,
        "finish_wandb",
        lambda run, exit_code: events.append(f"finish_{exit_code}"),
    )
    base_config = RunConfig()
    config = replace(
        base_config,
        training=replace(
            base_config.training,
            adapter_dir=tmp_path / "adapter",
        ),
    )

    returned_model, returned_tokenizer, _ = workflow.run_training(config)

    assert returned_model is model
    assert returned_tokenizer is tokenizer
    assert events == [
        "validate",
        "wandb_init",
        "restore_checkpoint",
        "load_model",
        "attach",
        "restore_prepared",
        "load_splits",
        "format",
        "format",
        "lengths",
        "choose_context",
        "truncate",
        "truncate",
        "build_trainer",
        "build_checkpoint_callback",
        "add_checkpoint_callback",
        "train",
        "save_adapter",
        "upload_artifact",
        "finish_0",
    ]


def test_wandb_artifact_upload_adds_directory_and_waits(
    monkeypatch, tmp_path: Path
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeArtifact:
        def __init__(self, **kwargs):
            captured["artifact_kwargs"] = kwargs

        def add_dir(self, *, local_path: str, name: str) -> None:
            captured["directory"] = (local_path, name)

    class LoggedArtifact:
        def wait(self, *, timeout: int):
            captured["timeout"] = timeout
            return self

    class FakeRun:
        def log_artifact(self, artifact, *, aliases):
            captured["aliases"] = aliases
            return LoggedArtifact()

    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(Artifact=FakeArtifact)
    )
    config = TrackingConfig(upload_timeout_seconds=123)

    logged = log_model_artifact(
        FakeRun(), adapter_dir, config, {"max_steps": 100}
    )

    assert isinstance(logged, LoggedArtifact)
    assert captured["directory"] == (str(adapter_dir), "adapter")
    assert captured["aliases"] == ["latest"]
    assert captured["timeout"] == 123


def test_wandb_upload_rejects_missing_adapter(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing adapter"):
        log_model_artifact(
            object(), tmp_path / "missing", TrackingConfig(), {}
        )


def test_wandb_checkpoint_upload_adds_resumable_directory_and_waits(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_dir = tmp_path / "checkpoint-5"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "trainer_state.json").write_text(
        "{}", encoding="utf-8"
    )
    captured: dict[str, object] = {}

    class FakeArtifact:
        def __init__(self, **kwargs):
            captured["artifact_kwargs"] = kwargs

        def add_dir(self, *, local_path: str, name: str) -> None:
            captured["directory"] = (local_path, name)

    class LoggedArtifact:
        name = "checkpoint:v0"

        def wait(self, *, timeout: int):
            captured["timeout"] = timeout
            return self

    class FakeRun:
        def log_artifact(self, artifact, *, aliases):
            captured["aliases"] = aliases
            return LoggedArtifact()

    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(Artifact=FakeArtifact)
    )
    config = TrackingConfig(upload_timeout_seconds=456)

    logged = log_checkpoint_artifact(
        FakeRun(),
        checkpoint_dir,
        global_step=5,
        config=config,
        metadata={"base_model": "Qwen/Qwen3.5-4B"},
    )

    assert isinstance(logged, LoggedArtifact)
    assert captured["directory"] == (str(checkpoint_dir), "checkpoint")
    assert captured["aliases"] == ["latest", "step-5"]
    assert captured["timeout"] == 456
    artifact_kwargs = captured["artifact_kwargs"]
    assert isinstance(artifact_kwargs, dict)
    assert artifact_kwargs["metadata"]["global_step"] == 5


def test_checkpoint_callback_uploads_the_just_saved_step(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_dir = tmp_path / "checkpoint-10"
    checkpoint_dir.mkdir()
    calls: list[tuple[Path, int]] = []

    class FakeTrainerCallback:
        pass

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(TrainerCallback=FakeTrainerCallback),
    )
    monkeypatch.setattr(
        tracking,
        "log_checkpoint_artifact",
        lambda run, path, step, config, metadata: (
            calls.append((path, step))
            or SimpleNamespace(name="checkpoint:v1")
        ),
    )
    control = object()
    callback = checkpoint_upload_callback(
        object(), TrackingConfig(), {"max_steps": 30}
    )

    returned = callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=10),
        control,
    )

    assert returned is control
    assert calls == [(checkpoint_dir, 10)]


def test_latest_checkpoint_ignores_incomplete_save_and_uses_highest_step(
    tmp_path: Path,
) -> None:
    for step in (5, 10):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(
            "{}", encoding="utf-8"
        )
    (tmp_path / "checkpoint-15").mkdir()
    (tmp_path / "checkpoint-invalid").mkdir()

    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-10"
