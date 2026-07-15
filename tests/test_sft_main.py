from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from trainer.sft.config import RunConfig, TrackingConfig
from trainer.sft import main as workflow
from trainer.sft.tracking import log_model_artifact


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


def test_complete_workflow_runs_modules_in_order_and_uploads(
    monkeypatch, tmp_path: Path
) -> None:
    events: list[str] = []
    model = object()
    tokenizer = object()
    wandb_run = object()

    class FakeTrainer:
        def train(self):
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
        "load_base_model",
        lambda config: events.append("load_model") or (model, tokenizer),
    )
    monkeypatch.setattr(
        workflow,
        "verify_long_context_stack",
        lambda loaded_model: events.append("verify"),
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
        "format_dataset",
        lambda dataset, loaded_tokenizer: events.append("format") or dataset,
    )
    monkeypatch.setattr(
        workflow,
        "load_or_measure_lengths",
        lambda dataset, loaded_tokenizer, config: events.append("lengths")
        or [100],
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
    monkeypatch.setattr(workflow, "distributed_world_size", lambda: 2)
    monkeypatch.setattr(
        workflow,
        "build_trainer",
        lambda *args: events.append("build_trainer") or FakeTrainer(),
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
    config = RunConfig()
    config = SimpleNamespace(
        model=config.model,
        data=config.data,
        training=SimpleNamespace(
            **{
                **config.training.__dict__,
                "adapter_dir": tmp_path / "adapter",
                "effective_batch_size": config.training.effective_batch_size,
            }
        ),
        tracking=config.tracking,
    )

    returned_model, returned_tokenizer, _ = workflow.run_training(config)

    assert returned_model is model
    assert returned_tokenizer is tokenizer
    assert events == [
        "validate",
        "wandb_init",
        "load_model",
        "verify",
        "attach",
        "load_splits",
        "format",
        "format",
        "lengths",
        "choose_context",
        "build_trainer",
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

