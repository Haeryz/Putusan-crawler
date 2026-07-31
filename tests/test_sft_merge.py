from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

from trainer.sft.checkpoint import load_adapter, save_merged
from trainer.sft.config import MODEL_PROFILES
from trainer.sft.merge import (
    adapter_artifact_reference,
    download_adapter_artifact,
)


def test_adapter_artifact_reference_supports_entity_or_default_entity() -> None:
    assert adapter_artifact_reference("qwen-lora", "putusan-sft", "haeryz") == (
        "haeryz/putusan-sft/qwen-lora:latest"
    )
    assert adapter_artifact_reference("qwen-lora", "putusan-sft", None) == (
        "putusan-sft/qwen-lora:latest"
    )


def test_wandb_adapter_download_requires_expected_adapter_subdirectory(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeArtifact:
        def download(self, *, root: str) -> str:
            adapter = Path(root) / "adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            return root

    class FakeApi:
        def artifact(self, reference: str, *, type: str):
            assert reference == "team/project/model:latest"
            assert type == "model"
            return FakeArtifact()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=FakeApi))

    adapter = download_adapter_artifact(
        "team/project/model:latest", tmp_path / "download"
    )

    assert adapter == tmp_path / "download" / "adapter"


def test_gemma_adapter_uses_multimodal_loader(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class FakeFastModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["kwargs"] = kwargs
            return "model", "tokenizer"

        @staticmethod
        def for_inference(model) -> None:
            calls["inference"] = model

    class WrongLoader:
        @staticmethod
        def from_pretrained(**kwargs):
            raise AssertionError("Gemma must use FastModel")

    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(
            FastModel=FakeFastModel,
            FastLanguageModel=WrongLoader,
        ),
    )

    assert load_adapter(tmp_path, MODEL_PROFILES["gemma"]) == (
        "model",
        "tokenizer",
    )
    assert calls["kwargs"] == {
        "model_name": str(tmp_path),
        "max_seq_length": 49_152,
        "load_in_4bit": True,
    }
    assert calls["inference"] == "model"


def test_merged_save_is_explicitly_16_bit(tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    model = SimpleNamespace(
        save_pretrained_merged=lambda *args, **kwargs: calls.update(
            {"args": args, "kwargs": kwargs}
        )
    )

    save_merged(model, "tokenizer", tmp_path / "merged", 0.5)

    assert calls["args"] == (str(tmp_path / "merged"), "tokenizer")
    assert calls["kwargs"] == {
        "save_method": "merged_16bit",
        "maximum_memory_usage": 0.5,
    }
