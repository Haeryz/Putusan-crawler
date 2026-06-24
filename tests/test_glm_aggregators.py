from __future__ import annotations

from pathlib import Path
from typing import Any

import llm_aggregator.anak_deepseek as anak_deepseek
import llm_aggregator.anak_glm as anak_glm
import llm_aggregator.tppo_glm as tppo_glm


def _preserve_shared_engine(monkeypatch: Any) -> None:
    for name in (
        "MODEL",
        "MODEL_LABEL",
        "DEFAULT_INPUT",
        "DEFAULT_OUTPUT_DIR",
        "DEFAULT_STATE",
        "DEFAULT_ENV",
        "DEFAULT_PAUSE_FILE",
        "DEFAULT_SPAN_SPEC",
        "DEFAULT_EXTRACTION_INSTRUCTIONS",
        "DEFAULT_SCHEMA_GUIDE",
        "PROGRAM_NAME",
        "CORPUS_LABEL",
        "CORPUS_NAME",
        "FORMAT_GUIDE_NAME",
    ):
        monkeypatch.setattr(anak_deepseek, name, getattr(anak_deepseek, name))


def test_anak_glm_replicates_anak_deepseek_with_glm_model_and_shared_env(
    monkeypatch: Any,
) -> None:
    _preserve_shared_engine(monkeypatch)

    args = anak_glm.build_parser().parse_args([])

    assert anak_deepseek.MODEL == "zai-org/GLM-5.2"
    assert anak_deepseek.MODEL_LABEL == "GLM"
    assert args.input_dir == Path("downloads/kasus anak/raw-text")
    assert args.output_dir == Path("LLM-aggregator/Anak/GLM/output")
    assert args.state == Path("LLM-aggregator/Anak/GLM/progress.jsonl")
    assert args.env_file == Path("LLM-aggregator/Anak/Deepseek/.env")
    assert args.pause_file == Path("LLM-aggregator/Anak/GLM/pause")
    assert args.reasoning_effort == "off"

    prompt = anak_deepseek.build_user_prompt("sample.txt", "PUTUSAN\nNomor 1")
    assert "You are Codex running the token-optimized Anak span-extraction task in:" in prompt
    assert "Assigned source: downloads/kasus anak/raw-text/sample.txt" in prompt
    assert "LLM-aggregator/Anak/GPT/CODEX_EXTRACTION_INSTRUCTIONS.md" in prompt


def test_tppo_glm_replicates_tppo_deepseek_with_glm_model_and_shared_env(
    monkeypatch: Any,
) -> None:
    _preserve_shared_engine(monkeypatch)

    args = tppo_glm.build_parser().parse_args([])

    assert anak_deepseek.MODEL == "zai-org/GLM-5.2"
    assert anak_deepseek.MODEL_LABEL == "GLM"
    assert args.input_dir == Path("downloads/TPPO/raw-text")
    assert args.output_dir == Path("LLM-aggregator/TPPO/GLM/output")
    assert args.state == Path("LLM-aggregator/TPPO/GLM/progress.jsonl")
    assert args.env_file == Path("LLM-aggregator/TPPO/Deepseek/.env")
    assert args.pause_file == Path("LLM-aggregator/TPPO/GLM/pause")
    assert args.reasoning_effort == "off"

    prompt = anak_deepseek.build_user_prompt("sample-tppo.txt", "PUTUSAN\nNomor 2")
    assert "You are Codex running the token-optimized TPPO span-extraction task in:" in prompt
    assert "Assigned source: downloads/TPPO/raw-text/sample-tppo.txt" in prompt
    assert "LLM-aggregator/TPPO/GPT/CODEX_EXTRACTION_INSTRUCTIONS.md" in prompt


def test_glm_api_call_uses_glm_model(monkeypatch: Any) -> None:
    _preserve_shared_engine(monkeypatch)
    anak_glm.build_parser()
    record = {key: [] for key in anak_deepseek.SECTION_KEYS}
    record["judul"] = ["PUTUSAN"]

    class Response:
        status_code = 200
        text = ""

        def iter_lines(self, decode_unicode: bool = False) -> list[str]:
            import json

            event = {
                "choices": [{"delta": {"content": json.dumps(record)}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            return [f"data: {json.dumps(event)}", "data: [DONE]"]

    class Session:
        last_json: dict[str, Any] | None = None

        def post(self, *args: Any, **kwargs: Any) -> Response:
            self.last_json = kwargs["json"]
            return Response()

    session = Session()
    anak_deepseek.call_deepseek(
        session,  # type: ignore[arg-type]
        api_key="test",
        source_name="one.txt",
        source_text="PUTUSAN",
        project=None,
        timeout_seconds=1,
        max_attempts=1,
        max_output_tokens=4096,
        base_delay_seconds=0,
    )

    assert session.last_json is not None
    assert session.last_json["model"] == "zai-org/GLM-5.2"
