from __future__ import annotations

from pathlib import Path

import llm_aggregator.anak_deepseek as anak_deepseek
import llm_aggregator.tppo_deepseek as tppo_deepseek


def test_tppo_wrapper_uses_tppo_paths_and_shared_prompt(monkeypatch) -> None:
    for name in (
        "DEFAULT_INPUT",
        "DEFAULT_OUTPUT_DIR",
        "DEFAULT_STATE",
        "DEFAULT_ENV",
        "DEFAULT_PAUSE_FILE",
        "PROGRAM_NAME",
        "CORPUS_LABEL",
    ):
        monkeypatch.setattr(anak_deepseek, name, getattr(anak_deepseek, name))

    args = tppo_deepseek.build_parser().parse_args([])

    assert args.input_dir == Path("downloads/TPPO/raw-text")
    assert args.output_dir == Path("LLM-aggregator/TPPO/Deepseek/output")
    assert args.state == Path("LLM-aggregator/TPPO/Deepseek/progress.jsonl")
    assert args.env_file == Path("LLM-aggregator/TPPO/Deepseek/.env")
    assert args.pause_file == Path("LLM-aggregator/TPPO/Deepseek/pause")
    assert not args.skip_empty_text
    assert anak_deepseek.MODEL == "deepseek-ai/DeepSeek-V4-Pro"

    prompt = anak_deepseek.build_user_prompt("sample-tppo.txt", "PUTUSAN\nNomor 2")
    span_spec = Path("LLM-aggregator/TPPO/GPT/SPAN_EXTRACTION_SPEC.md").read_text(
        encoding="utf-8"
    )
    assert "You are Codex running the token-optimized TPPO span-extraction task in:" in prompt
    assert "Assigned source: downloads/TPPO/raw-text/sample-tppo.txt" in prompt
    assert "source file, the TPPO Format PDF" in prompt
    assert "LLM-aggregator/TPPO/GPT/CODEX_EXTRACTION_INSTRUCTIONS.md" in prompt
    assert "YOUR ONLY OUTPUT: return the spans JSON object and nothing else." in prompt
    assert span_spec in prompt
