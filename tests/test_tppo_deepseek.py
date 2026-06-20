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
    assert "Return one JSON object with exactly these properties" in (
        anak_deepseek.SYSTEM_PROMPT
    )
