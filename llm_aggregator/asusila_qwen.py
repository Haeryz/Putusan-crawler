from __future__ import annotations

from pathlib import Path
from typing import Sequence

from llm_aggregator import anak_deepseek


def _configure() -> None:
    anak_deepseek.MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    anak_deepseek.MODEL_LABEL = "Qwen"
    anak_deepseek.DEFAULT_INPUT = Path("downloads/Asusila/raw-text")
    anak_deepseek.DEFAULT_OUTPUT_DIR = Path("LLM-aggregator/Asusila/Qwen/output")
    anak_deepseek.DEFAULT_STATE = Path("LLM-aggregator/Asusila/Qwen/progress.jsonl")
    anak_deepseek.DEFAULT_ENV = Path("LLM-aggregator/Asusila/Deepseek/.env")
    anak_deepseek.DEFAULT_PAUSE_FILE = Path("LLM-aggregator/Asusila/Qwen/pause")
    anak_deepseek.DEFAULT_SPAN_SPEC = Path("LLM-aggregator/Asusila/GPT/SPAN_EXTRACTION_SPEC.md")
    anak_deepseek.DEFAULT_EXTRACTION_INSTRUCTIONS = Path(
        "LLM-aggregator/Asusila/GPT/CODEX_EXTRACTION_INSTRUCTIONS.md"
    )
    anak_deepseek.DEFAULT_SCHEMA_GUIDE = Path("LLM-aggregator/Asusila/GPT/Putusan-schema.md")
    anak_deepseek.PROGRAM_NAME = "asusila-qwen-aggregate"
    anak_deepseek.CORPUS_LABEL = "Putusan Asusila"
    anak_deepseek.CORPUS_NAME = "Asusila"
    anak_deepseek.FORMAT_GUIDE_NAME = "the Pidana Biasa Format KKMA PDF"


def build_parser():
    _configure()
    return anak_deepseek.build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    _configure()
    return anak_deepseek.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
