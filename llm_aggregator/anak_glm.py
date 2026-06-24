from __future__ import annotations

from pathlib import Path
from typing import Sequence

from llm_aggregator import anak_deepseek


def _configure() -> None:
    anak_deepseek.MODEL = "zai-org/GLM-5.2"
    anak_deepseek.MODEL_LABEL = "GLM"
    anak_deepseek.DEFAULT_INPUT = Path("downloads/kasus anak/raw-text")
    anak_deepseek.DEFAULT_OUTPUT_DIR = Path("LLM-aggregator/Anak/GLM/output")
    anak_deepseek.DEFAULT_STATE = Path("LLM-aggregator/Anak/GLM/progress.jsonl")
    anak_deepseek.DEFAULT_ENV = Path("LLM-aggregator/Anak/Deepseek/.env")
    anak_deepseek.DEFAULT_PAUSE_FILE = Path("LLM-aggregator/Anak/GLM/pause")
    anak_deepseek.DEFAULT_SPAN_SPEC = Path("LLM-aggregator/Anak/GPT/SPAN_EXTRACTION_SPEC.md")
    anak_deepseek.DEFAULT_EXTRACTION_INSTRUCTIONS = Path(
        "LLM-aggregator/Anak/GPT/CODEX_EXTRACTION_INSTRUCTIONS.md"
    )
    anak_deepseek.DEFAULT_SCHEMA_GUIDE = Path("LLM-aggregator/Anak/GPT/Putusan-schema.md")
    anak_deepseek.PROGRAM_NAME = "anak-glm-aggregate"
    anak_deepseek.CORPUS_LABEL = "Putusan Anak"
    anak_deepseek.CORPUS_NAME = "Anak"
    anak_deepseek.FORMAT_GUIDE_NAME = "the SKKMA PDF"


def build_parser():
    _configure()
    return anak_deepseek.build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    _configure()
    return anak_deepseek.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
