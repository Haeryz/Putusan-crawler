from __future__ import annotations

from pathlib import Path
from typing import Sequence

from llm_aggregator import anak_deepseek


def _configure() -> None:
    anak_deepseek.DEFAULT_INPUT = Path("downloads/TPPO/raw-text")
    anak_deepseek.DEFAULT_OUTPUT_DIR = Path("LLM-aggregator/TPPO/Deepseek/output")
    anak_deepseek.DEFAULT_STATE = Path("LLM-aggregator/TPPO/Deepseek/progress.jsonl")
    anak_deepseek.DEFAULT_ENV = Path("LLM-aggregator/TPPO/Deepseek/.env")
    anak_deepseek.DEFAULT_PAUSE_FILE = Path("LLM-aggregator/TPPO/Deepseek/pause")
    anak_deepseek.PROGRAM_NAME = "tppo-deepseek-aggregate"
    anak_deepseek.CORPUS_LABEL = "Putusan TPPO"


def build_parser():
    _configure()
    return anak_deepseek.build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    _configure()
    return anak_deepseek.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
