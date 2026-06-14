from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

from extractor.core import ExtractionResult, extract_pdf


def _extract_one(
    source: Path,
    output: Path,
    threshold: float,
    overwrite: bool,
) -> ExtractionResult:
    return extract_pdf(
        source,
        output,
        fidelity_threshold=threshold,
        overwrite=overwrite,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-extractor",
        description="Extract embedded PDF text and independently validate fidelity.",
    )
    parser.add_argument("input", type=Path, help="PDF file or directory of PDFs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extractor-output"),
        help="destination for UTF-8 .txt files and audit report",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="minimum character similarity and token recall (default: 0.95)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="parallel PDF workers (default: up to 4)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _find_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.casefold() == ".pdf" else []
    return sorted(input_path.glob("*.pdf"), key=lambda path: path.name.casefold())


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pdfs = _find_pdfs(args.input)
    if not pdfs:
        print(f"No PDF files found at {args.input}", file=sys.stderr)
        return 2
    if not 0.0 <= args.threshold <= 1.0:
        print("--threshold must be between 0 and 1", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("--workers must be at least 1", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "audit.jsonl"
    results: list[ExtractionResult] = []
    errors: list[dict[str, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _extract_one,
                source,
                output_dir / f"{source.stem}.txt",
                args.threshold,
                args.overwrite,
            ): source
            for source in pdfs
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{result.status.upper():6}] {source.name} "
                    f"chars={result.raw_characters} "
                    f"char_similarity={result.metrics.character_similarity:.3%} "
                    f"token_recall={result.metrics.token_recall:.3%}"
                )
            except Exception as exc:
                errors.append({"source": str(source.resolve()), "error": str(exc)})
                print(f"[ERROR ] {source.name}: {exc}", file=sys.stderr)

    with report_path.open("w", encoding="utf-8", newline="\n") as report:
        for result in sorted(results, key=lambda item: item.source.casefold()):
            report.write(result.to_json() + "\n")
        for error in sorted(errors, key=lambda item: item["source"].casefold()):
            report.write(json.dumps(error, ensure_ascii=False, sort_keys=True) + "\n")

    passed = sum(result.status == "passed" for result in results)
    review = len(results) - passed
    print(
        f"Processed {len(results)}/{len(pdfs)} PDFs: "
        f"{passed} passed, {review} review, {len(errors)} errors. "
        f"Report: {report_path}"
    )
    return 1 if errors or review else 0


if __name__ == "__main__":
    raise SystemExit(main())
