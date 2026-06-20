from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from extractor.core import ExtractionResult, extract_pdf
from extractor.metrics import FidelityMetrics
from extractor.reporting import build_corpus_report


def _extract_one(source: Path, output_dir: Path, overwrite: bool) -> ExtractionResult:
    return extract_pdf(source, output_dir / f"{source.stem}.txt", overwrite=overwrite)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", type=Path, default=Path("downloads/TPPO/pdfs"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("downloads/TPPO/raw-text")
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pdf_dir = args.pdf_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "audit.jsonl"
    summary_path = output_dir / "metrics-summary.json"

    pdfs = sorted(pdf_dir.glob("*.pdf"), key=lambda path: path.name.casefold())
    pending = [
        pdf
        for pdf in pdfs
        if args.overwrite or not (output_dir / f"{pdf.stem}.txt").exists()
    ]
    if args.limit > 0:
        pending = pending[: args.limit]

    existing_results: list[ExtractionResult] = []
    if audit_path.exists() and not args.overwrite:
        with audit_path.open("r", encoding="utf-8") as audit:
            for line in audit:
                record = json.loads(line)
                if "error" not in record:
                    record["metrics"] = FidelityMetrics(**record["metrics"])
                    record["pages_below_threshold"] = tuple(
                        record["pages_below_threshold"]
                    )
                    record["warnings"] = tuple(record["warnings"])
                    existing_results.append(ExtractionResult(**record))

    if args.overwrite:
        audit_path.write_text("", encoding="utf-8", newline="\n")
        existing_results = []

    results = existing_results.copy()
    errors: list[dict[str, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_extract_one, source, output_dir, args.overwrite): source
            for source in pending
        }
        with audit_path.open("a", encoding="utf-8", newline="\n") as audit:
            for future in as_completed(futures):
                source = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    audit.write(result.to_json() + "\n")
                    audit.flush()
                    print(
                        f"[{result.status.upper():6}] {source.name} "
                        f"chars={result.raw_characters}"
                    )
                except Exception as exc:
                    error = {"source": str(source.resolve()), "error": str(exc)}
                    errors.append(error)
                    audit.write(json.dumps(error, ensure_ascii=False) + "\n")
                    audit.flush()
                    print(f"[ERROR ] {source.name}: {exc}")

    deduped_results = {
        result.source.casefold(): result
        for result in results
        if Path(result.output).exists()
    }
    clean_results = sorted(
        deduped_results.values(), key=lambda item: item.source.casefold()
    )
    with audit_path.open("w", encoding="utf-8", newline="\n") as audit:
        for result in clean_results:
            audit.write(result.to_json() + "\n")

    summary = build_corpus_report(clean_results)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )

    passed = sum(result.status == "passed" for result in clean_results)
    review = len(clean_results) - passed
    print(
        f"Processed this run: {len(pending)}. Audit records: {len(clean_results)}. "
        f"Text files: {len(list(output_dir.glob('*.txt')))}. "
        f"{passed} passed, {review} review, {len(errors)} errors."
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
