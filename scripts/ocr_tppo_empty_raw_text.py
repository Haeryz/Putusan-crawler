from __future__ import annotations

import argparse
from pathlib import Path

from extractor.core import extract_pdf_with_windows_ocr
from llm_aggregator.anak_deepseek import compact_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCR-regenerate TPPO raw-text files that contain only boilerplate."
    )
    parser.add_argument("--pdf-dir", type=Path, default=Path("downloads/TPPO/pdfs"))
    parser.add_argument(
        "--raw-text-dir", type=Path, default=Path("downloads/TPPO/raw-text")
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("downloads/TPPO/raw-text/ocr-repair-audit.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pdf_dir = args.pdf_dir.resolve()
    raw_text_dir = args.raw_text_dir.resolve()
    candidates: list[Path] = []
    for raw_text in sorted(raw_text_dir.glob("*.txt"), key=lambda path: path.name.casefold()):
        text = raw_text.read_text(encoding="utf-8-sig")
        if not compact_source(text):
            candidates.append(raw_text)
    if args.limit > 0:
        candidates = candidates[: args.limit]

    if not candidates:
        print("No boilerplate-only TPPO raw-text files found.")
        return 0

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    repaired = 0
    missing = 0
    with args.audit.open("a", encoding="utf-8", newline="\n") as audit:
        for raw_text in candidates:
            pdf = pdf_dir / f"{raw_text.stem}.pdf"
            if not pdf.exists():
                missing += 1
                print(f"[MISSING] {pdf.name}")
                continue
            result = extract_pdf_with_windows_ocr(
                pdf,
                raw_text,
                overwrite=True,
                dpi=args.dpi,
            )
            audit.write(result.to_json() + "\n")
            audit.flush()
            compacted = compact_source(raw_text.read_text(encoding="utf-8-sig"))
            status = "REPAIRED" if compacted else "EMPTY"
            if compacted:
                repaired += 1
            print(f"[{status:8}] {raw_text.name} chars={result.raw_characters}")

    print(
        f"OCR repair complete: {repaired}/{len(candidates)} repaired, "
        f"{missing} missing PDFs. Audit: {args.audit}"
    )
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
