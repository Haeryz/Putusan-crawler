from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import pymupdf
from pypdf import PdfReader

from extractor.metrics import FidelityMetrics, compare_text

PAGE_SEPARATOR = "\n\f\n"


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    source: str
    output: str
    source_sha256: str
    pages: int
    pages_with_text: int
    raw_characters: int
    elapsed_seconds: float
    status: str
    fidelity_threshold: float
    metrics: FidelityMetrics
    warnings: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_pymupdf(path: Path) -> tuple[list[str], int]:
    with pymupdf.open(path) as document:
        pages = [page.get_text("text", sort=True) for page in document]
        return pages, document.page_count


def _extract_pypdf(path: Path) -> list[str]:
    reader = PdfReader(path, strict=False)
    return [(page.extract_text(extraction_mode="layout") or "") for page in reader.pages]


def extract_pdf(
    source: Path,
    output: Path,
    *,
    fidelity_threshold: float = 0.95,
    overwrite: bool = False,
) -> ExtractionResult:
    source = source.resolve()
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    if not 0.0 <= fidelity_threshold <= 1.0:
        raise ValueError("fidelity_threshold must be between 0 and 1")

    started = perf_counter()
    primary_pages, page_count = _extract_pymupdf(source)
    reference_pages = _extract_pypdf(source)
    primary = PAGE_SEPARATOR.join(primary_pages)
    reference = PAGE_SEPARATOR.join(reference_pages)
    metrics = compare_text(primary, reference)
    pages_with_text = sum(bool(page.strip()) for page in primary_pages)

    warnings: list[str] = []
    if len(reference_pages) != page_count:
        warnings.append(
            f"validator page count differs: primary={page_count}, "
            f"reference={len(reference_pages)}"
        )
    if pages_with_text < page_count:
        warnings.append(
            f"{page_count - pages_with_text} page(s) have no embedded extractable text"
        )
    if metrics.reference_characters == 0:
        warnings.append("validator extracted no text; OCR or manual review is required")

    complete_pages = pages_with_text == page_count
    fidelity_passed = (
        metrics.reference_characters > 0
        and metrics.character_similarity >= fidelity_threshold
        and metrics.token_recall >= fidelity_threshold
    )
    status = "passed" if complete_pages and fidelity_passed else "review"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(primary, encoding="utf-8", newline="\n")

    return ExtractionResult(
        source=str(source),
        output=str(output),
        source_sha256=_sha256(source),
        pages=page_count,
        pages_with_text=pages_with_text,
        raw_characters=len(primary),
        elapsed_seconds=round(perf_counter() - started, 6),
        status=status,
        fidelity_threshold=fidelity_threshold,
        metrics=metrics,
        warnings=tuple(warnings),
    )
