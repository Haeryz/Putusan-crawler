from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import pymupdf
from pypdf import PdfReader

from extractor.metrics import FidelityMetrics, compare_text

PAGE_SEPARATOR = "\n\f\n"

_WINDOWS_OCR_SCRIPT = r"""
param([string]$ImageListJson)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.Streams.IRandomAccessStreamWithContentType, Windows.Storage.Streams, ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType=WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrResult, Windows.Media.Ocr, ContentType=WindowsRuntime] | Out-Null
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() |
    Where-Object {
        $_.Name -eq "AsTask" -and
        $_.IsGenericMethod -and
        $_.GetGenericArguments().Count -eq 1 -and
        $_.GetParameters().Count -eq 1 -and
        $_.ReturnType.Name -eq "Task``1"
    } |
    Select-Object -First 1
function Await-Op($op, [type]$type) {
    $task = $asTask.MakeGenericMethod($type).Invoke($null, @($op))
    $task.GetAwaiter().GetResult()
}
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) {
    throw "No Windows OCR engine is available for the current user profile."
}
$paths = ConvertFrom-Json -InputObject $ImageListJson
$texts = @()
foreach ($path in $paths) {
    $file = Await-Op ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
    $stream = Await-Op ($file.OpenReadAsync()) ([Windows.Storage.Streams.IRandomAccessStreamWithContentType])
    $decoder = Await-Op ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap = Await-Op ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $result = Await-Op ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
    $texts += $result.Text
}
ConvertTo-Json -Compress -InputObject $texts
"""


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
    lowest_page_character_similarity: float
    highest_page_character_error_rate: float
    highest_page_content_character_error_rate: float
    highest_page_word_error_rate: float
    lowest_page_token_recall: float
    pages_below_threshold: tuple[int, ...]
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
        # Content-stream order avoids interleaving diagonal watermark glyphs
        # into body lines, which occurs when geometrically sorting these PDFs.
        pages = [page.get_text("text", sort=False) for page in document]
        return pages, document.page_count


def _extract_pypdf(path: Path) -> list[str]:
    reader = PdfReader(path, strict=False)
    return [(page.extract_text() or "") for page in reader.pages]


def _extract_windows_ocr(path: Path, *, dpi: int = 150) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="sinergi-ocr-") as temporary:
        temporary_path = Path(temporary)
        image_paths: list[Path] = []
        with pymupdf.open(path) as document:
            matrix = pymupdf.Matrix(dpi / 72, dpi / 72)
            for index, page in enumerate(document, start=1):
                image_path = temporary_path / f"page-{index:04d}.png"
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                pixmap.save(image_path)
                image_paths.append(image_path)
        script_path = temporary_path / "ocr.ps1"
        script_path.write_text(_WINDOWS_OCR_SCRIPT, encoding="utf-8", newline="\n")
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                json.dumps([str(path) for path in image_paths]),
            ],
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        result = json.loads(completed.stdout)
        if isinstance(result, str):
            return [result]
        return [str(page) for page in result]


def extract_pdf_with_windows_ocr(
    source: Path,
    output: Path,
    *,
    overwrite: bool = False,
    dpi: int = 150,
) -> ExtractionResult:
    source = source.resolve()
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    started = perf_counter()
    pages = _extract_windows_ocr(source, dpi=dpi)
    raw_text = PAGE_SEPARATOR.join(pages)
    page_count = len(pages)
    pages_with_text = sum(bool(page.strip()) for page in pages)
    metrics = compare_text(raw_text, raw_text)
    warnings = (
        "OCR fallback used; text is generated from rendered page images and "
        "requires review"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(raw_text, encoding="utf-8", errors="replace", newline="\n")

    return ExtractionResult(
        source=str(source),
        output=str(output),
        source_sha256=_sha256(source),
        pages=page_count,
        pages_with_text=pages_with_text,
        raw_characters=len(raw_text),
        elapsed_seconds=round(perf_counter() - started, 6),
        status="review",
        fidelity_threshold=0.0,
        metrics=metrics,
        lowest_page_character_similarity=1.0 if pages else 0.0,
        highest_page_character_error_rate=0.0,
        highest_page_content_character_error_rate=0.0,
        highest_page_word_error_rate=0.0,
        lowest_page_token_recall=1.0 if pages else 0.0,
        pages_below_threshold=(),
        warnings=(warnings,),
    )


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
    reference_pages, page_count = _extract_pymupdf(source)
    primary_pages = _extract_pypdf(source)
    primary = PAGE_SEPARATOR.join(primary_pages)
    reference = PAGE_SEPARATOR.join(reference_pages)
    metrics = compare_text(primary, reference)
    page_metrics = [
        compare_text(primary_page, reference_page)
        for primary_page, reference_page in zip(primary_pages, reference_pages)
    ]
    pages_below_threshold = tuple(
        index
        for index, page_metric in enumerate(page_metrics, start=1)
        if page_metric.content_character_accuracy < fidelity_threshold
    )
    lowest_page_character_similarity = min(
        (page_metric.character_similarity for page_metric in page_metrics),
        default=0.0,
    )
    lowest_page_token_recall = min(
        (page_metric.token_recall for page_metric in page_metrics),
        default=0.0,
    )
    highest_page_character_error_rate = max(
        (page_metric.character_error_rate for page_metric in page_metrics),
        default=0.0,
    )
    highest_page_content_character_error_rate = max(
        (page_metric.content_character_error_rate for page_metric in page_metrics),
        default=0.0,
    )
    highest_page_word_error_rate = max(
        (page_metric.word_error_rate for page_metric in page_metrics),
        default=0.0,
    )
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
    if pages_below_threshold:
        warnings.append(
            f"{len(pages_below_threshold)} page(s) fall below the fidelity threshold"
        )

    complete_pages = pages_with_text == page_count
    fidelity_passed = (
        metrics.reference_characters > 0
        and metrics.content_character_accuracy >= fidelity_threshold
        and not pages_below_threshold
    )
    status = "passed" if complete_pages and fidelity_passed else "review"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(primary, encoding="utf-8", errors="replace", newline="\n")

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
        lowest_page_character_similarity=lowest_page_character_similarity,
        highest_page_character_error_rate=highest_page_character_error_rate,
        highest_page_content_character_error_rate=(
            highest_page_content_character_error_rate
        ),
        highest_page_word_error_rate=highest_page_word_error_rate,
        lowest_page_token_recall=lowest_page_token_recall,
        pages_below_threshold=pages_below_threshold,
        warnings=tuple(warnings),
    )
