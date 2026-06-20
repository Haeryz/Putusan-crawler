"""High-throughput, validated PDF text extraction."""

from extractor.core import ExtractionResult, extract_pdf, extract_pdf_with_windows_ocr

__all__ = ["ExtractionResult", "extract_pdf", "extract_pdf_with_windows_ocr"]
