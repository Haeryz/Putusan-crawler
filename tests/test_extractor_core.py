from pathlib import Path

import pymupdf

from extractor.core import PAGE_SEPARATOR, extract_pdf


def _make_pdf(path: Path, pages: list[str]) -> None:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_extract_pdf_writes_all_pages_and_passes_validation(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "source.txt"
    _make_pdf(source, ["First page exact text.", "Second page exact text."])

    result = extract_pdf(source, output)

    assert result.status == "passed"
    assert result.pages == 2
    assert result.pages_with_text == 2
    assert result.pages_below_threshold == ()
    assert output.read_text(encoding="utf-8") == (
        f"First page exact text.{PAGE_SEPARATOR}Second page exact text."
    )


def test_extract_pdf_marks_image_only_page_for_review(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "source.txt"
    _make_pdf(source, [""])

    result = extract_pdf(source, output)

    assert result.status == "review"
    assert result.pages_with_text == 0
    assert result.pages_below_threshold == ()
    assert "OCR or manual review is required" in " ".join(result.warnings)
