from pathlib import Path

import pytest

from sinergi.storage import CrawlRecord, JsonlStore, sanitize_filename, unique_path, verify_pdf


def test_sanitize_filename_replaces_path_separators() -> None:
    assert sanitize_filename("42/Pid.Sus/2026/PN_Pya.pdf", "fallback") == (
        "42_Pid.Sus_2026_PN_Pya.pdf"
    )


def test_unique_path_adds_suffix(tmp_path: Path) -> None:
    existing = tmp_path / "case.pdf"
    existing.write_bytes(b"%PDF-1")

    assert unique_path(existing) == tmp_path / "case-2.pdf"


def test_verify_pdf_accepts_pdf_magic(tmp_path: Path) -> None:
    pdf = tmp_path / "case.pdf"
    pdf.write_bytes(b"%PDF-1.7\ncontent")

    verify_pdf(pdf)


def test_verify_pdf_rejects_html(tmp_path: Path) -> None:
    html = tmp_path / "case.pdf"
    html.write_text("<html>not pdf</html>", encoding="utf-8")

    with pytest.raises(ValueError, match="not a PDF"):
        verify_pdf(html)


def test_store_reads_downloaded_detail_urls(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.prepare()
    store.append(CrawlRecord(status="downloaded", detail_url="https://example.test/case"))

    assert store.downloaded_detail_urls() == {"https://example.test/case"}
