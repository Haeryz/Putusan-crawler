from pathlib import Path

import pytest

from crawler.storage import CrawlRecord, JsonlStore, sanitize_filename, unique_path, verify_pdf


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


def test_store_deduplicates_downloaded_records_by_detail_url(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.prepare()
    pdf_dir = tmp_path / "pdfs"
    first = pdf_dir / "case.pdf"
    duplicate = pdf_dir / "case-2.pdf"
    first.write_bytes(b"%PDF-1.7\nfirst")
    duplicate.write_bytes(b"%PDF-1.7\nduplicate")
    store.append(
        CrawlRecord(
            status="downloaded",
            detail_url="https://example.test/case",
            pdf_url="https://example.test/pdf/1",
            output_path=str(first),
        )
    )
    store.append(
        CrawlRecord(
            status="downloaded",
            detail_url="https://example.test/case",
            pdf_url="https://example.test/pdf/2",
            output_path=str(duplicate),
        )
    )

    summary = store.deduplicate_downloads()

    assert summary.kept_records == 1
    assert summary.duplicate_records == 1
    assert summary.moved_files == 1
    assert first.exists()
    assert not duplicate.exists()
    assert (tmp_path / "duplicates" / "case-2.pdf").exists()
    assert len(store.success_path.read_text(encoding="utf-8").splitlines()) == 1


def test_store_deduplicates_downloaded_records_by_file_hash(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.prepare()
    pdf_dir = tmp_path / "pdfs"
    first = pdf_dir / "case-a.pdf"
    duplicate = pdf_dir / "case-b.pdf"
    first.write_bytes(b"%PDF-1.7\nsame body")
    duplicate.write_bytes(b"%PDF-1.7\nsame body")
    store.append(
        CrawlRecord(
            status="downloaded",
            detail_url="https://example.test/case-a",
            pdf_url="https://example.test/pdf/a",
            output_path=str(first),
        )
    )
    store.append(
        CrawlRecord(
            status="downloaded",
            detail_url="https://example.test/case-b",
            pdf_url="https://example.test/pdf/b",
            output_path=str(duplicate),
        )
    )

    summary = store.deduplicate_downloads()

    assert summary.kept_records == 1
    assert summary.duplicate_records == 1
    assert first.exists()
    assert (tmp_path / "duplicates" / "case-b.pdf").exists()


def test_store_removes_invalid_download_records_so_they_can_be_retried(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.prepare()
    missing = tmp_path / "pdfs" / "missing.pdf"
    store.append(
        CrawlRecord(
            status="downloaded",
            detail_url="https://example.test/missing",
            output_path=str(missing),
        )
    )

    summary = store.deduplicate_downloads()

    assert summary.invalid_records == 1
    assert store.downloaded_detail_urls() == set()
