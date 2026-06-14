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


def test_store_persists_and_clears_listing_checkpoint(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    checkpoint_key = "listing|prefix=Putusan PN"
    listing_url = "https://example.test/listing/page-7"

    store.save_listing_checkpoint(checkpoint_key, listing_url)

    reloaded = JsonlStore(tmp_path)
    assert reloaded.load_listing_checkpoint(checkpoint_key) == listing_url
    assert reloaded.load_listing_checkpoint("different") is None

    reloaded.clear_listing_checkpoint(checkpoint_key)
    assert reloaded.load_listing_checkpoint(checkpoint_key) is None
    assert reloaded.has_listing_checkpoint_state(checkpoint_key)


def test_store_infers_latest_matching_listing_page_from_log(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    start_url = (
        "https://putusan3.mahkamahagung.go.id/direktori/index/"
        "kategori/peradilan-anak-abh-1.html"
    )
    page_12 = start_url.removesuffix(".html") + "/page/12.html"
    page_13 = start_url.removesuffix(".html") + "/page/13.html"
    store.prepare()
    store.log(f"managed-listing-clicks url={page_12} cases=20 next={page_13}")
    store.log("managed-listing-clicks url=https://example.test/other cases=20 next=None")
    store.log(f"managed-listing-clicks url={page_13} cases=20 next=None")

    assert store.infer_listing_checkpoint_from_log(start_url) == page_13


def test_store_resumes_interrupted_target_from_download_baseline(
    tmp_path: Path,
) -> None:
    store = JsonlStore(tmp_path)

    started = store.begin_or_resume_target("scope", 264, 236)
    resumed = store.begin_or_resume_target("scope", 264, 359)

    assert started.completed == 0
    assert not started.resumed
    assert resumed.completed == 123
    assert resumed.baseline_downloaded == 236
    assert resumed.resumed


def test_completed_target_starts_fresh_next_time(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.begin_or_resume_target("scope", 264, 236)
    store.complete_target("scope")

    next_target = store.begin_or_resume_target("scope", 264, 500)

    assert next_target.completed == 0
    assert next_target.baseline_downloaded == 500
    assert not next_target.resumed


def test_processed_urls_include_semantic_duplicate_exclusions(
    tmp_path: Path,
) -> None:
    store = JsonlStore(tmp_path)
    store.prepare()
    downloaded = "https://example.test/downloaded"
    duplicate = "https://example.test/duplicate"
    store.append(CrawlRecord(status="downloaded", detail_url=downloaded))
    store.append(
        CrawlRecord(
            status="skipped_duplicate_content",
            detail_url=duplicate,
            error="same extracted text as canonical document",
        )
    )

    assert store.excluded_detail_urls() == {duplicate}
    assert store.processed_detail_urls() == {downloaded, duplicate}


def test_store_reads_utf8_bom_prefixed_jsonl(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.prepare()
    detail_url = "https://example.test/bom-record"
    store.success_path.write_text(
        CrawlRecord(status="downloaded", detail_url=detail_url).to_json() + "\n",
        encoding="utf-8-sig",
    )

    assert store.downloaded_detail_urls() == {detail_url}


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
