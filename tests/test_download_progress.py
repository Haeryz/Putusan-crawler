from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from crawler import cli
from crawler.crawler import (
    BulkDownloadResult,
    CrawlConfig,
    CrawlProgress,
    CrawlSummary,
    PutusanCrawler,
)
from crawler.storage import CrawlRecord
from crawler.parsing import ListingLinks


class _DummyDriver:
    def quit(self) -> None:
        pass


def test_parallel_download_refills_after_failures_until_success_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[CrawlProgress] = []
    urls = [f"https://putusan3.mahkamahagung.go.id/case/{index}" for index in range(5)]
    crawler = PutusanCrawler(
        CrawlConfig(
            out_dir=tmp_path,
            browser_backend="undetected-chrome",
            parallel_downloads=2,
            target_downloads=3,
            progress_callback=events.append,
        )
    )

    monkeypatch.setattr(crawler, "_launch_undetected_chrome", _DummyDriver)
    monkeypatch.setattr(
        crawler,
        "_collect_listing_detail_urls",
        lambda driver, start_url, target_count: (urls, {"user_agent": "test", "cookies": []}),
    )

    def download(session_state, detail_url, path_lock):
        index = int(detail_url.rsplit("/", 1)[-1])
        if index < 2:
            return BulkDownloadResult(
                CrawlRecord(status="error", detail_url=detail_url, error="test failure"),
                elapsed_seconds=0.01,
            )
        output_path = tmp_path / "pdfs" / f"{index}.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"%PDF-1.4\n%%EOF")
        return BulkDownloadResult(
            CrawlRecord(
                status="downloaded",
                detail_url=detail_url,
                output_path=str(output_path),
            ),
            elapsed_seconds=0.01,
            bytes_downloaded=14,
        )

    monkeypatch.setattr(crawler, "_download_case_parallel", download)

    summary = crawler.run()

    assert summary.downloaded == 3
    assert summary.failed_downloads == 2
    assert summary.metrics["attempted"] == 5
    assert [event.successful for event in events if event.phase == "downloaded"] == [1, 2, 3]
    assert events[-1].phase == "complete"


def test_skipped_candidate_does_not_advance_success_progress(tmp_path: Path) -> None:
    events: list[CrawlProgress] = []
    crawler = PutusanCrawler(
        CrawlConfig(out_dir=tmp_path, progress_callback=events.append)
    )

    crawler._report_candidate("https://putusan3.mahkamahagung.go.id/case/no-pdf")
    crawler._report_record(
        CrawlRecord(
            status="skipped_no_pdf",
            detail_url="https://putusan3.mahkamahagung.go.id/case/no-pdf",
            error="no PDF download link found",
        )
    )

    assert events[-1].successful == 0
    assert events[-1].skipped == 1
    assert events[-1].attempted == 1


def test_managed_parallel_batches_only_schedule_remaining_successes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    urls = [f"https://putusan3.mahkamahagung.go.id/case/{index}" for index in range(3)]
    crawler = PutusanCrawler(
        CrawlConfig(
            out_dir=tmp_path,
            target_downloads=1,
            parallel_downloads=4,
        )
    )
    crawler.store.prepare()
    batches: list[list[str]] = []

    class Page:
        url = "https://putusan3.mahkamahagung.go.id/listing"

    monkeypatch.setattr(crawler, "_goto_and_wait", lambda page, url: None)
    monkeypatch.setattr(crawler, "_safe_page_content", lambda page: "<html></html>")
    monkeypatch.setattr(
        crawler,
        "_parse_listing",
        lambda html, base_url: ListingLinks(urls, None),
    )

    def fetch(page, batch):
        batches.append(list(batch))
        return list(batch)

    def record(detail_url):
        if detail_url == urls[0]:
            return CrawlRecord(status="error", detail_url=detail_url, error="test failure")
        output_path = tmp_path / "pdfs" / "success.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"%PDF-1.4\n%%EOF")
        return CrawlRecord(
            status="downloaded",
            detail_url=detail_url,
            output_path=str(output_path),
        )

    monkeypatch.setattr(crawler, "_fetch_detail_pdf_batch", fetch)
    monkeypatch.setattr(crawler, "_record_fast_fetch_result", record)

    summary = crawler._crawl_listing_by_browser_fetch(
        Page(),
        Page.url,
        set(),
        set(),
        0,
        [],
        0,
    )

    assert summary.downloaded == 1
    assert summary.failed_downloads == 1
    assert batches == [[urls[0]], [urls[1]]]


def test_listing_checkpoint_resumes_from_interrupted_page(tmp_path: Path) -> None:
    start_url = "https://putusan3.mahkamahagung.go.id/direktori/index/page-1.html"
    resumed_url = "https://putusan3.mahkamahagung.go.id/direktori/index/page-8.html"
    config = CrawlConfig(out_dir=tmp_path, start_url=start_url)
    crawler = PutusanCrawler(config)
    crawler.store.prepare()
    crawler._checkpoint_listing_page(resumed_url)

    restarted = PutusanCrawler(config)

    assert restarted._initial_listing_url() == resumed_url


def test_listing_checkpoint_migrates_from_existing_run_log(tmp_path: Path) -> None:
    start_url = (
        "https://putusan3.mahkamahagung.go.id/direktori/index/"
        "kategori/peradilan-anak-abh-1.html"
    )
    resumed_url = start_url.removesuffix(".html") + "/page/27.html"
    crawler = PutusanCrawler(CrawlConfig(out_dir=tmp_path, start_url=start_url))
    crawler.store.prepare()
    crawler.store.log(
        f"managed-listing-clicks url={resumed_url} cases=16 next=None"
    )

    assert crawler._initial_listing_url() == resumed_url
    assert crawler.store.load_listing_checkpoint(
        crawler._listing_checkpoint_key()
    ) == resumed_url


def test_restart_listing_discards_checkpoint(tmp_path: Path) -> None:
    start_url = "https://putusan3.mahkamahagung.go.id/direktori/index/page-1.html"
    resumed_url = "https://putusan3.mahkamahagung.go.id/direktori/index/page-8.html"
    config = CrawlConfig(out_dir=tmp_path, start_url=start_url)
    crawler = PutusanCrawler(config)
    crawler.store.prepare()
    crawler._checkpoint_listing_page(resumed_url)

    restarted = PutusanCrawler(
        CrawlConfig(
            out_dir=tmp_path,
            start_url=start_url,
            resume_listing=False,
        )
    )

    assert restarted._initial_listing_url() == start_url
    assert restarted.store.load_listing_checkpoint(
        restarted._listing_checkpoint_key()
    ) is None


def test_completed_listing_does_not_restore_old_log_checkpoint(
    tmp_path: Path,
) -> None:
    start_url = (
        "https://putusan3.mahkamahagung.go.id/direktori/index/"
        "kategori/peradilan-anak-abh-1.html"
    )
    old_url = start_url.removesuffix(".html") + "/page/8.html"
    crawler = PutusanCrawler(CrawlConfig(out_dir=tmp_path, start_url=start_url))
    crawler.store.prepare()
    crawler.store.log(f"managed-listing-clicks url={old_url} cases=20 next=None")
    crawler._advance_listing_checkpoint(None)

    restarted = PutusanCrawler(
        CrawlConfig(out_dir=tmp_path, start_url=start_url)
    )

    assert restarted._initial_listing_url() == start_url


def test_crawler_summary_includes_downloads_from_interrupted_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = CrawlConfig(out_dir=tmp_path, target_downloads=3)
    first = PutusanCrawler(config)
    first.store.prepare()
    first.store.begin_or_resume_target(first._target_scope_key(), 3, 0)

    existing_path = tmp_path / "pdfs" / "existing.pdf"
    existing_path.write_bytes(b"%PDF-1.4\n%%EOF")
    first.store.append(
        CrawlRecord(
            status="downloaded",
            detail_url="https://putusan3.mahkamahagung.go.id/case/existing",
            output_path=str(existing_path),
        )
    )

    restarted = PutusanCrawler(config)
    monkeypatch.setattr(
        restarted,
        "_run_managed_chrome",
        lambda: restarted._summary([], 0),
    )

    summary = restarted.run()

    assert summary.downloaded == 1
    assert summary.target == 3
    assert summary.metrics["resumed_downloads"] == 1
    assert summary.metrics["downloaded_this_run"] == 0


def test_rich_download_progress_shows_remaining_and_outcome_counts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeCrawler:
        def __init__(self, config: CrawlConfig) -> None:
            self.config = config

        def run(self) -> CrawlSummary:
            callback = self.config.progress_callback
            assert callback is not None
            callback(CrawlProgress("downloading", 1, 0, 0, 0, "https://example.test/1"))
            callback(CrawlProgress("failed", 1, 0, 1, 0, "https://example.test/1"))
            callback(CrawlProgress("downloading", 2, 0, 1, 0, "https://example.test/2"))
            callback(CrawlProgress("downloaded", 2, 1, 1, 0, "https://example.test/2"))
            callback(CrawlProgress("complete", 2, 1, 1, 0, message="Crawl finished"))
            return CrawlSummary(1, 1, 1, [tmp_path / "2.pdf"])

    monkeypatch.setattr(cli, "PutusanCrawler", FakeCrawler)
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=160)

    cli._run_download_with_progress(
        console,
        CrawlConfig(out_dir=tmp_path, target_downloads=1),
        "up to 1 PDF(s)",
    )

    rendered = output.getvalue()
    assert "0 left" in rendered
    assert "ok 1" in rendered
    assert "fail 1" in rendered
    assert "tried 2" in rendered
    assert "ETA" in rendered


def test_rich_download_progress_starts_from_resumed_target_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeCrawler:
        def __init__(self, config: CrawlConfig) -> None:
            self.config = config

        def run(self) -> CrawlSummary:
            callback = self.config.progress_callback
            assert callback is not None
            callback(
                CrawlProgress(
                    "target_resumed",
                    attempted=0,
                    successful=123,
                    failed=0,
                    skipped=0,
                    message="Resuming target with 123 already completed",
                )
            )
            return CrawlSummary(123, 264, 0, [])

    monkeypatch.setattr(cli, "PutusanCrawler", FakeCrawler)
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=160)

    cli._run_download_with_progress(
        console,
        CrawlConfig(out_dir=tmp_path, target_downloads=264),
        "up to 264 PDF(s)",
    )

    rendered = output.getvalue()
    assert "123/264" in rendered
    assert "141 left" in rendered
    assert "ok 123" in rendered


def test_rich_spinner_returns_action_result() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=True)

    result = cli._run_with_spinner(
        console,
        "Scanning",
        lambda: "finished",
        enabled=True,
    )

    assert result == "finished"
