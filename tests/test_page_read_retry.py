import json
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError

import crawler.crawler as crawler_module
from crawler.crawler import CrawlConfig, PutusanCrawler, RateLimitedError
from crawler.storage import CrawlRecord


class FlakyContentPage:
    url = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/peradilan-anak-abh-1.html"

    def __init__(self, html: str, failures: int = 1) -> None:
        self.html = html
        self.failures = failures
        self.content_calls = 0
        self.load_state_waits = 0
        self.timeout_waits = 0

    def content(self) -> str:
        self.content_calls += 1
        if self.content_calls <= self.failures:
            raise PlaywrightError(
                "Page.content: Unable to retrieve content because the page is "
                "navigating and changing the content."
            )
        return self.html

    def title(self) -> str:
        return ""

    def wait_for_load_state(self, state: str, timeout: int) -> None:
        assert state == "domcontentloaded"
        assert timeout > 0
        self.load_state_waits += 1

    def wait_for_timeout(self, timeout: int) -> None:
        assert timeout > 0
        self.timeout_waits += 1


def test_wait_for_accessible_page_retries_transient_content_read(tmp_path: Path) -> None:
    page = FlakyContentPage(
        """
        <a href="/direktori/putusan/pn-1.html">
          Putusan PN GARUT 1/Pid.Sus-Anak/2026/PN Grt
        </a>
        """
    )
    crawler = PutusanCrawler(
        CrawlConfig(out_dir=tmp_path, timeout_seconds=1, manual_clearance_timeout_seconds=1)
    )

    crawler._wait_for_accessible_page(page)

    assert page.content_calls == 2
    assert page.load_state_waits == 1
    assert page.timeout_waits == 1


class ClearingChallengePage:
    url = "https://putusan3.mahkamahagung.go.id/direktori/putusan/case.html"

    def __init__(self) -> None:
        self.content_calls = 0
        self.waits: list[int] = []

    def content(self) -> str:
        self.content_calls += 1
        if self.content_calls == 1:
            return "<html><title>Just a moment</title><p>Verify you are human</p></html>"
        return """
        <a href="/direktori/download_file/hash/pdf/case">case.pdf</a>
        """

    def title(self) -> str:
        return "Just a moment" if self.content_calls == 1 else "Putusan PN Test"

    def wait_for_timeout(self, timeout: int) -> None:
        self.waits.append(timeout)


def test_wait_for_accessible_page_cools_down_after_challenge_clears(
    tmp_path: Path, monkeypatch
) -> None:
    page = ClearingChallengePage()
    crawler = PutusanCrawler(
        CrawlConfig(
            out_dir=tmp_path,
            timeout_seconds=1,
            manual_clearance_timeout_seconds=1,
            challenge_cooldown_seconds=30,
        )
    )
    monkeypatch.setattr(crawler, "_capture_live_page_view", lambda *args: None)

    crawler._wait_for_accessible_page(page)

    assert page.waits == [2_000, 30_000]
    assert "challenge-cleared cooldown_seconds=30" in (
        tmp_path / "run.log"
    ).read_text(encoding="utf-8")


class ScreenshotPage:
    url = "https://putusan3.mahkamahagung.go.id/direktori/putusan/case.html"

    def screenshot(self, **kwargs) -> bytes:
        assert kwargs["full_page"] is False
        assert kwargs["timeout"] > 0
        return b"\x89PNG\r\n\x1a\nfake"

    def title(self) -> str:
        return "Putusan PN Test"


def test_capture_live_page_view_writes_stable_dashboard_frame(tmp_path: Path) -> None:
    crawler = PutusanCrawler(
        CrawlConfig(out_dir=tmp_path, timeout_seconds=1, manual_clearance_timeout_seconds=1)
    )

    crawler._capture_live_page_view(ScreenshotPage(), "test-frame")

    image_path = tmp_path / ".hermes" / "browser_view.png"
    meta_path = tmp_path / ".hermes" / "browser_view.json"
    assert image_path.read_bytes() == b"\x89PNG\r\n\x1a\nfake"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["image_path"] == str(image_path)
    assert meta["page_url"] == ScreenshotPage.url
    assert meta["title"] == "Putusan PN Test"
    assert meta["label"] == "test-frame"


def test_fast_fetch_invalid_pdf_redirect_becomes_error_record(tmp_path: Path) -> None:
    crawler = PutusanCrawler(
        CrawlConfig(out_dir=tmp_path, timeout_seconds=1, manual_clearance_timeout_seconds=1)
    )

    record = crawler._record_fast_fetch_result(
        {
            "status": "downloaded",
            "detailUrl": "https://putusan3.mahkamahagung.go.id/direktori/putusan/case.html",
            "pdfUrl": "https://putusan3.mahkamahagung.go.id/direktori/putusan/case",
            "title": "Putusan PN Test",
            "filename": "case.pdf",
            "base64": "",
        }
    )

    assert record.status == "error"
    assert record.detail_url == "https://putusan3.mahkamahagung.go.id/direktori/putusan/case.html"
    assert "refusing non-Putusan PDF URL" in (record.error or "")


class FlakyEvaluatePage:
    def __init__(self, failures: int = 1) -> None:
        self.failures = failures
        self.evaluate_calls = 0
        self.load_state_waits = 0
        self.timeout_waits = 0

    def evaluate(self, expression: str, arg=None):
        self.evaluate_calls += 1
        if self.evaluate_calls <= self.failures:
            raise PlaywrightError(
                "Page.evaluate: Execution context was destroyed, most likely because "
                "of a navigation."
            )
        return {"ok": True, "expression": expression, "arg": arg}

    def wait_for_load_state(self, state: str, timeout: int) -> None:
        assert state == "domcontentloaded"
        assert timeout > 0
        self.load_state_waits += 1

    def wait_for_timeout(self, timeout: int) -> None:
        assert timeout > 0
        self.timeout_waits += 1


def test_safe_page_evaluate_retries_transient_navigation_error(tmp_path: Path) -> None:
    page = FlakyEvaluatePage()
    crawler = PutusanCrawler(
        CrawlConfig(out_dir=tmp_path, timeout_seconds=1, manual_clearance_timeout_seconds=1)
    )

    result = crawler._safe_page_evaluate(page, "() => 1", {"url": "x"})

    assert result == {"ok": True, "expression": "() => 1", "arg": {"url": "x"}}
    assert page.evaluate_calls == 2
    assert page.load_state_waits == 1
    assert page.timeout_waits == 1


class InterruptedNavigationPage:
    def __init__(self, failures: int = 1) -> None:
        self.failures = failures
        self.goto_calls: list[str] = []
        self.url = "about:blank"
        self.timeout_waits = 0

    def goto(self, url: str, wait_until: str, timeout: int) -> None:
        assert timeout > 0
        self.goto_calls.append(url)
        if url == "about:blank":
            self.url = url
            return
        if self.failures:
            self.failures -= 1
            self.url = "chrome-error://chromewebdata/"
            raise PlaywrightError(
                f'Page.goto: Navigation to "{url}" is interrupted by another '
                'navigation to "chrome-error://chromewebdata/"'
            )
        self.url = url

    def content(self) -> str:
        return "<html><body>listing loaded</body></html>"

    def title(self) -> str:
        return "Putusan listing"

    def wait_for_timeout(self, timeout: int) -> None:
        assert timeout > 0
        self.timeout_waits += 1


def test_goto_retries_after_chrome_error_interrupted_navigation(
    tmp_path: Path,
) -> None:
    page = InterruptedNavigationPage()
    crawler = PutusanCrawler(
        CrawlConfig(
            out_dir=tmp_path,
            timeout_seconds=1,
            manual_clearance_timeout_seconds=1,
            retry_attempts=3,
        )
    )
    target = (
        "https://putusan3.mahkamahagung.go.id/direktori/index/"
        "kategori/peradilan-anak-abh-1/page/27.html"
    )

    crawler._goto_and_wait(page, target)

    assert page.goto_calls == [target, "about:blank", target]
    assert page.url == target
    assert page.timeout_waits == 1
    assert "navigation-retry attempt=1" in (tmp_path / "run.log").read_text(
        encoding="utf-8"
    )


def test_goto_stops_after_navigation_retry_limit(tmp_path: Path) -> None:
    page = InterruptedNavigationPage(failures=5)
    crawler = PutusanCrawler(
        CrawlConfig(
            out_dir=tmp_path,
            timeout_seconds=1,
            manual_clearance_timeout_seconds=1,
            retry_attempts=2,
        )
    )
    target = (
        "https://putusan3.mahkamahagung.go.id/direktori/index/"
        "kategori/peradilan-anak-abh-1/page/27.html"
    )

    try:
        crawler._goto_and_wait(page, target)
    except PlaywrightError as exc:
        assert "interrupted by another navigation" in str(exc)
    else:
        raise AssertionError("expected navigation failure")

    assert page.goto_calls == [target, "about:blank", target]


class RateLimitedPdfPage:
    def evaluate(self, expression: str, arg=None):
        return {
            "ok": False,
            "status": 429,
            "url": arg,
            "contentType": "text/html",
            "retryAfter": "45",
            "base64": "",
        }


def test_pdf_fetch_429_preserves_retry_after(tmp_path: Path) -> None:
    crawler = PutusanCrawler(CrawlConfig(out_dir=tmp_path))
    pdf_url = (
        "https://putusan3.mahkamahagung.go.id/direktori/"
        "download_file/hash/pdf/case"
    )

    try:
        crawler._save_pdf_with_page_fetch(
            RateLimitedPdfPage(),
            pdf_url,
            tmp_path / "case.pdf",
        )
    except RateLimitedError as exc:
        assert exc.retry_after_seconds == 45
    else:
        raise AssertionError("expected HTTP 429 to raise RateLimitedError")


class ListingPage:
    url = (
        "https://putusan3.mahkamahagung.go.id/direktori/index/"
        "kategori/perdagangan-orang-1.html"
    )


def test_click_retry_returns_to_listing_and_backs_off_after_429(
    tmp_path: Path, monkeypatch
) -> None:
    crawler = PutusanCrawler(
        CrawlConfig(
            out_dir=tmp_path,
            retry_attempts=2,
            rate_limit_backoff_seconds=60,
        )
    )
    detail_url = (
        "https://putusan3.mahkamahagung.go.id/direktori/putusan/case.html"
    )
    click_calls: list[str] = []
    returned_to: list[str] = []
    sleeps: list[float] = []
    attempts = 0

    def click(page, url: str) -> None:
        click_calls.append(url)

    def download(context, page, url: str) -> CrawlRecord:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimitedError("PDF fetch failed with HTTP 429", 45)
        return CrawlRecord(status="downloaded", detail_url=url)

    monkeypatch.setattr(crawler, "_click_detail_link", click)
    monkeypatch.setattr(crawler, "_download_current_detail", download)
    monkeypatch.setattr(
        crawler,
        "_return_to_listing",
        lambda page, url: returned_to.append(url),
    )
    monkeypatch.setattr(crawler_module.time, "sleep", sleeps.append)

    record = crawler._download_case_by_click(None, ListingPage(), detail_url)

    assert record.status == "downloaded"
    assert click_calls == [detail_url, detail_url]
    assert returned_to == [ListingPage.url]
    assert sleeps == [60]
