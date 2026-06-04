from pathlib import Path

from playwright.sync_api import Error as PlaywrightError

from crawler.crawler import CrawlConfig, PutusanCrawler


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
