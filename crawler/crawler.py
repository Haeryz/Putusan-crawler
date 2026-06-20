from __future__ import annotations

import time
import base64
import json
import re
import shutil
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Callable
from urllib.parse import urlparse
from urllib.request import urlopen

from playwright.sync_api import (
    BrowserContext,
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from .parsing import (
    looks_like_challenge,
    parse_listing,
    parse_listing_last_page_index,
    parse_pdf_link,
    parse_title,
)
from .storage import CrawlRecord, JsonlStore, sanitize_filename, unique_path, verify_pdf

DEFAULT_START_URL = (
    "https://putusan3.mahkamahagung.go.id/direktori/index/"
    "kategori/pidana-khusus-1.html"
)
ALLOWED_HOST = "putusan3.mahkamahagung.go.id"


class ChallengeBlockedError(RuntimeError):
    """Raised when a challenge page does not clear through normal browser execution."""


class RateLimitedError(RuntimeError):
    """Raised when Putusan MA asks the crawler to reduce its request rate."""

    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class CrawlConfig:
    start_url: str = DEFAULT_START_URL
    out_dir: Path = Path("downloads")
    profile_dir: Path = Path(".browser-profile")
    target_downloads: int = 10
    download_all: bool = False
    headless: bool = True
    timeout_seconds: int = 120
    max_candidates: int | None = None
    retry_attempts: int = 3
    delay_seconds: float = 0.0
    rate_limit_backoff_seconds: float = 30.0
    challenge_cooldown_seconds: float = 0.0
    browser_channel: str | None = None
    browser_backend: str = "managed-chrome"
    chrome_version_main: int | None = None
    cdp_port: int | None = None
    parallel_downloads: int = 1
    keep_browser_open_on_error: bool = False
    debug_hold_seconds: int = 0
    detail_urls: tuple[str, ...] = ()
    crawl_listing: bool = True
    chrome_user_data_dir: Path | None = None
    chrome_profile: str = "Profile 4"
    manual_clearance_timeout_seconds: int = 300
    refresh_profile_snapshot: bool = True
    fast_fetch_timeout_seconds: int = 15
    count_parallel_pages: int = 16
    case_title_prefix: str | None = None
    skip_unpublished_listing_items: bool = True
    resume_listing: bool = True
    resume_target: bool = True
    progress_callback: Callable[["CrawlProgress"], None] | None = None


@dataclass(frozen=True)
class CrawlSummary:
    downloaded: int
    target: int | None
    failed_downloads: int
    output_paths: list[Path]
    elapsed_seconds: float | None = None
    metrics: dict[str, float | int] = field(default_factory=dict)


@dataclass(frozen=True)
class InventoryPage:
    page_index: int
    listing_url: str
    downloadable: int
    already_downloaded: int
    remaining: int
    detail_urls: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CrawlInventory:
    total_downloadable: int
    pages_scanned: int
    already_downloaded: int
    remaining: int
    pages: list[InventoryPage]


@dataclass(frozen=True)
class BulkDownloadResult:
    record: CrawlRecord
    elapsed_seconds: float
    bytes_downloaded: int = 0


@dataclass(frozen=True)
class CrawlProgress:
    phase: str
    attempted: int
    successful: int
    failed: int
    skipped: int
    detail_url: str | None = None
    message: str | None = None


class PutusanCrawler:
    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self.store = JsonlStore(config.out_dir)
        self.pdf_dir = config.out_dir / "pdfs"
        self._progress_attempted = 0
        self._progress_successful = 0
        self._progress_failed = 0
        self._progress_skipped = 0
        self._started_at: float | None = None
        self._target_completed_before = 0
        self._target_key: str | None = None

    def run(self) -> CrawlSummary:
        self._started_at = time.perf_counter()
        self.store.prepare()
        self._prepare_target_progress()
        self._report_progress(
            "target_resumed" if self._target_completed_before else "starting",
            message=(
                f"Resuming target with {self._target_completed_before} already completed"
                if self._target_completed_before
                else "Preparing browser and crawl state"
            ),
        )
        if (
            not self.config.download_all
            and self._target_completed_before >= self.config.target_downloads
        ):
            return self._summary([], 0)
        if self.config.browser_backend == "managed-chrome":
            return self._run_managed_chrome()
        if self.config.browser_backend == "undetected-chrome":
            return self._run_undetected_chrome()
        if self.config.browser_backend == "playwright":
            return self._run_playwright()
        if self.config.browser_backend == "playwright-cdp":
            return self._run_playwright_cdp()
        raise ValueError(f"unknown browser backend: {self.config.browser_backend}")

    def _target_reached(self, output_paths: list[Path]) -> bool:
        return (
            not self.config.download_all
            and self._target_completed_before + len(output_paths)
            >= self.config.target_downloads
        )

    def _summary_target(self) -> int | None:
        return None if self.config.download_all else self.config.target_downloads

    def _target_scope_key(self) -> str:
        return self._listing_checkpoint_key()

    def _prepare_target_progress(self) -> None:
        if self.config.download_all:
            return
        self._target_key = self._target_scope_key()
        progress = self.store.begin_or_resume_target(
            self._target_key,
            self.config.target_downloads,
            len(self.store.downloaded_detail_urls()),
            force_new=not self.config.resume_target,
        )
        self._target_completed_before = progress.completed
        self._progress_successful = progress.completed
        if progress.resumed:
            self.store.log(
                f"target-resume target={progress.target} completed={progress.completed} "
                f"remaining={max(0, progress.target - progress.completed)}"
            )

    def _remaining_target(self, output_paths: list[Path]) -> int:
        if self.config.download_all:
            return 0
        return max(
            0,
            self.config.target_downloads
            - self._target_completed_before
            - len(output_paths),
        )

    def _complete_target_if_satisfied(self, downloaded: int) -> None:
        if (
            self._target_key is not None
            and not self.config.download_all
            and downloaded >= self.config.target_downloads
        ):
            self.store.complete_target(self._target_key)

    def _listing_checkpoint_key(self) -> str:
        return (
            f"{self.config.start_url}|"
            f"prefix={self.config.case_title_prefix or ''}|"
            f"skip_unpublished={self.config.skip_unpublished_listing_items}"
        )

    def _initial_listing_url(self) -> str | None:
        if not self.config.crawl_listing:
            return None
        checkpoint_key = self._listing_checkpoint_key()
        if not self.config.resume_listing:
            self.store.clear_listing_checkpoint(checkpoint_key)
            return self.config.start_url
        listing_url = self.store.load_listing_checkpoint(checkpoint_key)
        if (
            not listing_url
            and not self.store.has_listing_checkpoint_state(checkpoint_key)
        ):
            listing_url = self.store.infer_listing_checkpoint_from_log(
                self.config.start_url
            )
            if listing_url and _is_allowed_listing_url(listing_url):
                self.store.save_listing_checkpoint(checkpoint_key, listing_url)
                self.store.log(f"listing-resume-migrated url={listing_url}")
        if listing_url and _is_allowed_listing_url(listing_url):
            self.store.log(f"listing-resume url={listing_url}")
            self._report_progress("resuming", detail_url=listing_url)
            return listing_url
        return self.config.start_url

    def _checkpoint_listing_page(self, listing_url: str) -> None:
        self.store.save_listing_checkpoint(self._listing_checkpoint_key(), listing_url)

    def _advance_listing_checkpoint(self, next_url: str | None) -> None:
        checkpoint_key = self._listing_checkpoint_key()
        if next_url:
            self.store.save_listing_checkpoint(checkpoint_key, next_url)
        else:
            self.store.clear_listing_checkpoint(checkpoint_key)

    def _report_progress(
        self,
        phase: str,
        *,
        detail_url: str | None = None,
        message: str | None = None,
    ) -> None:
        callback = self.config.progress_callback
        if callback is None:
            return
        callback(
            CrawlProgress(
                phase=phase,
                attempted=self._progress_attempted,
                successful=self._progress_successful,
                failed=self._progress_failed,
                skipped=self._progress_skipped,
                detail_url=detail_url,
                message=message,
            )
        )

    def _report_candidate(self, detail_url: str) -> None:
        self._progress_attempted += 1
        self._report_progress("downloading", detail_url=detail_url)

    def _report_record(self, record: CrawlRecord) -> None:
        if (
            record.status == "downloaded"
            and record.output_path
            and Path(record.output_path).is_file()
        ):
            self._progress_successful += 1
            phase = "downloaded"
        elif record.status == "error":
            self._progress_failed += 1
            phase = "failed"
        else:
            self._progress_skipped += 1
            phase = "skipped"
        self._report_progress(
            phase,
            detail_url=record.detail_url,
            message=record.error,
        )

    def count_downloadable(self) -> CrawlInventory:
        self.store.prepare()
        if self.config.browser_backend == "managed-chrome":
            return self._count_with_managed_chrome()
        if self.config.browser_backend == "playwright":
            return self._count_with_playwright()
        if self.config.browser_backend == "playwright-cdp":
            return self._count_with_playwright_cdp()
        raise ValueError(
            f"count-only mode does not support browser backend: {self.config.browser_backend}"
        )

    def _count_with_managed_chrome(self) -> CrawlInventory:
        chrome_process, port = self._launch_managed_chrome()
        should_close_chrome = True

        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(self.config.timeout_seconds * 1000)

            try:
                return self._count_with_page(page)
            except Exception:
                if self.config.keep_browser_open_on_error:
                    should_close_chrome = False
                    print("Debug: leaving managed Chrome open after count error.")
                    self._debug_hold_playwright_context(context)
                raise
            finally:
                if should_close_chrome:
                    with suppress(Exception):
                        self._terminate_chrome_process(chrome_process)
                with suppress(Exception):
                    browser.close()

    def _count_with_playwright(self) -> CrawlInventory:
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                str(self.config.profile_dir),
                headless=self.config.headless,
                accept_downloads=False,
                viewport={"width": 1366, "height": 900},
                channel=self.config.browser_channel,
                args=["--disable-quic"],
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(self.config.timeout_seconds * 1000)

            try:
                return self._count_with_page(page)
            finally:
                context.close()

    def _count_with_playwright_cdp(self) -> CrawlInventory:
        chrome_process, port = self._launch_visible_chrome_for_cdp()
        should_close_chrome = True

        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(self.config.timeout_seconds * 1000)

            try:
                return self._count_with_page(page)
            except Exception:
                if self.config.keep_browser_open_on_error:
                    should_close_chrome = False
                    print("Debug: leaving visible CDP Chrome open after count error.")
                    self._debug_hold_playwright_context(context)
                raise
            finally:
                with suppress(Exception):
                    browser.close()
                if should_close_chrome:
                    with suppress(Exception):
                        self._terminate_chrome_process(chrome_process)

    def _count_with_page(self, page: Page) -> CrawlInventory:
        if page is not None and self.config.count_parallel_pages > 1:
            try:
                return self._count_with_page_fast(page)
            except Exception as exc:  # noqa: BLE001 - preserve correctness if fast fetch is blocked.
                self.store.log(
                    f"count-fast-fallback error={type(exc).__name__}: {exc}"
                )

        downloaded_urls = self.store.processed_detail_urls()
        listing_url: str | None = self.config.start_url if self.config.crawl_listing else None
        seen_listing_urls: set[str] = set()
        seen_detail_urls: set[str] = set()
        pages: list[InventoryPage] = []

        while listing_url:
            if listing_url in seen_listing_urls:
                self.store.log(f"count-stopped-duplicate-listing url={listing_url}")
                break
            if self._candidate_limit_reached(len(seen_detail_urls)):
                break

            seen_listing_urls.add(listing_url)
            page_index = len(pages) + 1
            html, current_url = self._load_listing_for_count(page, listing_url, page_index)
            links = self._parse_listing(html, current_url)

            page_urls: list[str] = []
            for detail_url in links.case_urls:
                if self._candidate_limit_reached(len(seen_detail_urls)):
                    break
                if detail_url in seen_detail_urls:
                    continue
                seen_detail_urls.add(detail_url)
                page_urls.append(detail_url)

            already_downloaded = sum(1 for url in page_urls if url in downloaded_urls)
            pages.append(
                InventoryPage(
                    page_index=page_index,
                    listing_url=current_url,
                    downloadable=len(page_urls),
                    already_downloaded=already_downloaded,
                    remaining=len(page_urls) - already_downloaded,
                    detail_urls=page_urls,
                )
            )
            self.store.log(
                f"count-listing page={page_index} url={current_url} "
                f"cases={len(page_urls)} next={links.next_url}"
            )

            listing_url = links.next_url

        already_downloaded = sum(1 for url in seen_detail_urls if url in downloaded_urls)
        return CrawlInventory(
            total_downloadable=len(seen_detail_urls),
            pages_scanned=len(pages),
            already_downloaded=already_downloaded,
            remaining=len(seen_detail_urls) - already_downloaded,
            pages=pages,
        )

    def _count_with_page_fast(self, page: Page) -> CrawlInventory:
        downloaded_urls = self.store.processed_detail_urls()
        listing_url: str | None = (
            self.config.start_url if self.config.crawl_listing else None
        )
        if not listing_url:
            return CrawlInventory(
                total_downloadable=0,
                pages_scanned=0,
                already_downloaded=0,
                remaining=0,
                pages=[],
            )

        self._goto_and_wait(page, listing_url)
        first_html = self._safe_page_content(page)
        first_url = page.url
        last_page_index = parse_listing_last_page_index(first_html, first_url)
        if last_page_index <= 1:
            links = self._parse_listing(first_html, first_url)
            if links.next_url:
                raise RuntimeError("fast count could not discover the last listing page")

        seen_detail_urls: set[str] = set()
        pages: list[InventoryPage] = []

        first_links = self._parse_listing(first_html, first_url)
        self._append_inventory_page(
            pages,
            page_index=1,
            listing_url=first_url,
            case_urls=first_links.case_urls,
            seen_detail_urls=seen_detail_urls,
            downloaded_urls=downloaded_urls,
        )
        self.store.log(
            f"count-listing-fast page=1 url={first_url} "
            f"cases={pages[-1].downloadable if pages else 0} last={last_page_index}"
        )

        next_page_index = 2
        while (
            next_page_index <= last_page_index
            and not self._candidate_limit_reached(len(seen_detail_urls))
        ):
            batch_end = min(
                last_page_index,
                next_page_index + self.config.count_parallel_pages - 1,
            )
            batch_urls = [
                _listing_page_url(self.config.start_url, page_index)
                for page_index in range(next_page_index, batch_end + 1)
            ]
            results = self._fetch_listing_pages_with_page(page, batch_urls)
            for offset, result in enumerate(results):
                page_index = next_page_index + offset
                if self._candidate_limit_reached(len(seen_detail_urls)):
                    break
                html = str(result["text"])
                current_url = str(result["url"])
                links = self._parse_listing(html, current_url)
                before_count = len(seen_detail_urls)
                self._append_inventory_page(
                    pages,
                    page_index=page_index,
                    listing_url=current_url,
                    case_urls=links.case_urls,
                    seen_detail_urls=seen_detail_urls,
                    downloaded_urls=downloaded_urls,
                )
                self.store.log(
                    f"count-listing-fast page={page_index} url={current_url} "
                    f"cases={len(seen_detail_urls) - before_count} last={last_page_index}"
                )
            next_page_index = batch_end + 1

        already_downloaded = sum(1 for url in seen_detail_urls if url in downloaded_urls)
        return CrawlInventory(
            total_downloadable=len(seen_detail_urls),
            pages_scanned=len(pages),
            already_downloaded=already_downloaded,
            remaining=len(seen_detail_urls) - already_downloaded,
            pages=pages,
        )

    def _append_inventory_page(
        self,
        pages: list[InventoryPage],
        *,
        page_index: int,
        listing_url: str,
        case_urls: list[str],
        seen_detail_urls: set[str],
        downloaded_urls: set[str],
    ) -> None:
        page_urls: list[str] = []
        for detail_url in case_urls:
            if self._candidate_limit_reached(len(seen_detail_urls)):
                break
            if detail_url in seen_detail_urls:
                continue
            seen_detail_urls.add(detail_url)
            page_urls.append(detail_url)

        already_downloaded = sum(1 for url in page_urls if url in downloaded_urls)
        pages.append(
            InventoryPage(
                page_index=page_index,
                listing_url=listing_url,
                downloadable=len(page_urls),
                already_downloaded=already_downloaded,
                remaining=len(page_urls) - already_downloaded,
                detail_urls=page_urls,
            )
        )

    def _load_listing_for_count(
        self, page: Page, listing_url: str, page_index: int
    ) -> tuple[str, str]:
        if page_index == 1:
            self._goto_and_wait(page, listing_url)
            return self._safe_page_content(page), page.url

        try:
            return self._fetch_html_with_page(page, listing_url), listing_url
        except Exception as exc:  # noqa: BLE001 - fall back to a normal browser navigation.
            self.store.log(
                f"count-fetch-fallback url={listing_url} error={type(exc).__name__}: {exc}"
            )
            self._goto_and_wait(page, listing_url)
            return self._safe_page_content(page), page.url

    def _run_managed_chrome(self) -> CrawlSummary:
        self.store.prepare()
        chrome_process, port = self._launch_managed_chrome()
        should_close_chrome = True

        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(self.config.timeout_seconds * 1000)
            self._configure_chrome_downloads(page)

            try:
                return self._run_with_playwright_context(context, page)
            except Exception:
                if self.config.keep_browser_open_on_error:
                    should_close_chrome = False
                    print("Debug: leaving managed Chrome open after error.")
                    self._debug_hold_playwright_context(context)
                raise
            finally:
                if should_close_chrome:
                    with suppress(Exception):
                        self._terminate_chrome_process(chrome_process)
                with suppress(Exception):
                    browser.close()

    def _run_playwright(self) -> CrawlSummary:
        self.store.prepare()
        downloaded_urls = self.store.processed_detail_urls()
        output_paths: list[Path] = []
        failed_downloads = 0
        visited_this_run: set[str] = set()
        candidate_count = 0
        listing_url = self._initial_listing_url()

        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                str(self.config.profile_dir),
                headless=self.config.headless,
                accept_downloads=True,
                viewport={"width": 1366, "height": 900},
                channel=self.config.browser_channel,
                args=["--disable-quic"],
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(self.config.timeout_seconds * 1000)

            try:
                for detail_url in self.config.detail_urls:
                    if self._target_reached(output_paths):
                        break
                    if detail_url in visited_this_run:
                        continue
                    candidate_count += 1
                    visited_this_run.add(detail_url)
                    self._report_candidate(detail_url)
                    result = self._download_case(context, page, detail_url)
                    if result.status == "downloaded" and result.output_path:
                        output_paths.append(Path(result.output_path))
                        downloaded_urls.add(detail_url)
                    elif result.status == "error":
                        failed_downloads += 1
                    self.store.append(result)
                    self._report_record(result)

                while listing_url and not self._target_reached(output_paths):
                    self._checkpoint_listing_page(listing_url)
                    self._goto_and_wait(page, listing_url)
                    listing_html = self._safe_page_content(page)
                    links = self._parse_listing(listing_html, page.url)
                    self.store.log(
                        f"listing url={page.url} cases={len(links.case_urls)} next={links.next_url}"
                    )

                    for detail_url in links.case_urls:
                        if self._target_reached(output_paths):
                            break
                        if self._candidate_limit_reached(candidate_count):
                            return self._summary(output_paths, failed_downloads)
                        if detail_url in downloaded_urls or detail_url in visited_this_run:
                            continue

                        candidate_count += 1
                        visited_this_run.add(detail_url)
                        self._report_candidate(detail_url)
                        result = self._download_case(context, page, detail_url)
                        if result.status == "downloaded" and result.output_path:
                            output_paths.append(Path(result.output_path))
                            downloaded_urls.add(detail_url)
                        elif result.status == "error":
                            failed_downloads += 1
                        self.store.append(result)
                        self._report_record(result)

                        if self.config.delay_seconds:
                            page.wait_for_timeout(int(self.config.delay_seconds * 1000))

                    if not self._target_reached(output_paths):
                        self._advance_listing_checkpoint(links.next_url)
                    listing_url = links.next_url

                return self._summary(output_paths, failed_downloads)
            except Exception:
                if self.config.keep_browser_open_on_error:
                    print("Debug: leaving Playwright Chrome open after error.")
                    self._debug_hold_playwright_context(context)
                raise
            finally:
                context.close()

    def _run_playwright_cdp(self) -> CrawlSummary:
        self.store.prepare()
        chrome_process, port = self._launch_visible_chrome_for_cdp()
        should_close_chrome = True

        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(self.config.timeout_seconds * 1000)
            self._configure_chrome_downloads(page)

            try:
                summary = self._run_with_playwright_context(context, page)
                return summary
            except Exception:
                if self.config.keep_browser_open_on_error:
                    should_close_chrome = False
                    print("Debug: leaving visible CDP Chrome open after error.")
                    self._debug_hold_playwright_context(context)
                raise
            finally:
                with suppress(Exception):
                    browser.close()
                if should_close_chrome:
                    with suppress(Exception):
                        chrome_process.terminate()

    def _run_with_playwright_context(
        self, context: BrowserContext, page: Page
    ) -> CrawlSummary:
        downloaded_urls = self.store.processed_detail_urls()
        output_paths: list[Path] = []
        failed_downloads = 0
        visited_this_run: set[str] = set()
        candidate_count = 0
        listing_url = self._initial_listing_url()

        for detail_url in self.config.detail_urls:
            if self._target_reached(output_paths):
                break
            if detail_url in visited_this_run:
                continue
            candidate_count += 1
            visited_this_run.add(detail_url)
            self._report_candidate(detail_url)
            result = self._download_case(context, page, detail_url)
            if result.status == "downloaded" and result.output_path:
                output_paths.append(Path(result.output_path))
                downloaded_urls.add(detail_url)
            elif result.status == "error":
                failed_downloads += 1
            self.store.append(result)
            self._report_record(result)

        if listing_url and self.config.browser_backend == "managed-chrome":
            if self.config.parallel_downloads > 1:
                return self._crawl_listing_by_browser_fetch(
                    page,
                    listing_url,
                    downloaded_urls,
                    visited_this_run,
                    candidate_count,
                    output_paths,
                    failed_downloads,
                )
            return self._crawl_listing_by_clicks(
                context,
                page,
                listing_url,
                downloaded_urls,
                visited_this_run,
                candidate_count,
                output_paths,
                failed_downloads,
            )

        while listing_url and not self._target_reached(output_paths):
            self._checkpoint_listing_page(listing_url)
            self._goto_and_wait(page, listing_url)
            listing_html = self._safe_page_content(page)
            links = self._parse_listing(listing_html, page.url)
            self.store.log(
                f"playwright-cdp-listing url={page.url} cases={len(links.case_urls)} next={links.next_url}"
            )

            for detail_url in links.case_urls:
                if self._target_reached(output_paths):
                    break
                if self._candidate_limit_reached(candidate_count):
                    return self._summary(output_paths, failed_downloads)
                if detail_url in downloaded_urls or detail_url in visited_this_run:
                    continue

                candidate_count += 1
                visited_this_run.add(detail_url)
                self._report_candidate(detail_url)
                result = self._download_case(context, page, detail_url)
                if result.status == "downloaded" and result.output_path:
                    output_paths.append(Path(result.output_path))
                    downloaded_urls.add(detail_url)
                elif result.status == "error":
                    failed_downloads += 1
                self.store.append(result)
                self._report_record(result)

                if self.config.delay_seconds:
                    page.wait_for_timeout(int(self.config.delay_seconds * 1000))

            if not self._target_reached(output_paths):
                self._advance_listing_checkpoint(links.next_url)
            listing_url = links.next_url

        return self._summary(output_paths, failed_downloads)

    def _crawl_listing_by_clicks(
        self,
        context: BrowserContext,
        page: Page,
        listing_url: str,
        downloaded_urls: set[str],
        visited_this_run: set[str],
        candidate_count: int,
        output_paths: list[Path],
        failed_downloads: int,
    ) -> CrawlSummary:
        while listing_url and not self._target_reached(output_paths):
            self._checkpoint_listing_page(listing_url)
            self._goto_and_wait(page, listing_url)
            current_listing_url = page.url
            listing_html = self._safe_page_content(page)
            links = self._parse_listing(listing_html, current_listing_url)
            self.store.log(
                f"managed-listing-clicks url={current_listing_url} "
                f"cases={len(links.case_urls)} next={links.next_url}"
            )

            for detail_url in links.case_urls:
                if self._target_reached(output_paths):
                    break
                if self._candidate_limit_reached(candidate_count):
                    return self._summary(output_paths, failed_downloads)
                if detail_url in downloaded_urls or detail_url in visited_this_run:
                    continue

                candidate_count += 1
                visited_this_run.add(detail_url)
                self._report_candidate(detail_url)
                result = self._download_case_by_click(context, page, detail_url)
                if result.status == "downloaded" and result.output_path:
                    output_paths.append(Path(result.output_path))
                    downloaded_urls.add(detail_url)
                elif result.status == "error":
                    failed_downloads += 1
                self.store.append(result)
                self._report_record(result)

                if self._target_reached(output_paths):
                    break

                self._return_to_listing(page, current_listing_url)
                if self.config.delay_seconds:
                    page.wait_for_timeout(int(self.config.delay_seconds * 1000))

            if not self._target_reached(output_paths):
                self._advance_listing_checkpoint(links.next_url)
            listing_url = links.next_url

        return self._summary(output_paths, failed_downloads)

    def _crawl_listing_by_browser_fetch(
        self,
        page: Page,
        listing_url: str,
        downloaded_urls: set[str],
        visited_this_run: set[str],
        candidate_count: int,
        output_paths: list[Path],
        failed_downloads: int,
    ) -> CrawlSummary:
        next_url: str | None = listing_url

        while next_url and not self._target_reached(output_paths):
            self._checkpoint_listing_page(next_url)
            if self._candidate_limit_reached(candidate_count):
                break

            try:
                self._goto_and_wait(page, next_url)
            except Exception as exc:  # noqa: BLE001 - keep completed downloads from the run.
                self.store.log(
                    f"managed-listing-fast-timeout url={next_url} "
                    f"error={type(exc).__name__}: {exc}"
                )
                break

            html = self._safe_page_content(page)
            current_url = page.url

            links = self._parse_listing(html, current_url)
            self.store.log(
                f"managed-listing-fast url={current_url} cases={len(links.case_urls)} "
                f"next={links.next_url}"
            )

            page_detail_urls: list[str] = []
            for detail_url in links.case_urls:
                if self._candidate_limit_reached(candidate_count):
                    break
                if detail_url in downloaded_urls or detail_url in visited_this_run:
                    continue
                candidate_count += 1
                visited_this_run.add(detail_url)
                page_detail_urls.append(detail_url)

            max_workers = max(1, self.config.parallel_downloads)
            index = 0
            while index < len(page_detail_urls):
                if self._target_reached(output_paths):
                    break
                if self.config.download_all:
                    batch_size = max_workers
                else:
                    remaining = self._remaining_target(output_paths)
                    batch_size = min(max_workers, remaining)
                batch = page_detail_urls[index : index + batch_size]
                index += len(batch)
                for detail_url in batch:
                    self._report_candidate(detail_url)
                results = self._fetch_detail_pdf_batch(page, batch)
                for result in results:
                    record = self._record_fast_fetch_result(result)
                    if record.status == "downloaded" and record.output_path:
                        output_paths.append(Path(record.output_path))
                        downloaded_urls.add(record.detail_url)
                    elif record.status == "error":
                        failed_downloads += 1
                    self.store.append(record)
                    self._report_record(record)

            if not self._target_reached(output_paths):
                self._advance_listing_checkpoint(links.next_url)
            next_url = links.next_url

        return self._summary(output_paths, failed_downloads)

    def _fetch_html_with_page(self, page: Page, url: str) -> str:
        _ensure_allowed_listing_url(url)
        result = self._safe_page_evaluate(
            page,
            """
            async (url) => {
                const response = await fetch(url, {
                    credentials: "include",
                    redirect: "follow",
                    headers: {"Accept": "text/html,*/*;q=0.8"}
                });
                return {
                    ok: response.ok,
                    status: response.status,
                    url: response.url,
                    text: await response.text()
                };
            }
            """,
            url,
        )
        if not result["ok"]:
            raise RuntimeError(f"listing fetch failed with HTTP {result['status']}: {url}")
        _ensure_allowed_listing_url(result["url"])
        return str(result["text"])

    def _write_live_browser_view(
        self,
        *,
        image_bytes: bytes,
        page_url: str | None,
        title: str | None,
        label: str,
    ) -> None:
        live_dir = self.config.out_dir / ".hermes"
        image_path = live_dir / "browser_view.png"
        meta_path = live_dir / "browser_view.json"
        try:
            live_dir.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(image_bytes)
            meta_path.write_text(
                json.dumps(
                    {
                        "image_path": str(image_path),
                        "page_url": page_url,
                        "title": title,
                        "label": label,
                        "captured_at": datetime.now(UTC).isoformat(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            self.store.log(f"live-browser-view-write-error label={label} error={exc}")

    def _capture_live_page_view(self, page: Page, label: str) -> None:
        try:
            self._write_live_browser_view(
                image_bytes=page.screenshot(full_page=False, timeout=2_000),
                page_url=page.url,
                title=self._safe_page_title(page, max_wait_seconds=1),
                label=label,
            )
        except Exception as exc:  # noqa: BLE001 - monitoring must not stop crawling.
            self.store.log(
                f"live-browser-view-page-error label={label} "
                f"error={type(exc).__name__}: {exc}"
            )

    def _capture_live_driver_view(self, driver, label: str) -> None:
        try:
            self._write_live_browser_view(
                image_bytes=driver.get_screenshot_as_png(),
                page_url=getattr(driver, "current_url", None),
                title=getattr(driver, "title", None),
                label=label,
            )
        except Exception as exc:  # noqa: BLE001 - monitoring must not stop crawling.
            self.store.log(
                f"live-browser-view-driver-error label={label} "
                f"error={type(exc).__name__}: {exc}"
            )

    def _fetch_listing_pages_with_page(
        self, page: Page, urls: list[str]
    ) -> list[dict[str, object]]:
        for url in urls:
            _ensure_allowed_listing_url(url)
        results = self._safe_page_evaluate(
            page,
            """
            async ({urls, concurrency, timeoutMs}) => {
                const fetchWithTimeout = async (url) => {
                    const controller = new AbortController();
                    const timeout = setTimeout(() => controller.abort(), timeoutMs);
                    try {
                        const response = await fetch(url, {
                            credentials: "include",
                            redirect: "follow",
                            headers: {"Accept": "text/html,*/*;q=0.8"},
                            signal: controller.signal
                        });
                        return {
                            ok: response.ok,
                            status: response.status,
                            url: response.url,
                            text: await response.text()
                        };
                    } catch (error) {
                        return {
                            ok: false,
                            status: 0,
                            url,
                            error: `${error.name || "Error"}: ${error.message || error}`,
                            text: ""
                        };
                    } finally {
                        clearTimeout(timeout);
                    }
                };

                const results = new Array(urls.length);
                let next = 0;
                const workerCount = Math.max(1, Math.min(concurrency, urls.length));
                const workers = Array.from({length: workerCount}, async () => {
                    while (next < urls.length) {
                        const index = next++;
                        results[index] = await fetchWithTimeout(urls[index]);
                    }
                });
                await Promise.all(workers);
                return results;
            }
            """,
            {
                "urls": urls,
                "concurrency": max(1, self.config.count_parallel_pages),
                "timeoutMs": max(1, self.config.fast_fetch_timeout_seconds) * 1000,
            },
        )
        for result in results:
            if not result["ok"]:
                raise RuntimeError(
                    f"listing fetch failed with HTTP {result['status']}: {result['url']}"
                )
            _ensure_allowed_listing_url(str(result["url"]))
        return results

    def _fetch_detail_pdf_batch(self, page: Page, detail_urls: list[str]) -> list[dict[str, object]]:
        for detail_url in detail_urls:
            _ensure_allowed_detail_url(detail_url)
        return self._safe_page_evaluate(
            page,
            """
            async ({urls, concurrency, timeoutMs}) => {
                const fetchWithTimeout = async (url, options) => {
                    const controller = new AbortController();
                    const timeout = setTimeout(() => controller.abort(), timeoutMs);
                    try {
                        return await fetch(url, {...options, signal: controller.signal});
                    } finally {
                        clearTimeout(timeout);
                    }
                };
                const toBase64 = (buffer) => {
                    let binary = "";
                    const bytes = new Uint8Array(buffer);
                    const chunkSize = 0x8000;
                    for (let i = 0; i < bytes.length; i += chunkSize) {
                        binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
                    }
                    return btoa(binary);
                };
                const parseDetail = (html, baseUrl) => {
                    const doc = new DOMParser().parseFromString(html, "text/html");
                    const titleNode = doc.querySelector("h1,h2,h3,.entry-title,.post-title,title");
                    const title = titleNode ? titleNode.textContent.trim() : "";
                    const candidates = Array.from(doc.querySelectorAll("a[href],[data-href],[data-url],[data-download],[data-file],[onclick]"));
                    const pdf = candidates.map((node) => {
                        const rawValues = ["href", "data-href", "data-url", "data-download", "data-file"]
                            .map((name) => node.getAttribute(name))
                            .filter(Boolean);
                        const onclick = node.getAttribute("onclick") || "";
                        const matches = onclick.matchAll(/(https?:\\/\\/[^\\s"'<>]+|\\/direktori\\/download_file\\/[^\\s"'<>]+)/gi);
                        for (const match of matches) rawValues.push(match[1]);
                        for (const raw of rawValues) {
                            try {
                                const href = new URL(raw, baseUrl).href;
                                const parsed = new URL(href);
                                if (
                                    parsed.protocol === "https:" &&
                                    parsed.hostname.toLowerCase() === "putusan3.mahkamahagung.go.id" &&
                                    parsed.pathname.includes("/direktori/download_file/") &&
                                    parsed.pathname.includes("/pdf/")
                                ) {
                                    return {node, href};
                                }
                            } catch (_) {
                                continue;
                            }
                        }
                        return null;
                    }).find(Boolean);
                    if (!pdf) return {title, pdfUrl: null, filename: null};
                    return {
                        title,
                        pdfUrl: pdf.href,
                        filename: pdf.node.textContent.trim() || null
                    };
                };
                const fetchOne = async (detailUrl) => {
                    try {
                        const detailResponse = await fetchWithTimeout(detailUrl, {
                            credentials: "include",
                            redirect: "follow",
                            headers: {"Accept": "text/html,*/*;q=0.8"}
                        });
                        const detailHtml = await detailResponse.text();
                        if (!detailResponse.ok) {
                            return {
                                status: "error",
                                detailUrl,
                                error: `detail HTTP ${detailResponse.status}`
                            };
                        }
                        const parsed = parseDetail(detailHtml, detailResponse.url);
                        if (!parsed.pdfUrl) {
                            return {
                                status: "skipped_no_pdf",
                                detailUrl: detailResponse.url,
                                title: parsed.title,
                                error: "no PDF download link found"
                            };
                        }
                        const pdfResponse = await fetchWithTimeout(parsed.pdfUrl, {
                            credentials: "include",
                            redirect: "follow",
                            headers: {"Accept": "application/pdf,*/*;q=0.8"}
                        });
                        const buffer = await pdfResponse.arrayBuffer();
                        if (!pdfResponse.ok) {
                            return {
                                status: "error",
                                detailUrl: detailResponse.url,
                                pdfUrl: pdfResponse.url,
                                title: parsed.title,
                                filename: parsed.filename,
                                error: `PDF HTTP ${pdfResponse.status}`
                            };
                        }
                        return {
                            status: "downloaded",
                            detailUrl: detailResponse.url,
                            pdfUrl: pdfResponse.url,
                            title: parsed.title,
                            filename: parsed.filename,
                            contentType: pdfResponse.headers.get("content-type") || "",
                            base64: toBase64(buffer)
                        };
                    } catch (error) {
                        return {
                            status: "error",
                            detailUrl,
                            error: `${error.name || "Error"}: ${error.message || error}`
                        };
                    }
                };
                const results = new Array(urls.length);
                let next = 0;
                const workers = Array.from({length: Math.max(1, concurrency)}, async () => {
                    while (next < urls.length) {
                        const index = next++;
                        results[index] = await fetchOne(urls[index]);
                    }
                });
                await Promise.all(workers);
                return results;
            }
            """,
            {
                "urls": detail_urls,
                "concurrency": max(1, self.config.parallel_downloads),
                "timeoutMs": max(1, self.config.fast_fetch_timeout_seconds) * 1000,
            },
        )

    def _record_fast_fetch_result(self, result: dict[str, object]) -> CrawlRecord:
        detail_url = str(result.get("detailUrl") or "")
        if result.get("status") != "downloaded":
            return CrawlRecord(
                status=str(result.get("status") or "error"),
                detail_url=detail_url,
                title=str(result.get("title") or "") or None,
                error=str(result.get("error") or "unknown fast fetch failure"),
            )

        pdf_url = str(result.get("pdfUrl") or "")
        try:
            _ensure_allowed_detail_url(detail_url)
            _ensure_allowed_pdf_url(pdf_url)
            fallback_stem = self._fallback_stem(detail_url)
            filename = sanitize_filename(
                str(result.get("filename") or "") or None,
                fallback_stem,
            )
            output_path = unique_path(self.pdf_dir / filename)
            output_path.write_bytes(base64.b64decode(str(result["base64"])))
            verify_pdf(output_path)
            return CrawlRecord(
                status="downloaded",
                detail_url=detail_url,
                pdf_url=pdf_url,
                output_path=str(output_path),
                title=str(result.get("title") or "") or None,
                filename=output_path.name,
            )
        except Exception as exc:  # noqa: BLE001 - one bad candidate should not stop traversal.
            return CrawlRecord(
                status="error",
                detail_url=detail_url,
                pdf_url=pdf_url or None,
                title=str(result.get("title") or "") or None,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _return_to_listing(self, page: Page, listing_url: str) -> None:
        if page.url == listing_url:
            return
        try:
            page.go_back(wait_until="domcontentloaded", timeout=self.config.timeout_seconds * 1000)
            self._wait_for_accessible_page(page)
        except PlaywrightError:
            self._goto_and_wait(page, listing_url)

    def _run_undetected_chrome(self) -> CrawlSummary:
        if self.config.parallel_downloads > 1:
            return self._run_parallel_undetected()

        self.store.prepare()
        downloaded_urls = self.store.processed_detail_urls()
        output_paths: list[Path] = []
        failed_downloads = 0
        visited_this_run: set[str] = set()
        candidate_count = 0
        listing_url = self._initial_listing_url()

        driver = self._launch_undetected_chrome()
        should_close_driver = True
        try:
            for detail_url in self.config.detail_urls:
                if self._target_reached(output_paths):
                    break
                if detail_url in visited_this_run:
                    continue
                candidate_count += 1
                visited_this_run.add(detail_url)
                self._report_candidate(detail_url)
                result = self._download_case_undetected(driver, detail_url)
                if result.status == "downloaded" and result.output_path:
                    output_paths.append(Path(result.output_path))
                    downloaded_urls.add(detail_url)
                elif result.status == "error":
                    failed_downloads += 1
                self.store.append(result)
                self._report_record(result)

            while listing_url and not self._target_reached(output_paths):
                self._checkpoint_listing_page(listing_url)
                self._uc_goto_and_wait(driver, listing_url)
                listing_html = driver.page_source
                links = self._parse_listing(listing_html, driver.current_url)
                self.store.log(
                    f"undetected-listing url={driver.current_url} "
                    f"cases={len(links.case_urls)} next={links.next_url}"
                )

                for detail_url in links.case_urls:
                    if self._target_reached(output_paths):
                        break
                    if self._candidate_limit_reached(candidate_count):
                        return self._summary(output_paths, failed_downloads)
                    if detail_url in downloaded_urls or detail_url in visited_this_run:
                        continue

                    candidate_count += 1
                    visited_this_run.add(detail_url)
                    self._report_candidate(detail_url)
                    result = self._download_case_undetected(driver, detail_url)
                    if result.status == "downloaded" and result.output_path:
                        output_paths.append(Path(result.output_path))
                        downloaded_urls.add(detail_url)
                    elif result.status == "error":
                        failed_downloads += 1
                    self.store.append(result)
                    self._report_record(result)

                    if self.config.delay_seconds:
                        time.sleep(self.config.delay_seconds)

                if not self._target_reached(output_paths):
                    self._advance_listing_checkpoint(links.next_url)
                listing_url = links.next_url

            return self._summary(output_paths, failed_downloads)
        except Exception:
            if self.config.keep_browser_open_on_error:
                should_close_driver = False
                print("Debug: leaving Chrome open after error.")
                self._debug_hold_driver(driver)
            raise
        finally:
            if should_close_driver:
                with suppress(OSError):
                    driver.quit()

    def _run_parallel_undetected(self) -> CrawlSummary:
        started_at = time.perf_counter()
        self.store.prepare()
        downloaded_urls = self.store.processed_detail_urls()
        detail_urls = list(self.config.detail_urls)
        explicit_count = len(detail_urls)

        driver = self._launch_undetected_chrome()
        session_state: dict[str, object] | None = None
        should_close_driver = True
        try:
            if self.config.crawl_listing:
                collected_urls, session_state = self._collect_listing_detail_urls(
                    driver,
                    self.config.start_url,
                    (
                        None
                        if self.config.max_candidates is None
                        else self.config.max_candidates + len(downloaded_urls)
                    ),
                )
                detail_urls.extend(collected_urls)
            if session_state is None:
                session_state = self._requests_state_from_driver(driver)
        except Exception:
            if self.config.keep_browser_open_on_error:
                should_close_driver = False
                print("Debug: leaving Chrome open after error.")
                self._debug_hold_driver(driver)
            raise
        finally:
            if should_close_driver:
                with suppress(OSError):
                    driver.quit()

        queue: list[str] = []
        seen: set[str] = set()
        for index, detail_url in enumerate(detail_urls):
            if detail_url in seen:
                continue
            if index >= explicit_count and detail_url in downloaded_urls:
                continue
            seen.add(detail_url)
            queue.append(detail_url)
            if (
                self.config.max_candidates is not None
                and len(queue) >= self.config.max_candidates
            ):
                break

        output_paths: list[Path] = []
        failed_downloads = 0
        total_bytes = 0
        path_lock = Lock()
        max_workers = max(1, self.config.parallel_downloads)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            queue_index = 0
            while queue_index < len(queue) and not self._target_reached(output_paths):
                if self.config.download_all:
                    batch_size = max_workers
                else:
                    remaining = self._remaining_target(output_paths)
                    batch_size = min(max_workers, remaining)
                batch = queue[queue_index : queue_index + batch_size]
                queue_index += len(batch)
                futures = []
                for detail_url in batch:
                    self._report_candidate(detail_url)
                    futures.append(
                        executor.submit(
                            self._download_case_parallel,
                            session_state,
                            detail_url,
                            path_lock,
                        )
                    )
                for future in as_completed(futures):
                    result = future.result()
                    self.store.append(result.record)
                    self._report_record(result.record)
                    if result.record.status == "downloaded" and result.record.output_path:
                        output_paths.append(Path(result.record.output_path))
                        total_bytes += result.bytes_downloaded
                        downloaded_urls.add(result.record.detail_url)
                    elif result.record.status == "error":
                        failed_downloads += 1

        elapsed = time.perf_counter() - started_at
        total_downloaded = self._target_completed_before + len(output_paths)
        summary = CrawlSummary(
            downloaded=total_downloaded,
            target=self._summary_target(),
            failed_downloads=failed_downloads,
            output_paths=output_paths,
            elapsed_seconds=elapsed,
            metrics={
                "candidates_available": len(queue),
                "attempted": self._progress_attempted,
                "skipped": self._progress_skipped,
                "resumed_downloads": self._target_completed_before,
                "downloaded_this_run": len(output_paths),
                "parallel_downloads": max_workers,
                "bytes_downloaded": total_bytes,
                "downloads_per_second": len(output_paths) / elapsed if elapsed else 0,
                "success_rate_percent": (
                    round(
                        100 * len(output_paths) / self._progress_attempted,
                        1,
                    )
                    if self._progress_attempted
                    else 0
                ),
            },
        )
        self._complete_target_if_satisfied(total_downloaded)
        self._report_progress("complete", message="Crawl finished")
        return summary

    def _collect_listing_detail_urls(
        self, driver, start_url: str, target_count: int | None
    ) -> tuple[list[str], dict[str, object]]:
        collected: list[str] = []
        seen: set[str] = set()
        session_state: dict[str, object] | None = None
        page_index = 1
        max_pages = (
            max(1, self.config.max_candidates // 20 + 2)
            if self.config.max_candidates is not None
            else None
        )
        next_url: str | None = start_url

        while (
            next_url
            and (target_count is None or len(collected) < target_count)
            and (max_pages is None or page_index <= max_pages)
        ):
            self._uc_goto_and_wait(driver, next_url)
            session_state = self._requests_state_from_driver(driver)
            links = self._parse_listing(driver.page_source, driver.current_url)
            self.store.log(
                f"parallel-listing page={page_index} url={driver.current_url} "
                f"cases={len(links.case_urls)}"
            )
            for detail_url in links.case_urls:
                if detail_url in seen:
                    continue
                seen.add(detail_url)
                collected.append(detail_url)
                if target_count is not None and len(collected) >= target_count:
                    break
            next_url = links.next_url
            page_index += 1

        if session_state is None:
            session_state = self._requests_state_from_driver(driver)
        return collected, session_state

    def _requests_state_from_driver(self, driver) -> dict[str, object]:
        return {
            "user_agent": driver.execute_script("return navigator.userAgent"),
            "cookies": [
                {
                    "name": cookie["name"],
                    "value": cookie["value"],
                    "domain": cookie.get("domain"),
                    "path": cookie.get("path", "/"),
                }
                for cookie in driver.get_cookies()
            ],
        }

    def _download_case_parallel(
        self, session_state: dict[str, object], detail_url: str, path_lock: Lock
    ) -> BulkDownloadResult:
        import requests

        started_at = time.perf_counter()
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": str(session_state["user_agent"]),
                "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        )
        for cookie in session_state["cookies"]:  # type: ignore[union-attr]
            session.cookies.set(
                cookie["name"],
                cookie["value"],
                domain=cookie.get("domain"),
                path=cookie.get("path", "/"),
            )

        try:
            _ensure_allowed_detail_url(detail_url)
            detail_response = session.get(detail_url, timeout=self.config.timeout_seconds)
            detail_response.raise_for_status()
            html = detail_response.text
            if _http_challenge_page(html):
                raise ChallengeBlockedError("detail request returned challenge HTML")
            title = parse_title(html)
            pdf_link = parse_pdf_link(html, detail_response.url)
            if not pdf_link:
                record = CrawlRecord(
                    status="skipped_no_pdf",
                    detail_url=detail_url,
                    title=title,
                    error="no PDF download link found",
                )
                return BulkDownloadResult(record, time.perf_counter() - started_at)
            _ensure_allowed_pdf_url(pdf_link.url)

            pdf_response = session.get(
                pdf_link.url,
                headers={"Referer": detail_response.url, "Accept": "application/pdf,*/*;q=0.8"},
                timeout=self.config.timeout_seconds,
            )
            pdf_response.raise_for_status()
            if not pdf_response.content.startswith(b"%PDF-"):
                raise RuntimeError(
                    f"PDF request returned {pdf_response.headers.get('content-type', 'unknown')}"
                )

            fallback_stem = self._fallback_stem(detail_url)
            filename = sanitize_filename(pdf_link.filename, fallback_stem)
            with path_lock:
                output_path = unique_path(self.pdf_dir / filename)
                output_path.write_bytes(pdf_response.content)
            verify_pdf(output_path)
            record = CrawlRecord(
                status="downloaded",
                detail_url=detail_url,
                pdf_url=pdf_link.url,
                output_path=str(output_path),
                title=title,
                filename=output_path.name,
            )
            return BulkDownloadResult(
                record,
                time.perf_counter() - started_at,
                bytes_downloaded=len(pdf_response.content),
            )
        except Exception as exc:  # noqa: BLE001 - batch runs should report per-case failures.
            record = CrawlRecord(
                status="error",
                detail_url=detail_url,
                error=f"{type(exc).__name__}: {exc}",
            )
            return BulkDownloadResult(record, time.perf_counter() - started_at)

    def _launch_undetected_chrome(self):
        import undetected_chromedriver as uc

        options = uc.ChromeOptions()
        options.add_argument("--window-size=1366,900")
        profile_dir = (self.config.profile_dir / "undetected-chrome").resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)
        options.add_argument(f"--user-data-dir={profile_dir}")
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        options.add_experimental_option(
            "prefs",
            {
                "download.default_directory": str(self.pdf_dir.resolve()),
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "plugins.always_open_pdf_externally": True,
            },
        )
        if self.config.headless:
            options.add_argument("--headless=new")

        version_main = self.config.chrome_version_main or _detect_chrome_version_main()
        driver = uc.Chrome(
            options=options,
            use_subprocess=True,
            version_main=version_main,
        )
        driver.set_page_load_timeout(self.config.timeout_seconds)
        return driver

    def _launch_managed_chrome(self) -> tuple[subprocess.Popen, int]:
        chrome_path = _find_chrome_executable()
        if chrome_path is None:
            raise RuntimeError("Could not find chrome.exe")

        user_data_dir = self._prepare_managed_chrome_profile()
        port = self.config.cdp_port or _find_free_port()
        args = [
            str(chrome_path),
            f"--remote-debugging-port={port}",
            f"--user-data-dir={user_data_dir}",
            f"--profile-directory={self.config.chrome_profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-quic",
            "--new-window",
            "about:blank",
        ]
        if self.config.headless:
            args.insert(-1, "--headless=new")

        process = subprocess.Popen(args)
        try:
            self._wait_for_cdp(port)
        except Exception:
            with suppress(Exception):
                self._terminate_chrome_process(process)
            raise
        return process, port

    def _prepare_managed_chrome_profile(self) -> Path:
        source_root = self.config.chrome_user_data_dir or _default_chrome_user_data_dir()
        source_profile = source_root / self.config.chrome_profile
        if not source_profile.exists():
            raise RuntimeError(
                f"Chrome profile does not exist: {source_profile}. "
                "Pass --chrome-profile with a valid profile directory name."
            )

        target_root = (self.config.profile_dir / "managed-chrome").resolve()
        target_profile = target_root / self.config.chrome_profile
        if target_profile.exists() and not self.config.refresh_profile_snapshot:
            return target_root

        if target_root.exists():
            try:
                shutil.rmtree(target_root)
            except OSError as exc:
                raise RuntimeError(
                    f"Could not refresh managed Chrome profile at {target_root}. "
                    "Close any Chrome window opened by this crawler, then rerun."
                ) from exc

        target_root.mkdir(parents=True, exist_ok=True)
        for filename in (
            "First Run",
            "FirstLaunchAfterInstallation",
            "Last Browser",
            "Last Version",
            "Local State",
            "Variations",
        ):
            source = source_root / filename
            if source.exists() and source.is_file():
                shutil.copy2(source, target_root / filename)

        try:
            shutil.copytree(source_profile, target_profile, ignore=_chrome_profile_ignore)
        except OSError as exc:
            raise RuntimeError(
                f"Could not copy Chrome profile {source_profile}. "
                "Close Chrome once so profile files are not locked, then rerun."
            ) from exc

        self._write_download_preferences(target_profile)

        return target_root

    def _write_download_preferences(self, target_profile: Path) -> None:
        preferences_path = target_profile / "Preferences"
        preferences: dict[str, object] = {}
        if preferences_path.exists():
            try:
                preferences = json.loads(preferences_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                preferences = {}

        download = preferences.setdefault("download", {})
        if isinstance(download, dict):
            download["default_directory"] = str(self.pdf_dir.resolve())
            download["directory_upgrade"] = True
            download["prompt_for_download"] = False

        plugins = preferences.setdefault("plugins", {})
        if isinstance(plugins, dict):
            plugins["always_open_pdf_externally"] = True

        preferences_path.write_text(json.dumps(preferences), encoding="utf-8")

    def _configure_chrome_downloads(self, page: Page) -> None:
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        with suppress(Exception):
            session = page.context.new_cdp_session(page)
            session.send(
                "Page.setDownloadBehavior",
                {
                    "behavior": "allow",
                    "downloadPath": str(self.pdf_dir.resolve()),
                },
            )

    @staticmethod
    def _terminate_chrome_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _launch_visible_chrome_for_cdp(self) -> tuple[subprocess.Popen, int]:
        chrome_path = _find_chrome_executable()
        if chrome_path is None:
            raise RuntimeError("Could not find chrome.exe")

        port = self.config.cdp_port or _find_free_port()
        profile_dir = (self.config.profile_dir / "playwright-cdp").resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)
        args = [
            str(chrome_path),
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-quic",
            "--new-window",
            "about:blank",
        ]
        if self.config.headless:
            args.insert(-1, "--headless=new")

        process = subprocess.Popen(args)
        try:
            self._wait_for_cdp(port)
        except Exception:
            with suppress(Exception):
                self._terminate_chrome_process(process)
            raise
        return process, port

    def _wait_for_cdp(self, port: int) -> None:
        deadline = time.monotonic() + min(self.config.timeout_seconds, 30)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2) as response:
                    if response.status == 200:
                        return
            except Exception as exc:  # noqa: BLE001 - retry until Chrome exposes CDP.
                last_error = exc
            time.sleep(0.25)
        raise RuntimeError(f"Chrome CDP did not become ready on port {port}: {last_error}")

    def _download_case_undetected(self, driver, detail_url: str) -> CrawlRecord:
        last_error: Exception | None = None
        _ensure_allowed_detail_url(detail_url)
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                self._uc_goto_and_wait(driver, detail_url)
                html = driver.page_source
                title = parse_title(html)
                pdf_link = parse_pdf_link(html, driver.current_url)
                if not pdf_link:
                    return CrawlRecord(
                        status="skipped_no_pdf",
                        detail_url=detail_url,
                        title=title,
                        error="no PDF download link found",
                    )
                _ensure_allowed_pdf_url(pdf_link.url)

                fallback_stem = self._fallback_stem(detail_url)
                filename = sanitize_filename(pdf_link.filename, fallback_stem)
                output_path = unique_path(self.pdf_dir / filename)
                self._save_pdf_undetected(driver, pdf_link.url, driver.current_url, output_path)
                verify_pdf(output_path)
                return CrawlRecord(
                    status="downloaded",
                    detail_url=detail_url,
                    pdf_url=pdf_link.url,
                    output_path=str(output_path),
                    title=title,
                    filename=output_path.name,
                )
            except Exception as exc:  # noqa: BLE001 - we need to keep crawling.
                last_error = exc
                self.store.log(
                    f"undetected-attempt={attempt} detail_url={detail_url} "
                    f"error={type(exc).__name__}: {exc}"
                )
                time.sleep(min(2 * attempt, 10))

        return CrawlRecord(
            status="error",
            detail_url=detail_url,
            error=f"{type(last_error).__name__}: {last_error}",
        )

    def _save_pdf_undetected(
        self, driver, pdf_url: str, referer_url: str, output_path: Path
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        before = {path.resolve() for path in self.pdf_dir.glob("*")}
        clicked = driver.execute_script(
            """
            const url = arguments[0];
            const links = Array.from(document.querySelectorAll("a[href]"));
            const link = links.find((node) => node.href === url || node.getAttribute("href") === url);
            if (!link) return false;
            link.scrollIntoView({block: "center", inline: "center"});
            link.click();
            return true;
            """,
            pdf_url,
        )
        if not clicked:
            raise RuntimeError(f"PDF link is not present on the current detail page: {pdf_url}")

        downloaded_path = self._wait_for_chrome_download(before)
        if downloaded_path.resolve() != output_path.resolve():
            if output_path.exists():
                output_path = unique_path(output_path)
            downloaded_path.replace(output_path)

    def _wait_for_chrome_download(self, before: set[Path]) -> Path:
        deadline = time.monotonic() + self.config.timeout_seconds
        last_candidate: Path | None = None

        while time.monotonic() < deadline:
            active_downloads = list(self.pdf_dir.glob("*.crdownload"))
            candidates = [
                path
                for path in self.pdf_dir.glob("*.pdf")
                if path.resolve() not in before and path.stat().st_size > 0
            ]
            if candidates and not active_downloads:
                newest = max(candidates, key=lambda path: path.stat().st_mtime)
                with suppress(ValueError):
                    verify_pdf(newest)
                    return newest
                last_candidate = newest
            time.sleep(0.5)

        if last_candidate:
            verify_pdf(last_candidate)
            return last_candidate
        raise RuntimeError("Chrome did not finish a PDF download before timeout")

    def _uc_goto_and_wait(self, driver, url: str) -> None:
        driver.get(url)
        self._capture_live_driver_view(driver, "driver-goto")
        deadline = time.monotonic() + self.config.timeout_seconds
        passive_printed = False
        interactive_printed = False
        last_status_at = 0.0
        last_capture_at = 0.0

        while time.monotonic() < deadline:
            now = time.monotonic()
            if now - last_capture_at >= 5:
                self._capture_live_driver_view(driver, "driver-wait")
                last_capture_at = now
            network_error = self._uc_network_error_visible(driver)
            if network_error:
                raise RuntimeError(network_error)
            if not self._uc_challenge_visible(driver):
                self._capture_live_driver_view(driver, "driver-accessible")
                return
            if self._uc_interactive_challenge_visible(driver):
                if not interactive_printed or now - last_status_at >= 10:
                    print(
                        "Action required: Cloudflare human verification is visible. "
                        "Click the checkbox in the opened Chrome window; the crawler "
                        "will continue automatically after the page loads."
                    )
                    interactive_printed = True
                    last_status_at = now
            elif not passive_printed or now - last_status_at >= 10:
                print("Cloudflare challenge is visible; waiting for browser clearance.")
                passive_printed = True
                last_status_at = now
            time.sleep(2)

        raise ChallengeBlockedError(
            f"Cloudflare challenge did not clear within {self.config.timeout_seconds}s"
        )

    def _debug_hold_driver(self, driver) -> None:
        if self.config.debug_hold_seconds <= 0:
            return
        deadline = time.monotonic() + self.config.debug_hold_seconds
        print(f"Debug: holding Chrome open for {self.config.debug_hold_seconds}s.")
        while time.monotonic() < deadline:
            try:
                if not driver.window_handles:
                    return
            except Exception:
                return
            time.sleep(1)

    def _debug_hold_playwright_context(self, context: BrowserContext) -> None:
        if self.config.debug_hold_seconds <= 0:
            return
        deadline = time.monotonic() + self.config.debug_hold_seconds
        print(f"Debug: holding Playwright Chrome open for {self.config.debug_hold_seconds}s.")
        while time.monotonic() < deadline:
            if not context.pages:
                return
            time.sleep(1)

    @staticmethod
    def _uc_challenge_visible(driver) -> bool:
        title = (driver.title or "").lower()
        if "just a moment" in title:
            return True
        visible_text = (
            driver.execute_script("return document.body ? document.body.innerText : ''") or ""
        ).lower()
        challenge_markers = (
            "checking if the site connection is secure",
            "verify you are human",
            "needs to review the security of your connection",
        )
        return any(marker in visible_text for marker in challenge_markers)

    @staticmethod
    def _uc_interactive_challenge_visible(driver) -> bool:
        title = (driver.title or "").lower()
        visible_text = (
            driver.execute_script("return document.body ? document.body.innerText : ''") or ""
        ).lower()
        return (
            "just a moment" in title
            and "verify you are human" in visible_text
            and "performing security verification" in visible_text
        )

    @staticmethod
    def _uc_network_error_visible(driver) -> str | None:
        title = (driver.title or "").lower()
        if not title.startswith("putusan3.mahkamahagung.go.id"):
            return None
        visible_text = (
            driver.execute_script("return document.body ? document.body.innerText : ''") or ""
        ).lower()
        if "this site can" in visible_text and "be reached" in visible_text:
            return "Chrome could not reach putusan3.mahkamahagung.go.id"
        return None

    def _download_case(
        self, context: BrowserContext, page: Page, detail_url: str
    ) -> CrawlRecord:
        last_error: Exception | None = None
        _ensure_allowed_detail_url(detail_url)
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                self._goto_and_wait(page, detail_url)
                return self._download_current_detail(context, page, detail_url)
            except Exception as exc:  # noqa: BLE001 - we need to keep crawling.
                last_error = exc
                self.store.log(
                    f"attempt={attempt} detail_url={detail_url} error={type(exc).__name__}: {exc}"
                )
                time.sleep(min(2 * attempt, 10))

        return CrawlRecord(
            status="error",
            detail_url=detail_url,
            error=f"{type(last_error).__name__}: {last_error}",
        )

    def _download_case_by_click(
        self, context: BrowserContext, page: Page, detail_url: str
    ) -> CrawlRecord:
        last_error: Exception | None = None
        listing_url = page.url
        _ensure_allowed_detail_url(detail_url)
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                self._click_detail_link(page, detail_url)
                return self._download_current_detail(context, page, detail_url)
            except Exception as exc:  # noqa: BLE001 - keep crawling after bad cases.
                last_error = exc
                self.store.log(
                    f"click-attempt={attempt} detail_url={detail_url} "
                    f"error={type(exc).__name__}: {exc}"
                )
                if attempt < self.config.retry_attempts:
                    self._return_to_listing(page, listing_url)
                    time.sleep(self._retry_delay_seconds(exc, attempt))

        return CrawlRecord(
            status="error",
            detail_url=detail_url,
            error=f"{type(last_error).__name__}: {last_error}",
        )

    def _click_detail_link(self, page: Page, detail_url: str) -> None:
        clicked = page.evaluate(
            """
            (url) => {
                const links = Array.from(document.querySelectorAll("a[href]"));
                const link = links.find((node) => node.href === url);
                if (!link) return false;
                link.scrollIntoView({block: "center", inline: "center"});
                link.click();
                return true;
            }
            """,
            detail_url,
        )
        if not clicked:
            raise RuntimeError(f"detail link is not present on the listing page: {detail_url}")
        page.wait_for_url("**/direktori/putusan/*.html", wait_until="domcontentloaded")
        self._wait_for_accessible_page(page)

    def _download_current_detail(
        self, context: BrowserContext, page: Page, expected_detail_url: str
    ) -> CrawlRecord:
        actual_detail_url = page.url if _is_allowed_detail_url(page.url) else expected_detail_url
        html = self._safe_page_content(page)
        title = parse_title(html)
        pdf_link = parse_pdf_link(html, page.url)
        if not pdf_link:
            return CrawlRecord(
                status="skipped_no_pdf",
                detail_url=actual_detail_url,
                title=title,
                error="no PDF download link found",
            )
        _ensure_allowed_pdf_url(pdf_link.url)

        fallback_stem = self._fallback_stem(actual_detail_url)
        filename = sanitize_filename(pdf_link.filename, fallback_stem)
        output_path = unique_path(self.pdf_dir / filename)
        self._save_pdf(context, page, pdf_link.url, output_path)
        verify_pdf(output_path)
        return CrawlRecord(
            status="downloaded",
            detail_url=actual_detail_url,
            pdf_url=pdf_link.url,
            output_path=str(output_path),
            title=title,
            filename=output_path.name,
        )

    def _save_pdf(
        self, context: BrowserContext, page: Page, pdf_url: str, output_path: Path
    ) -> None:
        _ensure_allowed_pdf_url(pdf_url)
        if self.config.browser_backend == "managed-chrome":
            self._save_pdf_with_page_fetch(page, pdf_url, output_path)
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        link = page.locator(f'a[href="{pdf_url}"]').first

        try:
            if link.count() > 0:
                with page.expect_download(timeout=20_000) as download_info:
                    link.click()
                download = download_info.value
                download.save_as(str(output_path))
                return
        except (PlaywrightError, PlaywrightTimeoutError) as exc:
            self.store.log(f"browser-download-fallback url={pdf_url} error={exc}")

        response = context.request.get(pdf_url, timeout=self.config.timeout_seconds * 1000)
        if not response.ok:
            raise RuntimeError(f"PDF request failed with HTTP {response.status}")
        body = response.body()
        output_path.write_bytes(body)

    def _save_pdf_with_page_fetch(self, page: Page, pdf_url: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = self._safe_page_evaluate(
            page,
            """
            async (url) => {
                const response = await fetch(url, {
                    credentials: "include",
                    redirect: "follow",
                    headers: {
                        "Accept": "application/pdf,*/*;q=0.8"
                    }
                });
                const buffer = await response.arrayBuffer();
                let binary = "";
                const bytes = new Uint8Array(buffer);
                const chunkSize = 0x8000;
                for (let i = 0; i < bytes.length; i += chunkSize) {
                    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
                }
                return {
                    ok: response.ok,
                    status: response.status,
                    url: response.url,
                    contentType: response.headers.get("content-type") || "",
                    retryAfter: response.headers.get("retry-after") || "",
                    base64: btoa(binary)
                };
            }
            """,
            pdf_url,
        )
        if not result["ok"]:
            if result["status"] == 429:
                retry_after = _parse_retry_after_seconds(result.get("retryAfter"))
                raise RateLimitedError(
                    "PDF fetch failed with HTTP 429",
                    retry_after_seconds=retry_after,
                )
            raise RuntimeError(f"PDF fetch failed with HTTP {result['status']}")
        _ensure_allowed_pdf_url(result["url"])
        body = base64.b64decode(result["base64"])
        output_path.write_bytes(body)

    def _retry_delay_seconds(self, error: Exception, attempt: int) -> float:
        if isinstance(error, RateLimitedError):
            server_delay = error.retry_after_seconds or 0.0
            configured_delay = self.config.rate_limit_backoff_seconds * attempt
            return max(server_delay, configured_delay)
        return min(2 * attempt, 10)

    def _save_pdf_with_chrome_click(self, page: Page, pdf_url: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        before = {path.resolve() for path in self.pdf_dir.glob("*")}
        clicked = page.evaluate(
            """
            (url) => {
                const links = Array.from(document.querySelectorAll("a[href]"));
                const link = links.find((node) => node.href === url || node.getAttribute("href") === url);
                if (!link) return false;
                link.scrollIntoView({block: "center", inline: "center"});
                link.click();
                return true;
            }
            """,
            pdf_url,
        )
        if not clicked:
            raise RuntimeError(f"PDF link is not present on the current detail page: {pdf_url}")

        downloaded_path = self._wait_for_chrome_download(before)
        if downloaded_path.resolve() != output_path.resolve():
            if output_path.exists():
                output_path = unique_path(output_path)
            downloaded_path.replace(output_path)

    def _goto_and_wait(self, page: Page, url: str) -> None:
        last_error: PlaywrightError | None = None
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=self.config.timeout_seconds * 1000,
                )
                if str(page.url).startswith("chrome-error://"):
                    raise PlaywrightError(
                        f"Chrome opened an internal error page while navigating to {url}"
                    )
                self._capture_live_page_view(page, "page-goto")
                self._wait_for_accessible_page(page)
                return
            except (PlaywrightError, PlaywrightTimeoutError) as exc:
                if not _is_retryable_navigation_error(exc, str(page.url)):
                    raise
                last_error = exc
                self.store.log(
                    f"navigation-retry attempt={attempt} url={url} "
                    f"page_url={page.url} error={type(exc).__name__}: {exc}"
                )
                if attempt >= self.config.retry_attempts:
                    break
                with suppress(PlaywrightError, PlaywrightTimeoutError):
                    page.goto("about:blank", wait_until="commit", timeout=5_000)
                with suppress(PlaywrightError):
                    page.wait_for_timeout(min(1_000 * attempt, 3_000))

        assert last_error is not None
        raise last_error

    def _wait_for_accessible_page(self, page: Page) -> None:
        timeout_seconds = (
            self.config.manual_clearance_timeout_seconds
            if self.config.browser_backend == "managed-chrome"
            else self.config.timeout_seconds
        )
        deadline = time.monotonic() + timeout_seconds
        printed = False
        challenge_seen = False
        last_capture_at = 0.0

        while time.monotonic() < deadline:
            now = time.monotonic()
            if now - last_capture_at >= 5:
                self._capture_live_page_view(page, "page-wait")
                last_capture_at = now
            try:
                html = self._safe_page_content(page, max_wait_seconds=5)
                title = self._safe_page_title(page, max_wait_seconds=2)
            except (PlaywrightError, PlaywrightTimeoutError) as exc:
                if not _is_transient_page_read_error(exc):
                    raise
                self.store.log(
                    "challenge-navigation-wait "
                    f"url={page.url} error={type(exc).__name__}: {exc}"
                )
                page.wait_for_timeout(1_000)
                continue
            if parse_pdf_link(html, page.url) or self._parse_listing(html, page.url).case_urls:
                self._capture_live_page_view(page, "page-accessible")
                self._cool_down_after_challenge(page, challenge_seen)
                return
            if not looks_like_challenge(html, title):
                self._capture_live_page_view(page, "page-accessible")
                self._cool_down_after_challenge(page, challenge_seen)
                return
            challenge_seen = True
            if not printed:
                if self.config.browser_backend == "managed-chrome":
                    print(
                        "Action required: Cloudflare verification is visible in Chrome. "
                        "Complete it in the opened Chrome window; the crawler will continue "
                        "automatically after the page loads."
                    )
                elif self.config.headless:
                    print("Cloudflare challenge is visible; waiting for normal browser clearance.")
                else:
                    print("Cloudflare challenge is visible; waiting for the page to become accessible.")
                printed = True
            page.wait_for_timeout(2_000)

        raise ChallengeBlockedError(
            f"Cloudflare challenge did not clear within {timeout_seconds}s"
        )

    def _cool_down_after_challenge(self, page: Page, challenge_seen: bool) -> None:
        if not challenge_seen or self.config.challenge_cooldown_seconds <= 0:
            return
        cooldown_ms = int(self.config.challenge_cooldown_seconds * 1000)
        self.store.log(
            f"challenge-cleared cooldown_seconds={self.config.challenge_cooldown_seconds:g} "
            f"url={page.url}"
        )
        page.wait_for_timeout(cooldown_ms)

    def _safe_page_evaluate(
        self,
        page: Page,
        expression: str,
        arg: object | None = None,
        *,
        max_wait_seconds: float | None = None,
    ) -> object:
        wait_seconds = max_wait_seconds or self.config.timeout_seconds
        deadline = time.monotonic() + wait_seconds
        last_error: PlaywrightError | None = None

        while True:
            try:
                if arg is None:
                    return page.evaluate(expression)
                return page.evaluate(expression, arg)
            except PlaywrightError as exc:
                if not _is_transient_page_read_error(exc):
                    raise
                last_error = exc
                if time.monotonic() >= deadline:
                    raise
                with suppress(PlaywrightError, PlaywrightTimeoutError):
                    page.wait_for_load_state("domcontentloaded", timeout=1_000)
                with suppress(PlaywrightError):
                    page.wait_for_timeout(500)

            if time.monotonic() >= deadline and last_error is not None:
                raise last_error

    def _safe_page_content(self, page: Page, max_wait_seconds: float | None = None) -> str:
        wait_seconds = max_wait_seconds or self.config.timeout_seconds
        deadline = time.monotonic() + wait_seconds
        last_error: PlaywrightError | None = None

        while True:
            try:
                return page.content()
            except PlaywrightError as exc:
                if not _is_transient_page_read_error(exc):
                    raise
                last_error = exc
                if time.monotonic() >= deadline:
                    raise
                with suppress(PlaywrightError, PlaywrightTimeoutError):
                    page.wait_for_load_state("domcontentloaded", timeout=1_000)
                with suppress(PlaywrightError):
                    page.wait_for_timeout(500)

            if time.monotonic() >= deadline and last_error is not None:
                raise last_error

    def _safe_page_title(self, page: Page, max_wait_seconds: float = 2) -> str:
        deadline = time.monotonic() + max_wait_seconds
        last_error: PlaywrightError | None = None

        while True:
            try:
                return page.title()
            except PlaywrightError as exc:
                if not _is_transient_page_read_error(exc):
                    raise
                last_error = exc
                if time.monotonic() >= deadline:
                    return ""
                with suppress(PlaywrightError, PlaywrightTimeoutError):
                    page.wait_for_load_state("domcontentloaded", timeout=500)
                with suppress(PlaywrightError):
                    page.wait_for_timeout(250)

            if time.monotonic() >= deadline and last_error is not None:
                return ""

    def _candidate_limit_reached(self, candidate_count: int) -> bool:
        return (
            self.config.max_candidates is not None
            and candidate_count >= self.config.max_candidates
        )

    def _summary(self, output_paths: list[Path], failed_downloads: int) -> CrawlSummary:
        self.store.deduplicate_downloads()
        output_paths = [path for path in output_paths if path.exists()]
        elapsed = (
            time.perf_counter() - self._started_at
            if self._started_at is not None
            else None
        )
        total_downloaded = self._target_completed_before + len(output_paths)
        summary = CrawlSummary(
            downloaded=total_downloaded,
            target=self._summary_target(),
            failed_downloads=failed_downloads,
            output_paths=output_paths,
            elapsed_seconds=elapsed,
            metrics={
                "attempted": self._progress_attempted,
                "skipped": self._progress_skipped,
                "resumed_downloads": self._target_completed_before,
                "downloaded_this_run": len(output_paths),
                "success_rate_percent": (
                    round(
                        100 * len(output_paths) / self._progress_attempted,
                        1,
                    )
                    if self._progress_attempted
                    else 0
                ),
            },
        )
        self._complete_target_if_satisfied(total_downloaded)
        self._report_progress("complete", message="Crawl finished")
        return summary

    def _parse_listing(self, html: str, base_url: str):
        return parse_listing(
            html,
            base_url,
            case_title_prefix=self.config.case_title_prefix,
            skip_unpublished=self.config.skip_unpublished_listing_items,
        )

    @staticmethod
    def _fallback_stem(detail_url: str) -> str:
        path = urlparse(detail_url).path
        stem = Path(path).stem
        return stem or "putusan"


def _default_chrome_user_data_dir() -> Path:
    return Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "User Data"


def _chrome_profile_ignore(directory: str, names: list[str]) -> set[str]:
    ignored_names = {
        "Cache",
        "Code Cache",
        "Crashpad",
        "DawnGraphiteCache",
        "DawnWebGPUCache",
        "GPUCache",
        "GrShaderCache",
        "GraphiteDawnCache",
        "LOCK",
        "LOG",
        "LOG.old",
        "ShaderCache",
        "SingletonCookie",
        "SingletonLock",
        "SingletonSocket",
        "component_crx_cache",
        "extensions_crx_cache",
    }
    return {
        name
        for name in names
        if name in ignored_names
        or name.startswith("BrowserMetrics")
        or name.startswith("CrashpadMetrics")
        or name.endswith(".tmp")
    }


def _is_transient_page_read_error(exc: PlaywrightError) -> bool:
    message = str(exc).lower()
    markers = (
        "page.content: unable to retrieve content because the page is navigating",
        "execution context was destroyed",
        "cannot find context with specified id",
        "most likely because of a navigation",
    )
    return any(marker in message for marker in markers)


def _is_retryable_navigation_error(exc: PlaywrightError, page_url: str) -> bool:
    if page_url.startswith("chrome-error://"):
        return True
    message = str(exc).lower()
    markers = (
        "interrupted by another navigation",
        "chrome-error://chromewebdata/",
        "net::err_",
        "navigation timeout",
        "page.goto: timeout",
    )
    return any(marker in message for marker in markers)


def _is_allowed_detail_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.netloc.lower() == ALLOWED_HOST
        and parsed.path.startswith("/direktori/putusan/")
        and parsed.path.lower().endswith(".html")
    )


def _is_allowed_listing_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.netloc.lower() == ALLOWED_HOST
        and parsed.path.startswith("/direktori/index/")
        and parsed.path.lower().endswith(".html")
    )


def _is_allowed_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.netloc.lower() == ALLOWED_HOST
        and parsed.path.startswith("/direktori/download_file/")
        and "/pdf/" in parsed.path
    )


def _ensure_allowed_detail_url(url: str) -> None:
    if not _is_allowed_detail_url(url):
        raise RuntimeError(f"refusing non-Putusan detail URL: {url}")


def _ensure_allowed_listing_url(url: str) -> None:
    if not _is_allowed_listing_url(url):
        raise RuntimeError(f"refusing non-Putusan listing URL: {url}")


def _ensure_allowed_pdf_url(url: str) -> None:
    if not _is_allowed_pdf_url(url):
        raise RuntimeError(f"refusing non-Putusan PDF URL: {url}")


def _detect_chrome_version_main() -> int | None:
    chrome_app_dirs = [
        Path(r"C:\Program Files\Google\Chrome\Application"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application"),
        Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "Application",
    ]
    for app_dir in chrome_app_dirs:
        if not app_dir.exists():
            continue
        for child in app_dir.iterdir():
            match = re.fullmatch(r"(\d+)\.\d+\.\d+\.\d+", child.name)
            if match:
                return int(match.group(1))

    candidates = [
        "chrome.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for executable in candidates:
        try:
            completed = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        match = re.search(r"(\d+)\.", completed.stdout)
        if match:
            return int(match.group(1))
    return None


def _find_chrome_executable() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "Application" / "chrome.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _listing_page_url(start_url: str, page_index: int) -> str:
    if page_index <= 1:
        return start_url

    parsed = urlparse(start_url)
    path = parsed.path
    path = re.sub(r"/page/\d+\.html$", ".html", path)
    if path.endswith(".html"):
        path = f"{path[:-5]}/page/{page_index}.html"
    else:
        path = f"{path.rstrip('/')}/page/{page_index}.html"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme}://{parsed.netloc}{path}{query}"


def _http_challenge_page(html: str) -> bool:
    title_match = re.search(r"<title[^>]*>\s*([^<]+)", html, flags=re.I)
    title = title_match.group(1).strip().lower() if title_match else ""
    if "just a moment" in title:
        return True
    text = re.sub(r"<[^>]+>", " ", html).lower()
    challenge_markers = (
        "checking if the site connection is secure",
        "verify you are human",
        "needs to review the security of your connection",
    )
    return any(marker in text for marker in challenge_markers)


def _parse_retry_after_seconds(value: object) -> float | None:
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return max(0.0, seconds)





