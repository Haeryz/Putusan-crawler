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
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse
from urllib.request import urlopen

from playwright.sync_api import (
    BrowserContext,
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from .parsing import looks_like_challenge, parse_listing, parse_pdf_link, parse_title
from .storage import CrawlRecord, JsonlStore, sanitize_filename, unique_path, verify_pdf

DEFAULT_START_URL = (
    "https://putusan3.mahkamahagung.go.id/direktori/index/"
    "kategori/pidana-khusus-1.html"
)
ALLOWED_HOST = "putusan3.mahkamahagung.go.id"


class ChallengeBlockedError(RuntimeError):
    """Raised when a challenge page does not clear through normal browser execution."""


@dataclass(frozen=True)
class CrawlConfig:
    start_url: str = DEFAULT_START_URL
    out_dir: Path = Path("downloads")
    profile_dir: Path = Path(".browser-profile")
    target_downloads: int = 10
    headless: bool = True
    timeout_seconds: int = 120
    max_candidates: int | None = None
    retry_attempts: int = 3
    delay_seconds: float = 0.0
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


@dataclass(frozen=True)
class CrawlSummary:
    downloaded: int
    target: int
    failed_downloads: int
    output_paths: list[Path]
    elapsed_seconds: float | None = None
    metrics: dict[str, float | int] = field(default_factory=dict)


@dataclass(frozen=True)
class BulkDownloadResult:
    record: CrawlRecord
    elapsed_seconds: float
    bytes_downloaded: int = 0


class PutusanCrawler:
    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self.store = JsonlStore(config.out_dir)
        self.pdf_dir = config.out_dir / "pdfs"

    def run(self) -> CrawlSummary:
        if self.config.browser_backend == "managed-chrome":
            return self._run_managed_chrome()
        if self.config.browser_backend == "undetected-chrome":
            return self._run_undetected_chrome()
        if self.config.browser_backend == "playwright":
            return self._run_playwright()
        if self.config.browser_backend == "playwright-cdp":
            return self._run_playwright_cdp()
        raise ValueError(f"unknown browser backend: {self.config.browser_backend}")

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
                with suppress(Exception):
                    browser.close()
                if should_close_chrome:
                    with suppress(Exception):
                        chrome_process.terminate()

    def _run_playwright(self) -> CrawlSummary:
        self.store.prepare()
        downloaded_urls = self.store.downloaded_detail_urls()
        output_paths: list[Path] = []
        failed_downloads = 0
        visited_this_run: set[str] = set()
        candidate_count = 0
        listing_url: str | None = self.config.start_url if self.config.crawl_listing else None

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
                    if len(output_paths) >= self.config.target_downloads:
                        break
                    if detail_url in visited_this_run:
                        continue
                    candidate_count += 1
                    visited_this_run.add(detail_url)
                    result = self._download_case(context, page, detail_url)
                    if result.status == "downloaded" and result.output_path:
                        output_paths.append(Path(result.output_path))
                        downloaded_urls.add(detail_url)
                    elif result.status == "error":
                        failed_downloads += 1
                    self.store.append(result)

                while listing_url and len(output_paths) < self.config.target_downloads:
                    self._goto_and_wait(page, listing_url)
                    listing_html = page.content()
                    links = parse_listing(listing_html, page.url)
                    self.store.log(
                        f"listing url={page.url} cases={len(links.case_urls)} next={links.next_url}"
                    )

                    for detail_url in links.case_urls:
                        if len(output_paths) >= self.config.target_downloads:
                            break
                        if self._candidate_limit_reached(candidate_count):
                            return self._summary(output_paths, failed_downloads)
                        if detail_url in downloaded_urls or detail_url in visited_this_run:
                            continue

                        candidate_count += 1
                        visited_this_run.add(detail_url)
                        result = self._download_case(context, page, detail_url)
                        if result.status == "downloaded" and result.output_path:
                            output_paths.append(Path(result.output_path))
                            downloaded_urls.add(detail_url)
                        elif result.status == "error":
                            failed_downloads += 1
                        self.store.append(result)

                        if self.config.delay_seconds:
                            page.wait_for_timeout(int(self.config.delay_seconds * 1000))

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
        downloaded_urls = self.store.downloaded_detail_urls()
        output_paths: list[Path] = []
        failed_downloads = 0
        visited_this_run: set[str] = set()
        candidate_count = 0
        listing_url: str | None = self.config.start_url if self.config.crawl_listing else None

        for detail_url in self.config.detail_urls:
            if len(output_paths) >= self.config.target_downloads:
                break
            if detail_url in visited_this_run:
                continue
            candidate_count += 1
            visited_this_run.add(detail_url)
            result = self._download_case(context, page, detail_url)
            if result.status == "downloaded" and result.output_path:
                output_paths.append(Path(result.output_path))
                downloaded_urls.add(detail_url)
            elif result.status == "error":
                failed_downloads += 1
            self.store.append(result)

        if listing_url and self.config.browser_backend == "managed-chrome":
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

        while listing_url and len(output_paths) < self.config.target_downloads:
            self._goto_and_wait(page, listing_url)
            listing_html = page.content()
            links = parse_listing(listing_html, page.url)
            self.store.log(
                f"playwright-cdp-listing url={page.url} cases={len(links.case_urls)} next={links.next_url}"
            )

            for detail_url in links.case_urls:
                if len(output_paths) >= self.config.target_downloads:
                    break
                if self._candidate_limit_reached(candidate_count):
                    return self._summary(output_paths, failed_downloads)
                if detail_url in downloaded_urls or detail_url in visited_this_run:
                    continue

                candidate_count += 1
                visited_this_run.add(detail_url)
                result = self._download_case(context, page, detail_url)
                if result.status == "downloaded" and result.output_path:
                    output_paths.append(Path(result.output_path))
                    downloaded_urls.add(detail_url)
                elif result.status == "error":
                    failed_downloads += 1
                self.store.append(result)

                if self.config.delay_seconds:
                    page.wait_for_timeout(int(self.config.delay_seconds * 1000))

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
        while listing_url and len(output_paths) < self.config.target_downloads:
            self._goto_and_wait(page, listing_url)
            current_listing_url = page.url
            listing_html = page.content()
            links = parse_listing(listing_html, current_listing_url)
            self.store.log(
                f"managed-listing-clicks url={current_listing_url} "
                f"cases={len(links.case_urls)} next={links.next_url}"
            )

            for detail_url in links.case_urls:
                if len(output_paths) >= self.config.target_downloads:
                    break
                if self._candidate_limit_reached(candidate_count):
                    return self._summary(output_paths, failed_downloads)
                if detail_url in downloaded_urls or detail_url in visited_this_run:
                    continue

                candidate_count += 1
                visited_this_run.add(detail_url)
                result = self._download_case_by_click(context, page, detail_url)
                if result.status == "downloaded" and result.output_path:
                    output_paths.append(Path(result.output_path))
                    downloaded_urls.add(detail_url)
                elif result.status == "error":
                    failed_downloads += 1
                self.store.append(result)

                if len(output_paths) >= self.config.target_downloads:
                    break

                self._return_to_listing(page, current_listing_url)
                if self.config.delay_seconds:
                    page.wait_for_timeout(int(self.config.delay_seconds * 1000))

            listing_url = links.next_url

        return self._summary(output_paths, failed_downloads)

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
        downloaded_urls = self.store.downloaded_detail_urls()
        output_paths: list[Path] = []
        failed_downloads = 0
        visited_this_run: set[str] = set()
        candidate_count = 0
        listing_url: str | None = self.config.start_url if self.config.crawl_listing else None

        driver = self._launch_undetected_chrome()
        should_close_driver = True
        try:
            for detail_url in self.config.detail_urls:
                if len(output_paths) >= self.config.target_downloads:
                    break
                if detail_url in visited_this_run:
                    continue
                candidate_count += 1
                visited_this_run.add(detail_url)
                result = self._download_case_undetected(driver, detail_url)
                if result.status == "downloaded" and result.output_path:
                    output_paths.append(Path(result.output_path))
                    downloaded_urls.add(detail_url)
                elif result.status == "error":
                    failed_downloads += 1
                self.store.append(result)

            while listing_url and len(output_paths) < self.config.target_downloads:
                self._uc_goto_and_wait(driver, listing_url)
                listing_html = driver.page_source
                links = parse_listing(listing_html, driver.current_url)
                self.store.log(
                    f"undetected-listing url={driver.current_url} "
                    f"cases={len(links.case_urls)} next={links.next_url}"
                )

                for detail_url in links.case_urls:
                    if len(output_paths) >= self.config.target_downloads:
                        break
                    if self._candidate_limit_reached(candidate_count):
                        return self._summary(output_paths, failed_downloads)
                    if detail_url in downloaded_urls or detail_url in visited_this_run:
                        continue

                    candidate_count += 1
                    visited_this_run.add(detail_url)
                    result = self._download_case_undetected(driver, detail_url)
                    if result.status == "downloaded" and result.output_path:
                        output_paths.append(Path(result.output_path))
                        downloaded_urls.add(detail_url)
                    elif result.status == "error":
                        failed_downloads += 1
                    self.store.append(result)

                    if self.config.delay_seconds:
                        time.sleep(self.config.delay_seconds)

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
        downloaded_urls = self.store.downloaded_detail_urls()
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
                    self.config.target_downloads + len(downloaded_urls),
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
            if len(queue) >= self.config.target_downloads:
                break

        output_paths: list[Path] = []
        failed_downloads = 0
        total_bytes = 0
        path_lock = Lock()
        max_workers = max(1, self.config.parallel_downloads)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._download_case_parallel, session_state, detail_url, path_lock)
                for detail_url in queue
            ]
            for future in as_completed(futures):
                result = future.result()
                self.store.append(result.record)
                if result.record.status == "downloaded" and result.record.output_path:
                    output_paths.append(Path(result.record.output_path))
                    total_bytes += result.bytes_downloaded
                else:
                    failed_downloads += 1

        elapsed = time.perf_counter() - started_at
        return CrawlSummary(
            downloaded=len(output_paths),
            target=self.config.target_downloads,
            failed_downloads=failed_downloads,
            output_paths=output_paths,
            elapsed_seconds=elapsed,
            metrics={
                "queued": len(queue),
                "parallel_downloads": max_workers,
                "bytes_downloaded": total_bytes,
                "downloads_per_second": len(output_paths) / elapsed if elapsed else 0,
            },
        )

    def _collect_listing_detail_urls(
        self, driver, start_url: str, target_count: int
    ) -> tuple[list[str], dict[str, object]]:
        collected: list[str] = []
        seen: set[str] = set()
        session_state: dict[str, object] | None = None
        page_index = 1
        max_pages = max(1, (self.config.max_candidates or target_count * 3) // 20 + 2)

        while len(collected) < target_count and page_index <= max_pages:
            page_url = _listing_page_url(start_url, page_index)
            self._uc_goto_and_wait(driver, page_url)
            session_state = self._requests_state_from_driver(driver)
            links = parse_listing(driver.page_source, driver.current_url)
            self.store.log(
                f"parallel-listing page={page_index} url={driver.current_url} "
                f"cases={len(links.case_urls)}"
            )
            for detail_url in links.case_urls:
                if detail_url in seen:
                    continue
                seen.add(detail_url)
                collected.append(detail_url)
                if len(collected) >= target_count:
                    break
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
        self._wait_for_cdp(port)
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
        self._wait_for_cdp(port)
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
        deadline = time.monotonic() + self.config.timeout_seconds
        passive_printed = False
        interactive_printed = False
        last_status_at = 0.0

        while time.monotonic() < deadline:
            network_error = self._uc_network_error_visible(driver)
            if network_error:
                raise RuntimeError(network_error)
            if not self._uc_challenge_visible(driver):
                return
            now = time.monotonic()
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
                time.sleep(min(2 * attempt, 10))

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
        html = page.content()
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
        result = page.evaluate(
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
                    base64: btoa(binary)
                };
            }
            """,
            pdf_url,
        )
        if not result["ok"]:
            raise RuntimeError(f"PDF fetch failed with HTTP {result['status']}")
        _ensure_allowed_pdf_url(result["url"])
        body = base64.b64decode(result["base64"])
        output_path.write_bytes(body)

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
        page.goto(url, wait_until="domcontentloaded", timeout=self.config.timeout_seconds * 1000)
        self._wait_for_accessible_page(page)

    def _wait_for_accessible_page(self, page: Page) -> None:
        timeout_seconds = (
            self.config.manual_clearance_timeout_seconds
            if self.config.browser_backend == "managed-chrome"
            else self.config.timeout_seconds
        )
        deadline = time.monotonic() + timeout_seconds
        printed = False

        while time.monotonic() < deadline:
            html = page.content()
            title = page.title()
            if parse_pdf_link(html, page.url) or parse_listing(html, page.url).case_urls:
                return
            if not looks_like_challenge(html, title):
                return
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

    def _candidate_limit_reached(self, candidate_count: int) -> bool:
        return (
            self.config.max_candidates is not None
            and candidate_count >= self.config.max_candidates
        )

    def _summary(self, output_paths: list[Path], failed_downloads: int) -> CrawlSummary:
        return CrawlSummary(
            downloaded=len(output_paths),
            target=self.config.target_downloads,
            failed_downloads=failed_downloads,
            output_paths=output_paths,
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


def _is_allowed_detail_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.netloc.lower() == ALLOWED_HOST
        and parsed.path.startswith("/direktori/putusan/")
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
