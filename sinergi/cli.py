from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .crawler import DEFAULT_START_URL, CrawlConfig, PutusanCrawler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sinergi",
        description="Download verified PDF files from Putusan MA case listings.",
    )
    subparsers = parser.add_subparsers(dest="command")

    crawl = subparsers.add_parser("crawl", help="crawl listing pages and download PDFs")
    crawl.add_argument("--start-url", default=DEFAULT_START_URL)
    crawl.add_argument(
        "--no-listing",
        action="store_true",
        help="only process --detail-url/--detail-file inputs and do not crawl listings",
    )
    crawl.add_argument(
        "--detail-url",
        action="append",
        default=[],
        help="case detail URL to process before listing traversal; may be repeated",
    )
    crawl.add_argument(
        "--detail-file",
        type=Path,
        help="text file containing one case detail URL per line",
    )
    crawl.add_argument("--out-dir", type=Path, default=Path("downloads"))
    crawl.add_argument("--profile-dir", type=Path, default=Path(".browser-profile"))
    crawl.add_argument("--target-downloads", type=int, default=10)
    crawl.add_argument("--timeout-seconds", type=int, default=120)
    crawl.add_argument("--max-candidates", type=int)
    crawl.add_argument("--retry-attempts", type=int, default=3)
    crawl.add_argument("--delay-seconds", type=float, default=0.0)
    crawl.add_argument(
        "--parallel-downloads",
        type=int,
        default=1,
        help="download detail pages/PDFs concurrently after browser session clearance",
    )
    crawl.add_argument(
        "--keep-browser-open-on-error",
        action="store_true",
        help="leave Chrome open when an undetected-chrome run fails for debugging",
    )
    crawl.add_argument(
        "--debug-hold-seconds",
        type=int,
        default=0,
        help="seconds to keep Chrome/process alive after an error when debugging",
    )
    crawl.add_argument(
        "--browser-backend",
        choices=("managed-chrome", "undetected-chrome", "playwright", "playwright-cdp"),
        default="managed-chrome",
        help="browser automation backend; managed-chrome is the default for Cloudflare",
    )
    crawl.add_argument(
        "--cdp-port",
        type=int,
        help="remote debugging port for --browser-backend playwright-cdp",
    )
    crawl.add_argument(
        "--chrome-version-main",
        type=int,
        help="installed Chrome major version for undetected-chromedriver, e.g. 148",
    )
    crawl.add_argument(
        "--browser-channel",
        choices=("chrome", "msedge"),
        help="use an installed browser channel instead of Playwright's bundled Chromium",
    )
    crawl.add_argument(
        "--chrome-user-data-dir",
        type=Path,
        help="Chrome User Data directory to copy from for --browser-backend managed-chrome",
    )
    crawl.add_argument(
        "--chrome-profile",
        default="Profile 4",
        help="Chrome profile directory to copy/reuse for --browser-backend managed-chrome",
    )
    crawl.add_argument(
        "--manual-clearance-timeout-seconds",
        type=int,
        default=300,
        help="seconds to wait for manual Cloudflare clearance in visible managed Chrome",
    )
    crawl.add_argument(
        "--no-refresh-profile-snapshot",
        action="store_true",
        help="reuse .browser-profile/managed-chrome instead of recopying the Chrome profile",
    )
    browser_mode = crawl.add_mutually_exclusive_group()
    browser_mode.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="run without opening a browser window",
    )
    browser_mode.add_argument(
        "--headed",
        action="store_false",
        dest="headless",
        help="open a visible browser window; this is the default",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        args = parser.parse_args(["crawl", *(argv or [])])

    if args.target_downloads <= 0:
        parser.error("--target-downloads must be greater than 0")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be greater than 0")
    if args.retry_attempts <= 0:
        parser.error("--retry-attempts must be greater than 0")
    if args.parallel_downloads <= 0:
        parser.error("--parallel-downloads must be greater than 0")
    if args.debug_hold_seconds < 0:
        parser.error("--debug-hold-seconds must be zero or greater")
    if args.manual_clearance_timeout_seconds <= 0:
        parser.error("--manual-clearance-timeout-seconds must be greater than 0")
    if not args.chrome_profile.strip():
        parser.error("--chrome-profile must not be empty")

    detail_urls = list(args.detail_url)
    if args.detail_file:
        detail_urls.extend(_read_detail_urls(args.detail_file))

    config = CrawlConfig(
        start_url=args.start_url,
        out_dir=args.out_dir,
        profile_dir=args.profile_dir,
        target_downloads=args.target_downloads,
        headless=args.headless,
        timeout_seconds=args.timeout_seconds,
        max_candidates=args.max_candidates,
        retry_attempts=args.retry_attempts,
        delay_seconds=args.delay_seconds,
        browser_channel=args.browser_channel,
        browser_backend=args.browser_backend,
        chrome_version_main=args.chrome_version_main,
        cdp_port=args.cdp_port,
        parallel_downloads=args.parallel_downloads,
        keep_browser_open_on_error=args.keep_browser_open_on_error,
        debug_hold_seconds=args.debug_hold_seconds,
        detail_urls=tuple(detail_urls),
        crawl_listing=not args.no_listing,
        chrome_user_data_dir=args.chrome_user_data_dir,
        chrome_profile=args.chrome_profile,
        manual_clearance_timeout_seconds=args.manual_clearance_timeout_seconds,
        refresh_profile_snapshot=not args.no_refresh_profile_snapshot,
    )

    try:
        summary = PutusanCrawler(config).run()
    except Exception as exc:  # noqa: BLE001 - CLI boundary should be human-readable.
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Downloaded: {summary.downloaded}/{summary.target}")
    print(f"Failed downloads: {summary.failed_downloads}")
    if summary.elapsed_seconds is not None:
        print(f"Elapsed seconds: {summary.elapsed_seconds:.2f}")
    for key, value in summary.metrics.items():
        print(f"{key}: {value}")
    for path in summary.output_paths:
        print(path)

    return 0 if summary.downloaded >= summary.target and summary.failed_downloads == 0 else 1


def _read_detail_urls(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
