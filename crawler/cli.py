from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from .crawler import (
    DEFAULT_START_URL,
    CrawlConfig,
    CrawlInventory,
    CrawlProgress,
    PutusanCrawler,
)


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
    crawl.add_argument(
        "--download-all",
        action="store_true",
        help="download every matching listing item until pagination ends",
    )
    crawl.add_argument(
        "--case-title-prefix",
        help=(
            "only queue listing items whose visible title starts with this text, "
            'for example "Putusan PN"'
        ),
    )
    crawl.add_argument(
        "--include-unpublished-listing-items",
        action="store_true",
        help="queue listing items marked Unpublish; by default they are skipped",
    )
    crawl.add_argument(
        "--count-only",
        action="store_true",
        help="scan listing pagination and count matching downloadable documents without downloading",
    )
    crawl.add_argument(
        "--json-summary",
        action="store_true",
        help="print count/download summary as JSON for scripts",
    )
    crawl.add_argument(
        "--plain",
        action="store_true",
        help="disable Rich animation and tables for plain terminal output",
    )
    crawl.add_argument("--timeout-seconds", type=int, default=120)
    crawl.add_argument("--max-candidates", type=int)
    crawl.add_argument("--retry-attempts", type=int, default=3)
    crawl.add_argument("--delay-seconds", type=float, default=0.0)
    crawl.add_argument(
        "--rate-limit-backoff-seconds",
        type=float,
        default=30.0,
        help="base wait after HTTP 429; multiplied by the retry attempt",
    )
    crawl.add_argument(
        "--challenge-cooldown-seconds",
        type=float,
        default=0.0,
        help="wait after a Cloudflare challenge clears before resuming navigation",
    )
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
        "--fast-fetch-timeout-seconds",
        type=int,
        default=15,
        help="per-request timeout for managed Chrome fast mode when --parallel-downloads > 1",
    )
    crawl.add_argument(
        "--count-parallel-pages",
        type=int,
        default=16,
        help="listing pages to fetch concurrently after Cloudflare clearance in --count-only mode",
    )
    crawl.add_argument(
        "--no-refresh-profile-snapshot",
        action="store_true",
        help="reuse .browser-profile/managed-chrome instead of recopying the Chrome profile",
    )
    crawl.add_argument(
        "--restart-listing",
        action="store_true",
        help="discard saved pagination progress and begin again from --start-url",
    )
    crawl.add_argument(
        "--new-target",
        action="store_true",
        help="start a fresh target instead of resuming an interrupted target of the same size",
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

    if not args.download_all and args.target_downloads <= 0:
        parser.error("--target-downloads must be greater than 0")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be greater than 0")
    if args.retry_attempts <= 0:
        parser.error("--retry-attempts must be greater than 0")
    if args.delay_seconds < 0:
        parser.error("--delay-seconds must be zero or greater")
    if args.rate_limit_backoff_seconds < 0:
        parser.error("--rate-limit-backoff-seconds must be zero or greater")
    if args.challenge_cooldown_seconds < 0:
        parser.error("--challenge-cooldown-seconds must be zero or greater")
    if args.parallel_downloads <= 0:
        parser.error("--parallel-downloads must be greater than 0")
    if args.debug_hold_seconds < 0:
        parser.error("--debug-hold-seconds must be zero or greater")
    if args.manual_clearance_timeout_seconds <= 0:
        parser.error("--manual-clearance-timeout-seconds must be greater than 0")
    if args.fast_fetch_timeout_seconds <= 0:
        parser.error("--fast-fetch-timeout-seconds must be greater than 0")
    if args.count_parallel_pages <= 0:
        parser.error("--count-parallel-pages must be greater than 0")
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
        download_all=args.download_all,
        headless=args.headless,
        timeout_seconds=args.timeout_seconds,
        max_candidates=args.max_candidates,
        retry_attempts=args.retry_attempts,
        delay_seconds=args.delay_seconds,
        rate_limit_backoff_seconds=args.rate_limit_backoff_seconds,
        challenge_cooldown_seconds=args.challenge_cooldown_seconds,
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
        fast_fetch_timeout_seconds=args.fast_fetch_timeout_seconds,
        count_parallel_pages=args.count_parallel_pages,
        case_title_prefix=args.case_title_prefix,
        skip_unpublished_listing_items=not args.include_unpublished_listing_items,
        resume_listing=not args.restart_listing,
        resume_target=not args.new_target,
    )

    rich_enabled = not args.json_summary and not args.plain
    console = Console()
    error_console = Console(stderr=True)

    try:
        if args.count_only:
            crawler = PutusanCrawler(config)
            inventory = _run_with_spinner(
                console,
                "Scanning listing pages and counting downloadable documents",
                crawler.count_downloadable,
                enabled=rich_enabled,
            )
            if args.json_summary:
                print(json.dumps(_inventory_to_dict(inventory), ensure_ascii=False))
            elif rich_enabled:
                _print_inventory_rich(console, inventory)
            else:
                _print_inventory_plain(inventory)
            return 0

        target_label = (
            "all matching PDFs"
            if args.download_all
            else f"up to {args.target_downloads} PDF(s)"
        )
        if rich_enabled:
            summary = _run_download_with_progress(console, config, target_label)
        else:
            summary = PutusanCrawler(config).run()
    except Exception as exc:  # noqa: BLE001 - CLI boundary should be human-readable.
        if rich_enabled:
            error_console.print(
                Panel(
                    f"[bold red]{type(exc).__name__}[/bold red]: {exc}",
                    title="Crawler failed",
                    border_style="red",
                    box=box.ASCII,
                )
            )
        else:
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json_summary:
        print(
            json.dumps(
                {
                    "downloaded": summary.downloaded,
                    "target": summary.target,
                    "failed_downloads": summary.failed_downloads,
                    "output_paths": [str(path) for path in summary.output_paths],
                    "metrics": summary.metrics,
                },
                ensure_ascii=False,
            )
        )
        return 0 if _download_target_satisfied(summary) else 1

    if rich_enabled:
        _print_download_summary_rich(console, summary)
    else:
        print(f"Downloaded: {summary.downloaded}/{_target_label(summary)}")
        print(f"Failed downloads: {summary.failed_downloads}")
        if summary.elapsed_seconds is not None:
            print(f"Elapsed seconds: {summary.elapsed_seconds:.2f}")
        for key, value in summary.metrics.items():
            print(f"{key}: {value}")
        for path in summary.output_paths:
            print(path)

    return 0 if _download_target_satisfied(summary) else 1


def _read_detail_urls(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _inventory_to_dict(inventory: CrawlInventory) -> dict[str, object]:
    return {
        "total_downloadable": inventory.total_downloadable,
        "pages_scanned": inventory.pages_scanned,
        "already_downloaded": inventory.already_downloaded,
        "remaining": inventory.remaining,
        "pages": [
            {
                "page_index": page.page_index,
                "listing_url": page.listing_url,
                "downloadable": page.downloadable,
                "already_downloaded": page.already_downloaded,
                "remaining": page.remaining,
                "detail_urls": page.detail_urls,
            }
            for page in inventory.pages
        ],
    }


def _run_with_spinner(console: Console, message: str, action, *, enabled: bool):
    if not enabled:
        return action()

    with Progress(
        SpinnerColumn("line"),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        progress.add_task(message, total=None)
        return action()


def _run_download_with_progress(
    console: Console,
    config: CrawlConfig,
    target_label: str,
):
    total = None if config.download_all else config.target_downloads
    with Progress(
        SpinnerColumn("line"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TextColumn("[yellow]{task.fields[remaining]} left"),
        TextColumn(
            "[green]ok {task.fields[successful]}[/green] "
            "[red]fail {task.fields[failed]}[/red] "
            "[magenta]skip {task.fields[skipped]}[/magenta] "
            "[dim]tried {task.fields[attempted]}[/dim]"
        ),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        refresh_per_second=8,
    ) as progress:
        task_id = progress.add_task(
            f"Preparing {target_label}",
            total=total,
            remaining="?" if total is None else str(total),
            successful=0,
            failed=0,
            skipped=0,
            attempted=0,
        )

        def update(event: CrawlProgress) -> None:
            remaining = (
                "?"
                if total is None
                else str(max(0, total - event.successful))
            )
            current = _progress_item_label(event.detail_url)
            descriptions = {
                "starting": event.message or "Preparing crawler",
                "target_resumed": event.message or "Resuming download target",
                "resuming": f"Resuming from {current}",
                "downloading": f"Downloading {current}",
                "downloaded": f"Saved {current}",
                "failed": f"Failed {current}",
                "skipped": f"Skipped {current}",
                "complete": event.message or "Crawl finished",
            }
            progress.update(
                task_id,
                completed=event.successful,
                description=descriptions.get(event.phase, event.phase.title()),
                remaining=remaining,
                successful=event.successful,
                failed=event.failed,
                skipped=event.skipped,
                attempted=event.attempted,
                refresh=True,
            )

        crawler = PutusanCrawler(replace(config, progress_callback=update))
        return crawler.run()


def _progress_item_label(detail_url: str | None) -> str:
    if not detail_url:
        return "candidate"
    path = urlparse(detail_url).path.rstrip("/")
    return path.rsplit("/", 1)[-1] or detail_url


def _download_target_satisfied(summary) -> bool:
    return summary.target is None or summary.downloaded >= summary.target


def _target_label(summary) -> str:
    return "all" if summary.target is None else str(summary.target)


def _print_inventory_plain(inventory: CrawlInventory) -> None:
    print(f"Total downloadable: {inventory.total_downloadable}")
    print(f"Pages scanned: {inventory.pages_scanned}")
    print(f"Already downloaded: {inventory.already_downloaded}")
    print(f"Remaining: {inventory.remaining}")
    for page in inventory.pages:
        print(
            f"Page {page.page_index}: {page.downloadable} downloadable, "
            f"{page.remaining} remaining - {page.listing_url}"
        )


def _print_inventory_rich(console: Console, inventory: CrawlInventory) -> None:
    summary = Table.grid(expand=True)
    summary.add_column(justify="center")
    summary.add_column(justify="center")
    summary.add_column(justify="center")
    summary.add_column(justify="center")
    summary.add_row(
        _metric("Downloadable", inventory.total_downloadable, "cyan"),
        _metric("Pages", inventory.pages_scanned, "magenta"),
        _metric("Already Done", inventory.already_downloaded, "green"),
        _metric("Remaining", inventory.remaining, "yellow"),
    )
    console.print(
        Panel(summary, title="Putusan Inventory", border_style="cyan", box=box.ASCII)
    )

    table = Table(title="Per-page scan", show_lines=False, box=box.ASCII)
    table.add_column("Page", justify="right", style="cyan", no_wrap=True)
    table.add_column("Downloadable", justify="right")
    table.add_column("Done", justify="right", style="green")
    table.add_column("Remaining", justify="right", style="yellow")
    table.add_column("URL", overflow="fold")
    for page in inventory.pages:
        table.add_row(
            str(page.page_index),
            str(page.downloadable),
            str(page.already_downloaded),
            str(page.remaining),
            page.listing_url,
        )
    console.print(table)


def _print_download_summary_rich(console: Console, summary) -> None:
    status = "complete" if _download_target_satisfied(summary) else "incomplete"
    border = "green" if status == "complete" else "yellow"
    summary_grid = Table.grid(expand=True)
    summary_grid.add_column(justify="center")
    summary_grid.add_column(justify="center")
    summary_grid.add_column(justify="center")
    summary_grid.add_row(
        _metric("Downloaded", f"{summary.downloaded}/{_target_label(summary)}", "green"),
        _metric("Failed", summary.failed_downloads, "red" if summary.failed_downloads else "green"),
        _metric("Status", status, border),
    )
    console.print(
        Panel(summary_grid, title="Crawl Summary", border_style=border, box=box.ASCII)
    )

    if summary.metrics:
        metrics = Table(title="Metrics", box=box.ASCII)
        metrics.add_column("Name", style="cyan")
        metrics.add_column("Value", justify="right")
        for key, value in summary.metrics.items():
            metrics.add_row(str(key), str(value))
        console.print(metrics)

    if summary.output_paths:
        outputs = Table(title="Downloaded PDFs", box=box.ASCII)
        outputs.add_column("#", justify="right", style="cyan", no_wrap=True)
        outputs.add_column("Path", overflow="fold")
        visible_paths = summary.output_paths[-20:]
        first_index = len(summary.output_paths) - len(visible_paths) + 1
        for index, path in enumerate(visible_paths, start=first_index):
            outputs.add_row(str(index), str(path))
        console.print(outputs)
        omitted = len(summary.output_paths) - len(visible_paths)
        if omitted:
            console.print(f"[dim]{omitted} earlier downloaded path(s) omitted.[/dim]")


def _metric(label: str, value, color: str) -> Text:
    text = Text()
    text.append(f"{value}\n", style=f"bold {color}")
    text.append(label, style="dim")
    return text


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
