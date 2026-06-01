from pathlib import Path

from crawler.crawler import CrawlConfig, PutusanCrawler
from crawler.storage import CrawlRecord, JsonlStore


class InventoryCrawler(PutusanCrawler):
    def __init__(self, config: CrawlConfig, pages: dict[str, str]) -> None:
        super().__init__(config)
        self.pages = pages

    def _load_listing_for_count(self, page, listing_url: str, page_index: int):
        return self.pages[listing_url], listing_url


class FastInventoryPage:
    def __init__(self, url: str) -> None:
        self.url = url


class FastInventoryCrawler(PutusanCrawler):
    def __init__(self, config: CrawlConfig, pages: dict[str, str]) -> None:
        super().__init__(config)
        self.pages = pages
        self.batch_urls: list[list[str]] = []

    def _goto_and_wait(self, page: FastInventoryPage, url: str) -> None:
        page.url = url

    def _safe_page_content(self, page: FastInventoryPage, max_wait_seconds=None) -> str:
        return self.pages[page.url]

    def _fetch_listing_pages_with_page(self, page, urls: list[str]):
        self.batch_urls.append(urls)
        return [{"ok": True, "status": 200, "url": url, "text": self.pages[url]} for url in urls]


def test_count_downloadable_walks_pages_and_reports_remaining(tmp_path: Path) -> None:
    start_url = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/peradilan-anak-abh-1.html"
    page_2_url = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/peradilan-anak-abh-1/page/2.html"
    pages = {
        start_url: f"""
        <a href="/direktori/putusan/pn-1.html">Putusan PN GARUT 1/Pid.Sus-Anak/2026/PN Grt</a>
        <p>Tanggal 1 Januari 2026</p>
        <a href="/direktori/putusan/pt-1.html">Putusan PT BANTEN 1/PID.SUS-Anak/2026/PT Btn</a>
        <p>Tanggal 1 Januari 2026</p>
        <a class="page-link" rel="next" href="{page_2_url}">Next</a>
        """,
        page_2_url: """
        <a href="/direktori/putusan/pn-2.html">Putusan PN TONDANO 2/Pid.Sus-Anak/2026/PN Tnn</a>
        <p>Tanggal 2 Januari 2026</p>
        <a href="/direktori/putusan/pn-3.html">Putusan PN SERANG 3/Pid.Sus-Anak/2026/PN Srg</a>
        <p>Unpublish</p>
        """,
    }
    store = JsonlStore(tmp_path)
    store.prepare()
    store.append(
        CrawlRecord(
            status="downloaded",
            detail_url="https://putusan3.mahkamahagung.go.id/direktori/putusan/pn-1.html",
        )
    )
    crawler = InventoryCrawler(
        CrawlConfig(
            start_url=start_url,
            out_dir=tmp_path,
            case_title_prefix="Putusan PN",
        ),
        pages,
    )

    inventory = crawler._count_with_page(page=None)

    assert inventory.total_downloadable == 2
    assert inventory.pages_scanned == 2
    assert inventory.already_downloaded == 1
    assert inventory.remaining == 1
    assert [page.downloadable for page in inventory.pages] == [1, 1]
    assert [page.remaining for page in inventory.pages] == [0, 1]
    assert inventory.pages[0].detail_urls == [
        "https://putusan3.mahkamahagung.go.id/direktori/putusan/pn-1.html"
    ]


def test_count_downloadable_fast_fetches_listing_pages_in_batches(tmp_path: Path) -> None:
    start_url = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/peradilan-anak-abh-1.html"
    page_2_url = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/peradilan-anak-abh-1/page/2.html"
    page_3_url = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/peradilan-anak-abh-1/page/3.html"
    pages = {
        start_url: f"""
        <a href="/direktori/putusan/pn-1.html">Putusan PN GARUT 1/Pid.Sus-Anak/2026/PN Grt</a>
        <a class="page-link" rel="next" href="{page_2_url}">Next</a>
        <a class="page-link" data-ci-pagination-page="3" href="{page_3_url}">Last</a>
        """,
        page_2_url: """
        <a href="/direktori/putusan/pn-2.html">Putusan PN TONDANO 2/Pid.Sus-Anak/2026/PN Tnn</a>
        """,
        page_3_url: """
        <a href="/direktori/putusan/pn-3.html">Putusan PN SERANG 3/Pid.Sus-Anak/2026/PN Srg</a>
        <a href="/direktori/putusan/pt-1.html">Putusan PT BANTEN 1/PID.SUS-Anak/2026/PT Btn</a>
        """,
    }
    crawler = FastInventoryCrawler(
        CrawlConfig(
            start_url=start_url,
            out_dir=tmp_path,
            case_title_prefix="Putusan PN",
            count_parallel_pages=8,
        ),
        pages,
    )

    inventory = crawler._count_with_page(FastInventoryPage(start_url))

    assert inventory.total_downloadable == 3
    assert inventory.pages_scanned == 3
    assert [page.downloadable for page in inventory.pages] == [1, 1, 1]
    assert inventory.pages[2].detail_urls == [
        "https://putusan3.mahkamahagung.go.id/direktori/putusan/pn-3.html"
    ]
    assert crawler.batch_urls == [[page_2_url, page_3_url]]
