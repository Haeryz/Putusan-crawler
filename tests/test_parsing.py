from crawler.parsing import (
    looks_like_challenge,
    parse_listing,
    parse_listing_last_page_index,
    parse_pdf_link,
)


def test_parse_listing_extracts_case_links_and_next() -> None:
    html = """
    <a href="/direktori/putusan/abc.html">case 1</a>
    <a href="https://putusan3.mahkamahagung.go.id/direktori/putusan/def.html">case 2</a>
    <a href="https://jdih.mahkamahagung.go.id/direktori/putusan/ghi.html">skip other host</a>
    <a href="/not-a-case.html">skip</a>
    <a class="next" href="/direktori/index/kategori/pidana-khusus-1/page/2.html">Next</a>
    """

    links = parse_listing(
        html,
        "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/pidana-khusus-1.html",
    )

    assert links.case_urls == [
        "https://putusan3.mahkamahagung.go.id/direktori/putusan/abc.html",
        "https://putusan3.mahkamahagung.go.id/direktori/putusan/def.html",
    ]
    assert (
        links.next_url
        == "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/pidana-khusus-1/page/2.html"
    )
    assert (
        parse_listing_last_page_index(
            html,
            "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/pidana-khusus-1.html",
        )
        == 1
    )


def test_parse_listing_extracts_rel_next_from_pagination() -> None:
    html = """
    <ul class="pagination justify-content-center">
      <li class="page-item active"><a class="page-link" href="#">1</a></li>
      <li class="page-item"><a href="https://putusan3.mahkamahagung.go.id/direktori/index/kategori/pidana-khusus-1/page/2.html" class="page-link" data-ci-pagination-page="2">2</a></li>
      <li class="page-item"><a href="#" aria-label="Next"></a><a href="https://putusan3.mahkamahagung.go.id/direktori/index/kategori/pidana-khusus-1/page/2.html" class="page-link" data-ci-pagination-page="2" rel="next">Next</a></li>
      <li class="page-item"><a href="#" aria-label="Next"></a><a href="https://putusan3.mahkamahagung.go.id/direktori/index/kategori/pidana-khusus-1/page/499.html" class="page-link" data-ci-pagination-page="499">Last</a></li>
    </ul>
    """

    links = parse_listing(
        html,
        "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/pidana-khusus-1.html",
    )

    assert (
        links.next_url
        == "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/pidana-khusus-1/page/2.html"
    )
    assert (
        parse_listing_last_page_index(
            html,
            "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/pidana-khusus-1.html",
        )
        == 499
    )


def test_parse_listing_filters_to_pn_and_skips_unpublish_before_footer_items() -> None:
    html = """
    <article>
      <a href="/direktori/putusan/ma.html">Putusan MAHKAMAH AGUNG Nomor 1 K/PID.SUS/2026</a>
      <p>Upload : 18-04-2026</p>
    </article>
    <article>
      <a href="/direktori/putusan/pn.html">Putusan PN GARUT Nomor 5/Pid.Sus-Anak/2026/PN Grt</a>
      <p>Tanggal 11 Maret 2026 - Terdakwa</p>
    </article>
    <article>
      <a href="/direktori/putusan/pt.html">Putusan PT BANTEN Nomor 9/PID.SUS-Anak/2022/PT BTN</a>
      <p>Tanggal 8 Desember 2022</p>
    </article>
    <article>
      <a href="/direktori/putusan/unpublish.html">Putusan PN SERANG Nomor 13/Pid.Sus-Anak/2019/PN Srg</a>
      <p>Unpublish</p>
    </article>
    <ul class="pagination">
      <li><a class="page-link" href="/direktori/index/kategori/peradilan-anak-abh-1/page/2.html">Next</a></li>
    </ul>
    <aside>
      <a href="/direktori/putusan/footer.html">Putusan PN CURUP Nomor 2/Pid.Sus-Anak/2022/PN Crp</a>
    </aside>
    """

    links = parse_listing(
        html,
        "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/peradilan-anak-abh-1.html",
        case_title_prefix="Putusan PN",
    )

    assert links.case_urls == [
        "https://putusan3.mahkamahagung.go.id/direktori/putusan/pn.html",
    ]
    assert (
        links.next_url
        == "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/peradilan-anak-abh-1/page/2.html"
    )


def test_parse_pdf_link_from_goal_snippet() -> None:
    html = """
    <ul class="portfolio-meta nobottommargin">
        <li><span><i class="icon-download"></i>Download Zip</span></li>
        <li><a href="https://putusan3.mahkamahagung.go.id/direktori/download_file/bbe7ff2bcfe58bdadb9c577d880c027e/zip/zaf14cd81bd6894491e8303832343038">42/Pid.Sus/2026/PN_Pya.zip</a></li>
        <li><span><i class="icon-files"></i>Download PDF</span></li>
        <li><a href="https://putusan3.mahkamahagung.go.id/direktori/download_file/bbe7ff2bcfe58bdadb9c577d880c027e/pdf/zaf14cd81bd6894491e8303832343038">42/Pid.Sus/2026/PN_Pya.pdf</a></li>
    </ul>
    """

    link = parse_pdf_link(
        html,
        "https://putusan3.mahkamahagung.go.id/direktori/putusan/zaf14cd81bd6894491e8303832343038.html",
    )

    assert link is not None
    assert (
        link.url
        == "https://putusan3.mahkamahagung.go.id/direktori/download_file/bbe7ff2bcfe58bdadb9c577d880c027e/pdf/zaf14cd81bd6894491e8303832343038"
    )
    assert link.filename == "42/Pid.Sus/2026/PN_Pya.pdf"


def test_parse_pdf_link_rejects_alternative_hosts() -> None:
    html = """
    <a href="https://jdih.mahkamahagung.go.id/storage/uploads/putusan/case.pdf">case.pdf</a>
    """

    link = parse_pdf_link(
        html,
        "https://putusan3.mahkamahagung.go.id/direktori/putusan/zaf14cd81bd6894491e8303832343038.html",
    )

    assert link is None


def test_looks_like_challenge_detects_cloudflare_page() -> None:
    assert looks_like_challenge("<p>Checking if the site connection is secure</p>", "Just a moment")


def test_looks_like_challenge_does_not_block_accessible_page_with_cloudflare_word() -> None:
    html = """
    <html>
      <body>
        <a href="https://putusan3.mahkamahagung.go.id/direktori/download_file/hash/pdf/case">case.pdf</a>
        <script src="/cdn-cgi/cloudflare-static/email-decode.min.js"></script>
      </body>
    </html>
    """

    assert not looks_like_challenge(html, "Direktori Putusan")
