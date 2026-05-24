from sinergi.parsing import looks_like_challenge, parse_listing, parse_pdf_link


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
