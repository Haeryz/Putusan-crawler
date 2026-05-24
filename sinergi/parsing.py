from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

CASE_DETAIL_RE = re.compile(r"/direktori/putusan/[^\"'#?]+\.html(?:$|[?#])", re.I)
DOWNLOAD_PDF_MARKERS = ("/direktori/download_file/", "/pdf/")
ALLOWED_HOST = "putusan3.mahkamahagung.go.id"


@dataclass(frozen=True)
class ListingLinks:
    case_urls: list[str]
    next_url: str | None


@dataclass(frozen=True)
class PdfLink:
    url: str
    filename: str | None


def parse_listing(html: str, base_url: str) -> ListingLinks:
    soup = BeautifulSoup(html, "html.parser")
    case_urls: list[str] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme != "https" or parsed.netloc.lower() != ALLOWED_HOST:
            continue
        path_with_query = parsed.path
        if parsed.query:
            path_with_query = f"{path_with_query}?{parsed.query}"
        if CASE_DETAIL_RE.search(path_with_query) and absolute not in seen:
            seen.add(absolute)
            case_urls.append(absolute)

    return ListingLinks(case_urls=case_urls, next_url=_find_next_url(soup, base_url))


def parse_pdf_link(html: str, base_url: str) -> PdfLink | None:
    soup = BeautifulSoup(html, "html.parser")

    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if (
            parsed.scheme == "https"
            and parsed.netloc.lower() == ALLOWED_HOST
            and all(marker in parsed.path for marker in DOWNLOAD_PDF_MARKERS)
        ):
            filename = anchor.get_text(" ", strip=True) or None
            return PdfLink(url=absolute, filename=filename)

    return None


def parse_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")

    for selector in ("h1", "h2", "h3", ".entry-title", ".post-title"):
        node = soup.select_one(selector)
        if node:
            title = node.get_text(" ", strip=True)
            if title:
                return title

    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        if title:
            return title

    return None


def looks_like_challenge(html: str, title: str | None = None) -> bool:
    haystack = f"{title or ''}\n{html}".lower()
    markers = (
        "just a moment",
        "checking if the site connection is secure",
        "verify you are human",
        "needs to review the security of your connection",
        "performing security verification",
        "cf-challenge",
        "cf-turnstile",
    )
    return any(marker in haystack for marker in markers)


def _find_next_url(soup: BeautifulSoup, base_url: str) -> str | None:
    candidates = soup.find_all("a", href=True)
    for anchor in candidates:
        rel = anchor.get("rel") or []
        text = anchor.get_text(" ", strip=True).lower()
        classes = " ".join(anchor.get("class") or []).lower()
        label = str(anchor.get("aria-label") or "").lower()

        if (
            "next" in rel
            or text in {"next", "selanjutnya", "berikutnya", ">"}
            or "next" in classes
            or label in {"next", "selanjutnya", "berikutnya"}
        ):
            return urljoin(base_url, str(anchor["href"]).strip())

    return None
