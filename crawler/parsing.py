from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

CASE_DETAIL_RE = re.compile(r"/direktori/putusan/[^\"'#?]+\.html(?:$|[?#])", re.I)
DOWNLOAD_PDF_MARKERS = ("/direktori/download_file/", "/pdf/")
ALLOWED_HOST = "putusan3.mahkamahagung.go.id"
DOWNLOAD_FILE_RE = re.compile(r"(https?://[^\s\"'<>]+|/direktori/download_file/[^\s\"'<>]+)", re.I)


@dataclass(frozen=True)
class ListingLinks:
    case_urls: list[str]
    next_url: str | None


@dataclass(frozen=True)
class PdfLink:
    url: str
    filename: str | None


def parse_listing(
    html: str,
    base_url: str,
    *,
    case_title_prefix: str | None = None,
    skip_unpublished: bool = True,
) -> ListingLinks:
    soup = BeautifulSoup(html, "html.parser")
    case_urls: list[str] = []
    seen: set[str] = set()
    normalized_prefix = case_title_prefix.casefold() if case_title_prefix else None

    for anchor in soup.find_all("a", href=True):
        if case_urls and _looks_like_pagination_anchor(anchor):
            break

        href = str(anchor["href"]).strip()
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme != "https" or parsed.netloc.lower() != ALLOWED_HOST:
            continue
        path_with_query = parsed.path
        if parsed.query:
            path_with_query = f"{path_with_query}?{parsed.query}"
        if not CASE_DETAIL_RE.search(path_with_query):
            continue

        title = anchor.get_text(" ", strip=True)
        if normalized_prefix and not title.casefold().startswith(normalized_prefix):
            continue
        if skip_unpublished and "unpublish" in _case_segment_text(anchor).casefold():
            continue
        if absolute in seen:
            continue

        seen.add(absolute)
        case_urls.append(absolute)

    return ListingLinks(case_urls=case_urls, next_url=_find_next_url(soup, base_url))


def parse_listing_last_page_index(html: str, base_url: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    last_page = _listing_page_index(base_url)

    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href or href == "#":
            continue

        text = anchor.get_text(" ", strip=True).lower()
        label = str(anchor.get("aria-label") or "").lower()
        rel = anchor.get("rel") or []
        if not (
            "last" in rel
            or text in {"last", "terakhir"}
            or label in {"last", "terakhir"}
        ):
            continue

        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if (
            parsed.scheme == "https"
            and parsed.netloc.lower() == ALLOWED_HOST
            and parsed.path.startswith("/direktori/index/")
            and parsed.path.lower().endswith(".html")
        ):
            last_page = max(last_page, _listing_page_index(absolute))

        data_page = str(anchor.get("data-ci-pagination-page") or "").strip()
        if data_page.isdigit():
            last_page = max(last_page, int(data_page))

    return last_page


def parse_pdf_link(html: str, base_url: str) -> PdfLink | None:
    soup = BeautifulSoup(html, "html.parser")

    for anchor in soup.find_all("a", href=True):
        filename = anchor.get_text(" ", strip=True) or None
        for raw_url in _pdf_url_candidates(anchor):
            absolute = urljoin(base_url, raw_url)
            if _is_putusan_pdf_download_url(absolute):
                return PdfLink(url=absolute, filename=filename)

    for node in soup.find_all(True):
        for raw_url in _pdf_url_candidates(node):
            absolute = urljoin(base_url, raw_url)
            if _is_putusan_pdf_download_url(absolute):
                filename = node.get_text(" ", strip=True) or None
                if not filename:
                    filename = _nearby_pdf_label(node)
                return PdfLink(url=absolute, filename=filename)

    return None


def _pdf_url_candidates(node: Tag) -> list[str]:
    values: list[str] = []
    for attr in ("href", "data-href", "data-url", "data-download", "data-file", "onclick"):
        value = node.get(attr)
        if not value:
            continue
        text = str(value).strip()
        if attr == "onclick":
            values.extend(match.group(1) for match in DOWNLOAD_FILE_RE.finditer(text))
        else:
            values.append(text)
    return values


def _is_putusan_pdf_download_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.netloc.lower() == ALLOWED_HOST
        and all(marker in parsed.path for marker in DOWNLOAD_PDF_MARKERS)
    )


def _nearby_pdf_label(node: Tag) -> str | None:
    for sibling in (node.previous_sibling, node.next_sibling):
        if isinstance(sibling, Tag):
            label = sibling.get_text(" ", strip=True) or None
            if label:
                return label
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
        href = str(anchor["href"]).strip()
        if not href or href == "#":
            continue
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
            return urljoin(base_url, href)

    return None


def _listing_page_index(url: str) -> int:
    match = re.search(r"/page/(\d+)\.html(?:$|[?#])", urlparse(url).path)
    if match:
        return int(match.group(1))
    return 1


def _case_segment_text(anchor: Tag) -> str:
    texts = [anchor.get_text(" ", strip=True)]

    for element in anchor.next_elements:
        if element is anchor:
            continue
        if isinstance(element, Tag):
            if element is not anchor and element.name == "a" and element.get("href"):
                href = str(element["href"]).strip()
                if CASE_DETAIL_RE.search(urlparse(href).path) or _looks_like_pagination_anchor(
                    element
                ):
                    break
            if element.name in {"script", "style"}:
                continue
        elif isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                texts.append(text)

    return " ".join(texts)


def _looks_like_pagination_anchor(anchor: Tag) -> bool:
    href = str(anchor.get("href") or "").strip()
    rel = anchor.get("rel") or []
    text = anchor.get_text(" ", strip=True).lower()
    classes = " ".join(anchor.get("class") or []).lower()
    label = str(anchor.get("aria-label") or "").lower()

    if CASE_DETAIL_RE.search(urlparse(href).path):
        return False

    return (
        "page-link" in classes
        or "pagination" in _ancestor_classes(anchor)
        or "next" in rel
        or text in {"next", "previous", "prev", "last", "selanjutnya", "berikutnya", "terakhir"}
        or text.isdigit()
        or label in {"next", "previous", "prev", "last", "selanjutnya", "berikutnya", "terakhir"}
    )


def _ancestor_classes(anchor: Tag) -> str:
    classes: list[str] = []
    for parent in anchor.parents:
        if not isinstance(parent, Tag):
            continue
        classes.extend(str(value) for value in (parent.get("class") or []))
    return " ".join(classes).lower()
