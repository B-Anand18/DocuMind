"""
crawler.py
----------
Web crawling utilities for the URL ingestion pipeline.

Strategy:
  1. Try requests (fast, works for static HTML sites).
  2. If the page comes back empty / too short, fall back to Playwright
     (handles JS-rendered sites).

The main function `extract_text_from_url` crawls the parent URL and up to
`max_child_urls` same-domain child pages concurrently, returning a list of
LangChain Document objects ready for chunking.
"""

import logging
import concurrent.futures
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Minimum characters required to consider a requests response "real" content.
MIN_CONTENT_LEN = 200


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_with_requests(url: str) -> str:
    """Fetch raw HTML via requests (fast, works for static sites)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (DocuMind RAG crawler/1.0)"}
        res = requests.get(url, timeout=15, headers=headers)
        res.raise_for_status()
        return res.text
    except Exception as exc:
        logger.warning(f"requests failed for {url}: {exc}")
        return ""


def fetch_with_playwright(url: str) -> str:
    """Fallback: fetch JS-rendered HTML via Playwright (headless Chromium)."""
    try:
        from playwright.sync_api import sync_playwright  # lazy import

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=20_000)
            content = page.content()
            browser.close()
            return content
    except Exception as exc:
        logger.warning(f"Playwright failed for {url}: {exc}")
        return ""


def fetch_page(url: str) -> str:
    """Try requests first; fall back to Playwright if content is too sparse."""
    html = fetch_with_requests(url)
    if not html or len(html.strip()) < MIN_CONTENT_LEN:
        logger.info(f"Sparse content from requests for {url}, trying Playwright…")
        html = fetch_with_playwright(url)
    return html


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def html_to_text(html: str) -> str:
    """Convert raw HTML to clean plain text (strips scripts, styles, nav, etc.)."""
    soup = BeautifulSoup(html, "lxml")
    # Remove boilerplate tags
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def collect_child_links(parent_url: str, html: str, max_links: int) -> list[str]:
    """Return up to max_links same-domain links found on the parent page."""
    base_domain = urlparse(parent_url).netloc
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    links: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # Resolve relative URLs, strip fragments
        full_url, _ = urldefrag(urljoin(parent_url, href))
        if urlparse(full_url).netloc != base_domain:
            continue
        if full_url in seen or full_url == parent_url:
            continue
        seen.add(full_url)
        links.append(full_url)
        if len(links) >= max_links:
            break

    return links


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_documents_from_url(url: str, max_child_urls: int = 30) -> list[Document]:
    """
    Crawl `url` and up to `max_child_urls` same-domain child pages.
    Returns a list of LangChain Documents, one per crawled page.

    Each Document's metadata contains:
      - source: the page URL
      - page:   0 (to keep consistent with PDF metadata)
    """
    documents: list[Document] = []

    # ── Parent page ─────────────────────────────────────────
    logger.info(f"Fetching parent URL: {url}")
    parent_html = fetch_page(url)
    if not parent_html:
        logger.error(f"Could not fetch parent URL: {url}")
        return documents

    parent_text = html_to_text(parent_html)
    if parent_text:
        documents.append(Document(page_content=parent_text, metadata={"source": url, "page": 0}))

    # ── Child pages (same domain, concurrent) ───────────────
    child_links = collect_child_links(url, parent_html, max_child_urls)
    logger.info(f"Found {len(child_links)} child link(s). Fetching concurrently…")

    def fetch_child(child_url: str) -> Document | None:
        html = fetch_page(child_url)
        if not html:
            return None
        text = html_to_text(html)
        if not text:
            return None
        return Document(page_content=text, metadata={"source": child_url, "page": 0})

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_child, link): link for link in child_links}
        for future in concurrent.futures.as_completed(futures):
            child_url = futures[future]
            try:
                doc = future.result()
                if doc:
                    documents.append(doc)
                    logger.info(f"  ✔ {child_url}")
            except Exception as exc:
                logger.warning(f"  ✘ {child_url}: {exc}")

    logger.info(f"Crawl complete. {len(documents)} page(s) collected from {url}")
    return documents
