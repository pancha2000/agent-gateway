"""
Lightweight web search + page text extraction.

Memory discipline (1 GB VPS):
  * stdlib `html.parser` only - no BeautifulSoup/lxml, no headless browser.
  * pages are streamed and cut off at WEB_MAX_PAGE_BYTES, so a huge page can
    never balloon RAM.
  * a handful of pages are fetched at once (semaphore), results are cached in
    a tiny in-process TTL cache (64 entries).

Search backends (tried in this order):
  1. Brave Search API   - only if BRAVE_API_KEY is set (reliable, free tier)
  2. DuckDuckGo HTML    - no key needed, but may rate-limit datacenter IPs
"""
from __future__ import annotations

import asyncio
import html
import ipaddress
import logging
import re
import socket
import time
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import httpx

log = logging.getLogger("gateway.web")

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
_FETCH_SEMAPHORE = asyncio.Semaphore(3)


class SearchError(RuntimeError):
    """No backend could produce results."""


# =========================================================================
# Tiny TTL cache
# =========================================================================
class TTLCache:
    def __init__(self, max_items: int = 64, ttl: float = 600.0) -> None:
        self._data: dict[Any, tuple[float, Any]] = {}
        self._max, self._ttl = max_items, ttl

    def get(self, key: Any) -> Any | None:
        item = self._data.get(key)
        if item and item[0] > time.monotonic():
            return item[1]
        self._data.pop(key, None)
        return None

    def put(self, key: Any, value: Any) -> None:
        if len(self._data) >= self._max:                      # drop oldest
            self._data.pop(next(iter(self._data)))
        self._data[key] = (time.monotonic() + self._ttl, value)


# =========================================================================
# DuckDuckGo HTML parsing
# =========================================================================
def _unwrap_ddg(href: str) -> str:
    """DDG wraps links as //duckduckgo.com/l/?uddg=<real-url>; unwrap them."""
    if href.startswith("//"):
        href = "https:" + href
    p = urlparse(href)
    if p.netloc.endswith("duckduckgo.com") and p.path.startswith("/l/"):
        target = parse_qs(p.query).get("uddg")
        if target:
            return target[0]
    return href


class _DDGParser(HTMLParser):
    """Collects <a class="result__a"> (title+url) and class="result__snippet"."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._cur: dict[str, str] | None = None
        self._mode: str | None = None       # "title" | "snippet" while capturing
        self._tag = ""
        self._depth = 0
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._mode:                      # nested tag inside the captured element
            if tag == self._tag:
                self._depth += 1
            return
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            self._cur = {"title": "", "url": _unwrap_ddg(a.get("href") or ""), "snippet": ""}
            self.results.append(self._cur)
            self._begin("title", tag)
        elif "result__snippet" in classes and self._cur is not None:
            self._begin("snippet", tag)

    def _begin(self, mode: str, tag: str) -> None:
        self._mode, self._tag, self._depth, self._buf = mode, tag, 1, []

    def handle_endtag(self, tag: str) -> None:
        if self._mode and tag == self._tag:
            self._depth -= 1
            if self._depth == 0 and self._cur is not None:
                self._cur[self._mode] = " ".join("".join(self._buf).split())
                self._mode = None

    def handle_data(self, data: str) -> None:
        if self._mode:
            self._buf.append(data)


def parse_ddg_html(markup: str, limit: int) -> list[dict[str, str]]:
    parser = _DDGParser()
    parser.feed(markup)
    out = []
    for r in parser.results:
        host = urlparse(r["url"]).netloc
        # skip ads / internal links / empty entries
        if r["url"].startswith("http") and "duckduckgo.com" not in host and r["title"]:
            out.append(r)
        if len(out) >= limit:
            break
    return out


async def _search_ddg(client: httpx.AsyncClient, query: str, limit: int) -> list[dict[str, str]]:
    resp = await client.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )
    results = parse_ddg_html(resp.text, limit) if resp.status_code == 200 else []
    if not results:
        # 202 / "anomaly" pages are DDG's bot challenge.
        raise SearchError(
            f"DuckDuckGo returned no results (HTTP {resp.status_code}); it may be rate-limiting "
            "this IP. Set BRAVE_API_KEY for a more reliable backend."
        )
    return results


# =========================================================================
# Brave Search API
# =========================================================================
_TAG_RE = re.compile(r"<[^>]+>")


async def _search_brave(client: httpx.AsyncClient, key: str, query: str, limit: int) -> list[dict[str, str]]:
    resp = await client.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": limit},
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
    )
    if resp.status_code != 200:
        raise SearchError(f"Brave API HTTP {resp.status_code}")
    items = resp.json().get("web", {}).get("results", [])
    return [
        {
            "title": html.unescape(_TAG_RE.sub("", i.get("title", ""))),
            "url": i.get("url", ""),
            "snippet": html.unescape(_TAG_RE.sub("", i.get("description", ""))),
        }
        for i in items[:limit] if i.get("url")
    ]


# =========================================================================
# Page fetching + text extraction (SSRF-safe)
# =========================================================================
async def _resolve(host: str, port: int) -> list[str]:
    """DNS lookup (separate function so tests can stub it)."""
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [i[4][0] for i in infos]


async def assert_public_url(url: str) -> None:
    """
    Refuse non-http(s), odd ports and any host resolving to a private/loopback/
    link-local address. This stops the AI (or a prompt-injected page) from
    pointing the gateway at the VPS's own internal services or cloud metadata
    endpoint (169.254.169.254).
    """
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("unsupported URL")
    if p.port not in (None, 80, 443):
        raise ValueError("port not allowed")
    for ip in await _resolve(p.hostname, p.port or (443 if p.scheme == "https" else 80)):
        if not ipaddress.ip_address(ip).is_global:
            raise ValueError("non-public address")


class _TextExtractor(HTMLParser):
    """HTML -> readable text. Keeps <pre> blocks verbatim (docs are full of code)."""

    SKIP = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "iframe"}
    BREAK = {"p", "div", "br", "li", "tr", "section", "article", "table", "ul", "ol",
             "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self._pre = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "pre":
            self._pre += 1
        if tag in self.BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "pre" and self._pre:
            self._pre -= 1
        if tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        self.parts.append(data if self._pre else " ".join(data.split()) + " ")


def html_to_text(markup: str, max_chars: int) -> str:
    ex = _TextExtractor()
    ex.feed(markup)
    text = "".join(ex.parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]


async def fetch_page_text(
    client: httpx.AsyncClient, url: str, *, max_bytes: int, max_chars: int,
) -> str | None:
    """Download one page (streamed, size-capped) and return its text, or None."""
    async with _FETCH_SEMAPHORE:
        try:
            for _ in range(4):                                # follow <=3 redirects manually
                await assert_public_url(url)                  # re-validated at EVERY hop
                async with client.stream(
                    "GET", url, headers={"User-Agent": USER_AGENT}, follow_redirects=False,
                ) as resp:
                    if resp.is_redirect and resp.headers.get("location"):
                        url = urljoin(url, resp.headers["location"])
                        continue
                    if resp.status_code >= 400:
                        return None
                    ctype = resp.headers.get("content-type", "").lower()
                    if "html" not in ctype and not ctype.startswith("text/"):
                        return None
                    buf = bytearray()
                    async for chunk in resp.aiter_bytes():
                        buf += chunk
                        if len(buf) >= max_bytes:
                            break
                    body = bytes(buf[:max_bytes]).decode(resp.encoding or "utf-8", errors="replace")
                return html_to_text(body, max_chars) if "html" in ctype else body[:max_chars]
        except (httpx.HTTPError, ValueError, OSError) as exc:
            log.info("fetch skipped %s: %s", url, exc)
    return None


# =========================================================================
# Public entry point
# =========================================================================
async def search(
    client: httpx.AsyncClient, *, query: str, max_results: int, fetch_top: int,
    brave_key: str | None, max_page_bytes: int, max_page_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Returns (engine_name, results). Raises SearchError if every backend fails."""
    engine, results, errors = "", [], []

    backends = []
    if brave_key:
        backends.append(("brave", lambda: _search_brave(client, brave_key, query, max_results)))
    backends.append(("duckduckgo", lambda: _search_ddg(client, query, max_results)))

    for name, run in backends:
        try:
            results, engine = await run(), name
            break
        except (SearchError, httpx.HTTPError) as exc:
            errors.append(f"{name}: {exc}")
            log.warning("search backend %s failed: %s", name, exc)
    if not results:
        raise SearchError("; ".join(errors) or "no results")

    if fetch_top:
        pages = await asyncio.gather(*[
            fetch_page_text(client, r["url"], max_bytes=max_page_bytes, max_chars=max_page_chars)
            for r in results[:fetch_top]
        ])
        for r, text in zip(results, pages):
            r["content"] = text
    return engine, results


def build_context(results: list[dict[str, Any]], max_chars: int = 8000) -> str:
    """Prompt-ready text block for the local LLM."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        block = f"[{i}] {r['title']}\n{r['url']}\n{r['snippet']}"
        if r.get("content"):
            block += f"\n--- page text ---\n{r['content']}"
        parts.append(block)
    return "\n\n".join(parts)[:max_chars]
