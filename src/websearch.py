import base64
import logging
import warnings
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from config import SearchConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*duckduckgo_search.*")


def google_search(query: str, api_key: str, cse_id: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the web using Google Custom Search JSON API."""
    results = []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cse_id, "q": query, "num": max_results}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        for item in data.get("items", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "href": item.get("link", ""),
                    "body": item.get("snippet", ""),
                }
            )
    except Exception as e:
        logger.warning("Error during Google search: %s", e)
    return results


def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search DuckDuckGo with fallbacks for the current duckduckgo_search package."""
    attempts = [
        {"backend": "html", "safesearch": "off"},
        {"backend": "lite", "safesearch": "off"},
        {"backend": "auto", "safesearch": "off"},
        {"backend": "html", "safesearch": "moderate"},
    ]
    for kwargs in attempts:
        try:
            original_warn = warnings.warn
            warnings.warn = _ignore_duckduckgo_rename_warning(original_warn)
            try:
                with DDGS(timeout=20) as ddgs:
                    raw_results = ddgs.text(query, max_results=max_results, **kwargs)
            finally:
                warnings.warn = original_warn
            results = _normalize_duckduckgo_results(raw_results, max_results)
            if results:
                return results
        except Exception as e:
            logger.warning("DuckDuckGo search attempt failed (%s): %s", kwargs, e)
    return []


def _ignore_duckduckgo_rename_warning(original_warn):
    def warn(message, *args, **kwargs):
        if "duckduckgo_search" in str(message) and "renamed to `ddgs`" in str(message):
            return None
        return original_warn(message, *args, **kwargs)

    return warn


def _normalize_duckduckgo_results(raw_results, max_results: int) -> List[Dict[str, str]]:
    results = []
    for item in raw_results or []:
        href = item.get("href") or item.get("url") or ""
        title = item.get("title") or ""
        body = item.get("body") or item.get("snippet") or ""
        if title and href:
            results.append({"title": title, "href": href, "body": body})
        if len(results) >= max_results:
            break
    return results


def bing_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search Bing by parsing the public HTML results page. No API key required."""
    try:
        response = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "count": max_results, "setlang": "en-US"},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=15,
        )
        response.raise_for_status()
    except Exception as e:
        logger.warning("Error during Bing search: %s", e)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for item in soup.select("li.b_algo"):
        heading = item.find("h2")
        link = heading.find("a") if heading else None
        if not link:
            continue
        title = link.get_text(" ", strip=True)
        href = _decode_bing_href(link.get("href") or "")
        body_tag = item.find("p")
        body = body_tag.get_text(" ", strip=True) if body_tag else ""
        if title and href:
            results.append({"title": title, "href": href, "body": body})
        if len(results) >= max_results:
            break
    return results


def tavily_search(query: str, api_key: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search with Tavily. Requires a Tavily API key."""
    if not api_key:
        logger.warning("Tavily search selected but search.tavily_api_key is empty")
        return []
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
            },
            headers={"Content-Type": "application/json"},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.warning("Error during Tavily search: %s", e)
        return []

    results = []
    for item in data.get("results", []) or []:
        title = item.get("title") or ""
        href = item.get("url") or item.get("href") or ""
        body = item.get("content") or item.get("snippet") or ""
        if title and href:
            results.append({"title": title, "href": href, "body": body})
        if len(results) >= max_results:
            break
    return results


def _decode_bing_href(href: str) -> str:
    parsed = urlparse(href)
    encoded = parse_qs(parsed.query).get("u", [""])[0]
    if not encoded:
        return href
    if encoded.startswith("a1"):
        encoded = encoded[2:]
    padding = "=" * (-len(encoded) % 4)
    try:
        return base64.urlsafe_b64decode((encoded + padding).encode("ascii")).decode("utf-8")
    except Exception:
        return href


def web_search(query: str, config: Optional[SearchConfig] = None, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the web. Returns dictionaries with title, href, and body."""
    provider = (config.provider if config else "duckduckgo").lower()
    configured_max_results = getattr(config, "max_results", None) if config else None
    max_results = configured_max_results or max_results

    if provider == "google":
        if config and config.google_api_key and config.google_cse_id:
            return google_search(query, config.google_api_key, config.google_cse_id, max_results)
        logger.warning("Google search selected but google_api_key/google_cse_id is missing")
        return []

    if provider == "tavily":
        return tavily_search(query, getattr(config, "tavily_api_key", None) or "", max_results=max_results)

    if provider == "bing":
        return bing_search(query, max_results=max_results)

    if provider == "duckduckgo":
        return duckduckgo_search(query, max_results=max_results)

    logger.warning("Unknown search provider '%s'; falling back to Bing HTML search", provider)
    return bing_search(query, max_results=max_results)
