import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import websearch


def test_decode_bing_redirect_url():
    href = "https://www.bing.com/ck/a?u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS8"
    assert websearch._decode_bing_href(href) == "https://example.com/"


def test_web_search_does_not_fallback_when_duckduckgo_empty(monkeypatch):
    class Config:
        provider = "duckduckgo"
        max_results = 5
        google_api_key = None
        google_cse_id = None
        tavily_api_key = None

    monkeypatch.setattr(websearch, "duckduckgo_search", lambda query, max_results=5: [])
    monkeypatch.setattr(
        websearch,
        "bing_search",
        lambda query, max_results=5: [{"title": "Bing", "href": "https://example.com", "body": "fallback"}],
    )

    assert websearch.web_search("query", Config()) == []


def test_web_search_unknown_provider_falls_back_to_bing(monkeypatch):
    class Config:
        provider = "unknown"
        max_results = 5
        google_api_key = None
        google_cse_id = None
        tavily_api_key = None

    monkeypatch.setattr(
        websearch,
        "bing_search",
        lambda query, max_results=5: [{"title": "Bing", "href": "https://example.com", "body": "fallback"}],
    )

    assert websearch.web_search("query", Config()) == [
        {"title": "Bing", "href": "https://example.com", "body": "fallback"}
    ]


def test_bing_html_parser_extracts_results(monkeypatch):
    class Response:
        text = """
        <html><body><ol>
          <li class="b_algo"><h2><a href="https://example.com">Example</a></h2><p>Body text</p></li>
        </ol></body></html>
        """

        def raise_for_status(self):
            pass

    monkeypatch.setattr(websearch.requests, "get", lambda *args, **kwargs: Response())
    assert websearch.bing_search("query") == [
        {"title": "Example", "href": "https://example.com", "body": "Body text"}
    ]


def test_tavily_search_extracts_results(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"title": "Example", "url": "https://example.com", "content": "Body text"},
                ]
            }

    monkeypatch.setattr(websearch.requests, "post", lambda *args, **kwargs: Response())
    assert websearch.tavily_search("query", "key") == [
        {"title": "Example", "href": "https://example.com", "body": "Body text"}
    ]


def test_web_search_uses_configured_max_results(monkeypatch):
    class Config:
        provider = "bing"
        max_results = 2
        google_api_key = None
        google_cse_id = None
        tavily_api_key = None

    seen = {}

    def fake_bing(query, max_results=5):
        seen["max_results"] = max_results
        return []

    monkeypatch.setattr(websearch, "bing_search", fake_bing)
    websearch.web_search("query", Config(), max_results=9)
    assert seen["max_results"] == 2
