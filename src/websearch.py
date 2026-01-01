from duckduckgo_search import DDGS
import requests
from typing import List, Dict, Optional
from config import SearchConfig

def google_search(query: str, api_key: str, cse_id: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using Google Custom Search JSON API.
    """
    results = []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": max_results
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        items = data.get("items", [])
        for item in items:
            results.append({
                "title": item.get("title", ""),
                "href": item.get("link", ""),
                "body": item.get("snippet", "")
            })
            
    except Exception as e:
        print(f"Error during Google search: {e}")
        # Fallback empty list or re-raise? 
        # For now return empty list so the bot handles it gracefully.
        
    return results

def web_search(query: str, config: Optional[SearchConfig] = None, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using the configured provider.
    Returns a list of dictionaries with 'title', 'href', and 'body'.
    """
    provider = "duckduckgo"
    if config:
        provider = config.provider.lower()
        
    if provider == "google" and config and config.google_api_key and config.google_cse_id:
        return google_search(query, config.google_api_key, config.google_cse_id, max_results)
    
    # Default to DuckDuckGo
    results = []
    try:
        with DDGS() as ddgs:
            ddgs_gen = ddgs.text(query, max_results=max_results)
            for r in ddgs_gen:
                results.append({
                    "title": r.get("title", ""),
                    "href": r.get("href", ""),
                    "body": r.get("body", "")
                })
    except Exception as e:
        print(f"Error during web search: {e}")
    
    return results
