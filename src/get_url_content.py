import trafilatura
from bs4 import BeautifulSoup
from typing import Optional

def get_url_content(url: str, max_length: int = 5000) -> str:
    """
    Fetch and extract the main text content from a URL.
    Uses Trafilatura for intelligent extraction, falling back to BeautifulSoup
    if Trafilatura fails to find the main content.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return f"Error: Failed to download content from {url}"
        
        # 1. Try Trafilatura (Best for articles/blogs)
        content = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
        
        # 2. Fallback to BeautifulSoup (Best for generic pages where Trafilatura is too strict)
        if content is None:
            soup = BeautifulSoup(downloaded, 'html.parser')
            
            # Remove scripts, styles, and other non-text elements
            for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
                element.decompose()
            
            # Get text and clean up whitespace
            text = soup.get_text(separator=' ')
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = '\n'.join(chunk for chunk in chunks if chunk)
            
        if not content:
            return f"Error: Could not extract meaningful text from {url}"
            
        # Truncate to avoid token limits
        if len(content) > max_length:
            return content[:max_length] + "... [Content Truncated]"
            
        return content
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

if __name__ == "__main__":
    # Quick test
    test_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    print(f"Testing with {test_url}...")
    result = get_url_content(test_url, max_length=500)
    print(result)
