import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

@tool
def scrape_webpage_tool(url: str) -> str:
    """
    Fetches and extracts text content from a webpage URL.
    Use this to read the content of a job portal or a specific job offer page.
    """
    try:
        # Use a generic User-Agent to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.extract()
            
        # Extract readable text
        text = soup.get_text(separator="\n", strip=True)
        
        # Return a truncated version to avoid exceeding token limits
        return text[:10000]
    except Exception as e:
        return f"Error fetching webpage {url}: {str(e)}"
