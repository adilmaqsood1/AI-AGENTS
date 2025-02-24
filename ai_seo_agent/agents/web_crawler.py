import requests
from bs4 import BeautifulSoup

class SimpleWebCrawler:
    def extract_keywords(self, url):
        """Crawl a website and extract keywords using requests and BeautifulSoup."""
        try:
            # Fetch the webpage
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract text and split into keywords
            text = soup.get_text(separator=" ", strip=True)
            keywords = list(set(text.split()))  # Simple keyword extraction
            
            return keywords
        except Exception as e:
            raise Exception(f"Failed to crawl {url}: {str(e)}")