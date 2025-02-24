from agents.web_crawler import SimpleWebCrawler
from agents.groq_analyzer import GroqKeywordAnalyzer

class KeywordAnalysisWorkflow:
    def __init__(self, groq_api_key):
        self.crawler = SimpleWebCrawler()
        self.analyzer = GroqKeywordAnalyzer(groq_api_key)

    def run(self, url):
        """Run the workflow: Crawl -> Analyze -> Generate Report."""
        keywords = self.crawler.extract_keywords(url)
        analysis = self.analyzer.analyze_keywords(keywords)
        return {
            "keywords": keywords,
            "analysis": analysis
        }