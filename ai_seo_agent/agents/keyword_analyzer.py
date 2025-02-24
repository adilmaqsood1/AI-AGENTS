from collections import Counter

class KeywordAnalyzer:
    def __init__(self):
        pass

    def analyze_keywords(self, keywords):
        """Analyze keywords and generate a basic report."""
        keyword_freq = Counter(keywords)
        total_keywords = len(keywords)
        keyword_percentages = {k: (v / total_keywords) * 100 for k, v in keyword_freq.items()}

        report = {
            "total_keywords": total_keywords,
            "keyword_frequencies": keyword_freq,
            "keyword_percentages": keyword_percentages
        }
        return report