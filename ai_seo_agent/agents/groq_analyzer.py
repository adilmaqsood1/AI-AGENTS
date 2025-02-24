import groq

class GroqKeywordAnalyzer:
    def __init__(self, api_key):
        self.client = groq.Client(api_key=api_key)

    def analyze_keywords(self, keywords):
        """Use Groq API to generate a detailed SEO analysis report."""
        keyword_text = ", ".join(keywords)

        # Define the prompt for Groq
        prompt = f"""
        Analyze these keywords: {keyword_text}. Provide a detailed SEO report including:
        1. **Keyword Categorization**: Group keywords into categories like Brand, Services, Core Values, etc.
        2. **Sentiment Analysis**: Determine the overall sentiment and highlight positive/neutral keywords.
        3. **SEO Recommendations**: Suggest actionable on-page, content, and technical SEO strategies.
        4. **Competitor Gap Analysis**: Identify missing keywords or opportunities.
        5. **Actionable Next Steps**: Provide a clear roadmap for SEO improvement.

        Format the report as a JSON object with the following structure:
        {{
          "categories": {{
            "Brand & Team": ["keyword1", "keyword2", ...],
            "Core Services": ["keyword1", "keyword2", ...],
            ...
          }},
          "sentiment": {{
            "overall": "positive/neutral/negative",
            "keywords_indicating_trust": ["keyword1", "keyword2", ...]
          }},
          "recommendations": {{
            "on_page": ["Optimize title tags with 'Award-Winning Digital Marketing'", ...],
            "content": ["Publish blog on 'AI-Driven SEO'", ...],
            "technical": ["Fix broken links using Screaming Frog", ...]
          }},
          "next_steps": [
            "Build a 'Client Success Hub' showcasing ROI metrics",
            "Create a 'Cybersecurity Checklist' PDF for lead generation"
          ]
        }}
        """

        # Call Groq API
        response = self.client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",  
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI trained in SEO and keyword analysis. Analyze the following keywords and provide a detailed SEO report."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1000
        )

        # Extract the analysis from the response
        analysis = response.choices[0].message.content
        return analysis
