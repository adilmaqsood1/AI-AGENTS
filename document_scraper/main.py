from crawl4ai import WebCrawler
from groq import Groq
from langgraph.graph import Graph
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
import time
import json
import os

groq_client = Groq(api_key="load_dotenv:GROQ_API_KEY")

def scrape_all_pages():
    all_texts = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://www.paklegaldatabase.com")
        page.click("text=Login")  # Adjust selector
        page.fill("input[name='email']", "adilmaqsood501@gmail.com")  # Adjust selector
        page.fill("input[name='password']", "crislynn50")  # Adjust selector
        page.click("button[type='submit']")  # Adjust selector
        page.wait_for_load_state("networkidle")
        page.click("text=Judgments")  # Adjust selector
        
        while True:
            pdf_links = page.query_selector_all("a[href*='.pdf']")
            for link in pdf_links:
                pdf_url = link.get_attribute("href")
                page.goto(pdf_url)
                time.sleep(3)
                pdf_text = page.evaluate("document.body.innerText")
                all_texts.append(pdf_text)
            next_button = page.query_selector("text=Next")  # Adjust selector
            if next_button and next_button.is_enabled():
                next_button.click()
                page.wait_for_load_state("networkidle")
            else:
                break
        browser.close()
        return all_texts

def agent1_scrape(input_data=None):  # Accept input parameter
    print("Agent 1: Scraping all pages...")
    return scrape_all_pages()

def label_text_with_groq(text):
    prompt = f"""
    Given the following legal text, label it in a structured format suitable for training a law chatbot. 
    Extract key information like case details, judgment summary, or legal terms, and format it as JSON.
    Text: {text[:1000]}
    """
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )
    return response.choices[0].message.content

def agent2_label(scraped_texts):  # Accept input from agent1
    print("Agent 2: Labeling data...")
    for text in scraped_texts:
        labeled_data = label_text_with_groq(text)
        with open("dataset.txt", "a") as f:
            f.write(labeled_data + "\n")
    print("Agent 2: Data saved to dataset.txt")
    return labeled_data  # Return last labeled data

# Define the graph
graph = Graph()
graph.add_node("scrape", agent1_scrape)
graph.add_node("label", agent2_label)
graph.add_edge("scrape", "label")
graph.set_entry_point("scrape")

# Compile and run
app = graph.compile()
result = app.invoke({})  # Pass an empty dict as initial input
print("Workflow completed. Final result:", result)

# Download the dataset.txt file
from google.colab import files
files.download("dataset.txt")