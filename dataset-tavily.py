import requests
from transformers import pipeline
from dotenv import load_dotenv
import os
import csv

load_dotenv()

TAVILY_API_ENDPOINT = "https://api.tavily.com/search"
API_KEY = os.getenv("TAVILY_API_KEY")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)


def search_tavily(query, search_depth="deep", max_results=10, include_images=False, include_answer=True):
    payload = {
        "api_key": API_KEY,
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results,
        "include_images": include_images,
        "include_answer": include_answer,
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(TAVILY_API_ENDPOINT, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error11: {response.status_code} - {response.text}")
        return None


def summarize_content(content):
    summary = summarizer(content, max_length=100, min_length=50, do_sample=False)
    return summary[0]['summary_text']


def custom_query_processing(query):
    return f"Provide a detailed explanation on {query} including practical tips, scientific evidence, and examples."


def save_to_csv(query, result, file_path="queries_results.csv"):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([query, result])


def extract_and_summarize(results):
    summarized_content = summarize_content(results)
    return summarized_content


def main():
    queries = [
        "What are the top sources of vitamin B6 for brain health?"
        # and 1000 more that we deleted for better visiability
    ]

    for query in queries:
        processed_query = custom_query_processing(query)
        results = search_tavily(processed_query, max_results=10, include_answer=True)

        return_text = results.get('answer', [])
        final_summary = extract_and_summarize(return_text)
        save_to_csv(query, final_summary)


if __name__ == "__main__":
    main()
