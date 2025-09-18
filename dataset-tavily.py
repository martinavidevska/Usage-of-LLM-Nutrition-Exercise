import requests
from transformers import pipeline
from dotenv import load_dotenv
import os
import csv

load_dotenv()

TAVILY_API_ENDPOINT = "https://api.tavily.com/search"
API_KEY = os.getenv("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")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)


def search_tavily(query, search_depth="advanced", max_results=10, include_images=False, include_answer=True):
    payload = {
        "api_key": "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",
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
    return f"Provide a detailed explanation on {query} including practical tips, scientific evidence, and examples. Focus on nutrition, exercise, and wellness."


def save_to_csv(query, result, file_path="queries_results.csv"):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([query, result])


def extract_and_summarize(results):
    summarized_content = summarize_content(results)
    return summarized_content


def main():
    queries = [
        "How should I train on rest days?",
        "What's a good full-body routine for consistency?",
        "How can I design an intense, challenging workout?",
        "How do I stay motivated in fitness?",
        "How can I improve mental toughness and deal with challenges?",
        "What's a good overall workout routine?",
        "How should I structure cardio in relation to weight training?",
        "How do I stay consistent with my workouts?",
        "What's a good lifting routine?",
        "How can I track my progress in the gym?",
        "How do I manage cheat meals effectively?",
        "What mindset should I have during workouts?",
        "How can I recover better from pain or soreness?",
        "How should I plan rest and recovery?",
        "What's the best way to track my weight management?",
        "What's a good routine to build mental toughness?",
        "How should I eat after workouts?",
        "How do I deal with gym anxiety?",
        "How do I stay focused during training?",
        "What's the hardest part of maintaining a fitness routine?",
        "How do I handle specific challenges like leg day or circuits?",
        "How do I improve performance in exercises like lifting or cardio?",
        "What's a good routine for full-body fitness?",
        "How do I maintain discipline in the gym?",
        "How should I handle rest, recovery, and progress tracking?",
        "How do I deal with setbacks or lack of motivation?",
        "What's the best way to track mental toughness and consistency?",
        "What are the top sources of vitamin B6 for brain health?",
        "How do I deal with progress?",
        "What's the best way to track my progress?",
        "How do I deal with cheat meals?",
        "What's a good full-body quitting routine?",
        "Give me motivation for circuits.",
        "Talk to me about cheat meals.",
        "What's the best way to maintain fitness?",
        "What's the hardest part of staying focused?",
        "What does 'consistent' mean in a workout context?",
        "Talk to me about circuits.",
        "Give me a killer gym workout.",
        "How should I approach leg day?",
        "What does 'circuit' mean in fitness?",
        "What's a good consistent routine?",
        "Give me fitness motivation.",
        "What's the best way to handle failure?",
        "Why is rest important?",
        "Tell me about mental toughness and how to improve it.",
        "What should I do when I feel unmotivated in fitness?",
        "What's the best way to track my weight management?",
        "What's a good full-body routine for cheat days?",
        "What's a good overall workout routine?",
        "What does 'overweight' mean in a fitness context?",
        "What's a good quitting routine?",
        "Give me motivation for focus.",
        "What mindset should I have to improve progress?",
        "What's the hardest part about progress?",
        "How do I maintain consistency?",
        "Give me motivation for failure.",
        "How do I stay motivated?",
        "What should I do when I feel behind?",
        "How should I train for cheat meals?",
        "How should I train if I am unprepared?",
        "What's a good routine to build toughness?",
        "Tell me about mental toughness.",
        "What mindset should I have to stay disciplined?",
        "How should I train when working out?",
        "How do I improve my workouts?",
        "What should I eat after gym sessions?",
        "How do I handle cheat meals?",
        "How do I deal with rest days?",
        "Why is gym anxiety important?",
        "What should I do when I feel like failing?",
        "What's the best way to track consistency?",
        "How should I train if overweight?",
        "Give me motivation for leg day.",
        "What should I do when I feel unfit?",
        "How should I train to stay disciplined?",
        "Why are circuits important?",
        "What's the best mindset for fitness?",
        "Give me a killer leg day workout.",
        "What should I eat after disciplined training?",
        "How do I handle being overweight?",
        "How do I stay motivated in workouts?",
        "What's the hardest part about failing?",
        "How do I improve before training sessions?",
        "What should I do on rest days?",
        "Tell me how to deal with overweight effectively.",
        "What does 'disciplined' mean in fitness?",
        "How do I stay consistent?",
        "What's a good progress routine?",
        "What mindset should I have during rest?",
        "What's a good mental toughness routine?",
        "How do I stay motivated on cheat days?",
        "What's the hardest part about staying disciplined?",
        "What's a good full-body routine for gym anxiety?",
        "How do I stay motivated in circuits?",
        "What's a good rest routine?",
        "How do I deal with progress effectively?",
        "Should I do cardio before or after weights?",
        "What should I do when lifting?",
        "Give me motivation for consistency.",
        "What does 'mental toughness' mean?",
        "Should I do cardio on cheat days or after weights?",
        "What's the best way to stay focused?",
        "What mindset should I have on leg day?",
        "Should I do cardio consistently or after weights?",
        "Talk to me about preparing before workouts.",
        "How do I deal with being unprepared?",
        "What does 'focused' mean in fitness?",
        "How do I stay consistent in workouts?",
        "What's a good disciplined routine?",
        "What's the hardest part about staying disciplined?",
        "What do you think about consistency?",
        "How do I improve focus?",
        "What's a good focused routine?",
        "What does 'fitness' mean?",
        "Give me progress motivation."
    ]

    for query in queries:
        processed_query = custom_query_processing(query)
        results = search_tavily(processed_query, max_results=10, include_answer=True)

        return_text = results.get('answer', [])
        final_summary = extract_and_summarize(return_text)
        save_to_csv(query, final_summary)


if __name__ == "__main__":
    main()
