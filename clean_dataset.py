import pandas as pd
import re

df = pd.read_csv('queries_results.csv')

def remove_unwanted_sentences(text):
    if isinstance(text, str):
        filtered_sentences = re.sub(r'\b(For confidential support call|For more info)\b.*?(\.|\n)', '', text, flags=re.IGNORECASE)
        return filtered_sentences.strip()
    return text

df['Result'] = df['Result'].apply(remove_unwanted_sentences)

df.to_csv('cleaned_dataset.csv', index=False)

