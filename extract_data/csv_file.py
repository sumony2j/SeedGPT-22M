import pandas as pd
import os
import json
import re
import html
import unicodedata

input_dir = "/LocalData/deepseek/dataset/cleaned_bookcorpus/train"
output_file = "./refined_bookcorpus_train_data.jsonl"


def clean_text(text):
    text = html.unescape(text)  # Decode HTML entities
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'https?://\S+', ' ', text)  # Remove URLs
    text = re.sub(r'www\.\S+', ' ', text)  # Remove www links
    text = re.sub(r"©?\s?\d{4}.*All rights reserved.*", "", text, flags=re.I)  # Copyright
    text = unicodedata.normalize("NFKC", text)  # Normalize Unicode
    
    # Remove emojis and all non-alphanumeric characters except punctuation and space
    text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)  # keep basic punctuation
    text = re.sub(r'[_]', '', text)  # optional: remove underscores
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    return text

all_data = []

for file in os.listdir(input_dir):
    p_file = f"{input_dir}/{file}"
    print(f"Processing File : {p_file}")
    dataset = pd.read_csv(p_file,header=None)
    for txt in dataset[0]:
        if isinstance(txt, str) and txt.strip():  # Skip empty or NaN rows
            cleaned = clean_text(txt)
            if cleaned:
                all_data.append({"text": cleaned})

with open(output_file, "w", encoding="utf-8") as f:
    for item in all_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")