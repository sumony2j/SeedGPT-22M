import pandas as pd
import os
import json
import re
import html
import unicodedata

input_dir = "/LocalData/deepseek/dataset/cc_news/cc_news/test"
output_file = "./cc_news_test_data.jsonl"


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

all_data = []
for file in os.listdir(input_dir):
    p_file = f"{input_dir}/{file}"
    print(f"Processing File : {p_file}")
    data = pd.read_parquet(path=p_file)
    data = data[data["text"].str.strip().astype(bool)]
    data["text"] = data["text"].apply(clean_text)
    all_data.append(data)

df = pd.concat(all_data,ignore_index=True)

with open(output_file,"a",encoding="utf-8") as f:
    for row in df["text"]:
        json.dump({"text":row},f,ensure_ascii=False)
        f.write("\n")
