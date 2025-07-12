import pandas as pd
import os
import json
import re
import unicodedata

input_dir = "/LocalData/deepseek/dataset/TinyStories/stories/data/train"
output_file = "./stories_train_data.jsonl"


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

all_data = []
for file in os.listdir(input_dir):
    p_file = f"{input_dir}/{file}"
    print(f"Processing File : {p_file}")
    data = pd.read_parquet(path=p_file)
    for txt in data["story"]:
        all_data.append(txt)


with open(output_file,"a",encoding="utf-8") as f:
    for row in all_data:
        json.dump({"text":row},f,ensure_ascii=False)
        f.write("\n")
