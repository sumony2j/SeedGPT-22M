import pandas as pd
import os
import json

input_dir = "./"
output_file = "training_data.jsonl"

all_data = []
for file in os.listdir(input_dir):
    p_file = f"{input_dir}/{file}"
    print(f"Processing File : {p_file}")
    data = pd.read_parquet(path=p_file)
    data = data[data["text"].str.strip().astype(bool)]
    all_data.append(data)

df = pd.concat(all_data,ignore_index=True)

with open(output_file,"a",encoding="utf-8") as f:
    for row in df["text"]:
        json.dump({"text":row},f,ensure_ascii=False)
        f.write("\n")

