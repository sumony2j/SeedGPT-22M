import os
import json
import pandas as pd
import sys

dataset_dir = "./SeedGPT_20M/fine-tuning/lmsys_chat_dir"

output_file = "./lmsys_chat.jsonl"


with open(output_file,"w",encoding="utf-8") as f:
    for dataset_file in os.listdir(dataset_dir):
        dataset_path = os.path.join(dataset_dir,dataset_file)
        dataset = pd.read_parquet(dataset_path)
        for idx,data in dataset.iterrows():
            if data["language"] == "English":
                record = data["conversation"].tolist()
                json.dump(record,f,ensure_ascii=False)
                f.write("\n")
