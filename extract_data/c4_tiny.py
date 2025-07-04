import os
import json

## Read c4-tiny dataset

input_file = "c4-train.00000-of-01024.json"

def text_itr(input_file):
    with open(input_file,"r",encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]

output_file = "training_data.jsonl"

num_data = 200000

with open(output_file,"a",encoding="utf-8") as f:
    for i,data in enumerate(text_itr(input_file=input_file)):
        if i >= num_data:
            print("\n#######Done#######\n")
            break
        json.dump({"text":data},f,ensure_ascii=False)
        f.write("\n")

    