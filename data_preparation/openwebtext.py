##
## Extract data to train custom tokenizer
##
 
import os
import json
import sys

## Read OpenWebText dataset

input_dir = "/LocalData/deepseek/dataset/Pile-OpenWebText2/raw/train"
output_file = "training_data.jsonl"

num_data = 2000000

def text_itr(input_file):
    with open(input_file,"r",encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["content"]


with open(output_file,"a",encoding="utf-8") as f:
    i = 0
    for input_file in os.listdir(input_dir):
        print(f"\n Processing File is : {input_dir}/{input_file}")
        for data in text_itr(input_file=f"{input_dir}/{input_file}"):
            if i >= num_data:
                print("\n#######Done#######\n")
                sys.exit(0)
            json.dump({"text":data},f,ensure_ascii=False)
            f.write("\n")
            i = i+1