import os
import json
import zarr
import numpy
import argparse
from tqdm import tqdm
from tokenizers import Tokenizer

    
def preprocess_stream_to_zarr(dataset_path:str,output_file:str):
    z = zarr.open(output_file,"w")
    
    if "tokenized_data" in z:
        token_store = z["tokenized_data"]
    else:
        token_store = z.create_dataset("tokenized_data",shape=(0,),
                                       compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=2),
                                       dtype="i4",chunks=100_000)

    for file in os.listdir(dataset_path):
        if file.endswith(".jsonl"):
            filepath = os.path.join(dataset_path,file)
            print(f"Extracting from file -- {filepath}\n")
            with open(filepath,"r",encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    txt = data["content"] + "</S>"
                    encoded = tokenizer.encode(txt)
                    encoded_txt = encoded.ids
                    encoded_txt = numpy.array(encoded_txt,dtype=numpy.int32)
                    ## Store in token_store
                    token_store.append(encoded_txt)
    print(f"Saved {token_store.shape[0]} tokens to Zarr store {output_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Save the tokenized dataset")
    parser.add_argument("--input_dir",type=str,default="./",
                    help="Directory containing input *.jsonl files ")
    parser.add_argument("--output_file",default="./encoded_token.zarr",type=str,
                    help="Path to the output zarr file")
    parser.add_argument("--tokenizer_file",type=str,default="./tokenizer.json",
                        help="tokenizer json file")
    arg = parser.parse_args()
    
    tokenizer = Tokenizer.from_file(arg.tokenizer_file)
    preprocess_stream_to_zarr(arg.input_dir,arg.output_file)
    