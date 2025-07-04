import argparse
import zarr
import numpy as np
import torch
import torch.distributed as dist

def get_batched_data(input_file,batch_size,context_len,device="cuda"):
    z = zarr.open(input_file,"r")
    dataset = z["tokenized_data"]
    dataset_size = dataset.shape[0]
    print(f"Data size  : {dataset_size} Context_len : {context_len}")
    total_data = (dataset_size-1)//context_len
    loc_index = np.arange(total_data)
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    index = loc_index[rank::world_size] 
    np.random.shuffle(index)
    num_batches = len(index)//batch_size
    
    for i in range(num_batches):
        data_index = index[i*batch_size:(i+1)*batch_size]*context_len
        data = [dataset[j:j+context_len+1] for j in data_index]
        batch = torch.tensor(np.stack(data),dtype=torch.long)
        x = batch[:,:context_len].to(device=device)
        y = batch[:,1:context_len+1].to(device=device)
        yield x,y
    """
    count = 0
    itr = 0 
    while True:
        if count + batch_size > len(index):
            count = 0
            np.random.shuffle(index)
            print(f"Finished iteration : {itr}")
            itr += 1
        data_index = index[count:count+batch_size]*context_len
        data = [dataset[i:i+context_len+1] for i in data_index]
        if len(data) == 0:  # just in case
            print(f"[Rank {rank}] No data found in batch, skipping...")
            continue
        batch = np.stack(data)
        batch = torch.tensor(batch,dtype=torch.long)
        x = batch[:,:context_len].to(device=device)
        y = batch[:,1:context_len+1].to(device=device)
        count += batch_size
        yield x,y
    """
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Load the tokenized data")
    parser.add_argument("--input_file",type=str,default="./encoded_tokens.zarr",
                        help="Path of the tokenized zarr file")
    parser.add_argument("--batch_size",type=str,default=256)
    parser.add_argument("--context_size",type=str,default=64)
    arg = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    get_batched_data(arg.input_file,arg.batch_size,arg.context_size,device=device)
    