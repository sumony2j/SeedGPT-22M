from transformer import Transformer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR,ExponentialLR
from transformers import get_cosine_schedule_with_warmup
import torch.distributed as dist
from data.loader import get_batched_data
import torch
import src.config
from tqdm import tqdm
from utils import get_logger,plot_graph,set_seed
import os
import numpy as np
import random

def setup():
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device=device)
    dist.init_process_group(backend="nccl")
    print("Setup Done")

def cleanup():
    dist.destroy_process_group()


@torch.no_grad()
def est_loss(model, steps):
    model.eval()
    out = {}
    for data_type in ["train", "val"]:
        losses = []
        data_file = (
            src.config.TOKENIZED_TRAINING_DATA_PATH
            if data_type == "train"
            else src.config.TOKENIZED_VAL_DATA_PATH
        )
        batch_iter = get_batched_data(
            data_file,
            src.config.BATCH_SIZE,
            src.config.CONTEXT_LENGTH,
            src.config.DEVICE
        )
        for i, (xb, xy) in enumerate(batch_iter):
            if i >= steps:
                break
            _, loss = model(xb, xy)
            losses.append(loss.item())
        out[data_type] = sum(losses) / len(losses) if losses else float("inf")
    model.train()
    return out

def train(model,opt):
    scheduler = ExponentialLR(optimizer=opt,gamma=0.95)
    #logger = get_logger()
    model.train()
    for e in tqdm(range(src.config.N_EPCOHS)):
        dataset = get_batched_data(src.config.TOKENIZED_TRAINING_DATA_PATH,
                src.config.BATCH_SIZE,
                src.config.CONTEXT_LENGTH,
                src.config.DEVICE)
        for step, (xb,yb) in tqdm(enumerate(dataset)):
            _,loss = model(xb,yb)
            opt.zero_grad()
            loss = loss.mean()
            loss.backward()
            opt.step()
        scheduler.step()
        
        if dist.get_rank() == 0 and e % src.config.N_EVAL_ITRS == 0:
            train_loss,val_loss = est_loss(model,src.config.N_EVAL_ITRS).values()
            # Log in parseable format
            #logger.info(f"\nEpoch : {e} | Step : {step} | Train Loss : {train_loss:.6f} | Val Loss : {val_loss:.6f}\n")
            print(f"\nEpoch : {e} | Step : {step} | Train Loss : {train_loss:.6f} | Val Loss : {val_loss:.6f}\n")
    return opt



def save_model(opt):
    model_path = src.config.SAVE_MODEL_PATH
     # Ensure parent directory exists
    dir_path = os.path.dirname(model_path)
             
    # Only rank 0 should create the directory and handle versioning
    if dist.get_rank() == 0:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    if os.path.exists(model_path):
        print("Model with same name exists. Saving with new version...")
        base,ext = os.path.splitext(model_path)
        version = 1
        while True:
            model_name = f"{base}_V{version}{ext}"
            if not os.path.exists(model_name):
                model_path = model_name
                break
            version += 1

    if dist.get_rank() == 0:
        torch.save({
            "model_state_dict" : model.state_dict(),
            "opt_state_dict" : opt.state_dict()
        },model_path)
        print(f"\n Model is saved to : {model_path}\n")
    # Wait for rank 0 to finish directory creation/versioning
    if dist.is_initialized():
        dist.barrier()

if __name__ == "__main__":
    ## Parameter
    setup() 
    set_seed(42 + dist.get_rank())
    SeedGPT = Transformer(vocab_size=src.config.VOCAB_SIZE,
                      emb_size=src.config.EMB_SIZE,
                      max_seq=src.config.CONTEXT_LENGTH,
                      num_block=src.config.N_BLOCK,
                      num_head=src.config.N_HEAD).to(device=src.config.DEVICE)

    model = DDP(SeedGPT,device_ids=[int(os.environ["LOCAL_RANK"])],
                output_device=int(os.environ["LOCAL_RANK"]),
                find_unused_parameters=True)
    
    optimizer = torch.optim.AdamW(model.parameters(),src.config.LEARNING_RATE)
    
    ## Optional
    PRETRAINED_MODEL = f"{src.config.SAVE_MODEL_PATH}"
    if os.path.exists(PRETRAINED_MODEL):
        map_loc = {f"cuda:0" : f"cuda:{int(os.environ['LOCAL_RANK'])}"}
        check_point = torch.load(PRETRAINED_MODEL,map_location=map_loc)
        model.load_state_dict(check_point["model_state_dict"])
        optimizer.load_state_dict(check_point["opt_state_dict"])
        print(f"\n[Rank {dist.get_rank()}] Loaded pre-trained weights from: {PRETRAINED_MODEL}\n")

    n_parameters = sum(p.numel() for p in SeedGPT.parameters())
    print(f"\nNumber of parameters need to be trained : {n_parameters/1e6:.2f} million")
    
    try:
        opt = train(model,optimizer)
    finally:
        save_model(opt=opt)
        cleanup()