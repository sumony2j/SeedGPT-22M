from transformer import Transformer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR,ExponentialLR
import torch.distributed as dist
from data.loader import get_batched_data
import torch
import config
from tqdm import tqdm
import os
import numpy as np
import random
from transformers import get_cosine_schedule_with_warmup


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup():
    dist.init_process_group(backend="nccl")
    device = torch.device(f'cuda:{int(os.environ["LOCAL_RANK"])}')
    torch.cuda.set_device(device=device)

def cleanup():
    dist.destroy_process_group()


SeedGPT = Transformer(vocab_size=config.VOCAB_SIZE,
                      emb_size=config.EMB_SIZE,
                      max_seq=config.CONTEXT_LENGTH,
                      num_block=config.N_BLOCK,
                      num_head=config.N_HEAD).to(device=config.DEVICE)

model = DDP(SeedGPT,device_ids=[int(os.environ["LOCAL_RANK"])],
                output_device=int(os.environ["LOCAL_RANK"]),
                find_unused_parameters=True)

opt = torch.optim.AdamW(model.parameters(),config.LEARNING_RATE)
#scheduler = StepLR(opt,step_size=5000,gamma=0.9)
scheduler = CosineAnnealingLR(optimizer=opt,T_max=50000)
scheduler = get_cosine_schedule_with_warmup(optimizer=opt,
                                            num_warmup_steps=500,
                                            num_training_steps=1000)
    
@torch.no_grad()
def est_loss(steps):
    model.eval()
    out = {}
    for data_type in ["train","val"]:
        losses = []
        data_file = config.TOKENIZED_TRAINING_DATA_PATH if data_type=="train" else config.TOKENIZED_VAL_DATA_PATH
        batch_iter = iter(get_batched_data(data_file,config.BATCH_SIZE,config.CONTEXT_LENGTH,
                                        config.DEVICE)) 
        for _ in range(steps):
            xb,xy = next(batch_iter)
            _,loss = model(xb,xy)
            losses.append(loss.item())
        out[data_type] = sum(losses)/len(losses)
    model.train()
    return out

"""
def train():
    model.train()
    dataset = get_batched_data(config.TOKENIZED_TRAINING_DATA_PATH,
                                config.BATCH_SIZE,
                                config.CONTEXT_LENGTH,
                                config.DEVICE)
    for e in tqdm(range(config.N_EPCOHS)):
        if e % config.N_EVAL_ITRS == 0:
            train_loss,val_loss = est_loss(config.N_EVAL_ITRS).values()
            print(f"Iteration - {e} , Train Loss : {train_loss:.6f} , Val Loss : {val_loss:.6f}")
        xb,yb = next(dataset)
        _,loss = model(xb,yb)
        opt.zero_grad()
        loss = loss.mean()
        loss.backward()
        opt.step()
        scheduler.step()
"""
def train(model):
    opt = torch.optim.AdamW(model.parameters(),config.LEARNING_RATE)
    scheduler = ExponentialLR(optimizer=opt,gamma=0.95)
    scheduler = get_cosine_schedule_with_warmup(optimizer=opt,num_warmup_steps=500,num_training_steps=50000)
    model.train()
    for e in tqdm(range(config.N_EPCOHS)):
        dataset = get_batched_data(config.TOKENIZED_TRAINING_DATA_PATH,
                config.BATCH_SIZE,
                config.CONTEXT_LENGTH,
                config.DEVICE)
        for _ ,(xb,yb) in enumerate(dataset):
            if e % config.N_EVAL_ITRS == 0:
                train_loss,val_loss = est_loss(model,config.N_EVAL_ITRS).values()
                print(f"Epoch - {e} , Train Loss : {train_loss:.6f} , Val Loss : {val_loss:.6f}")
            #xb,yb = next(dataset)
            _,loss = model(xb,yb)
            opt.zero_grad()
            loss = loss.mean()
            loss.backward()
            opt.step()
            scheduler.step()


def save_model():
    model_path = config.SAVE_MODEL_PATH
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
    
    # Wait for rank 0 to finish directory creation/versioning
    if dist.is_initialized():
        dist.barrier()
    if dist.get_rank() == 0:
        torch.save({
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : opt.state_dict()
            },model_path)
        print(f"Model is saved to : {model_path}")

if __name__ == "__main__":
    ## Parameter
    setup() 
    set_seed(42 + dist.get_rank())
    
    n_parameters = sum(p.numel() for p in SeedGPT.parameters())
    print(f"Number of parameters need to be trained : {n_parameters/1e6:.2f} million")
    
    try:
        train()
    finally:
        save_model()
        cleanup()
                
            
                
            
            
    