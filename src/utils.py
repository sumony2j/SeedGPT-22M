import logging
import time
import numpy
import random
import torch
import os
import matplotlib.pyplot as plt

def set_seed(seed : int = 42):
    """Set seed for reproducibility."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_logger(log_dir="./log",log_file="./train.log"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)
    logger = logging.getLogger("SeedGPT_V2")
    logger.setLevel(logging.INFO)
    
    log_path = os.path.join(log_dir,log_file)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
    return logger


def plot_graph(file_path,title = "Epochs VS Loss"):
    epochs = []
    train_loss = []
    val_loss = []
    with open(file=file_path,encoding="utf-8",mode="r") as f:
        for line in f:
            if "Epoch" in line and "Train Loss" in line and "Val Loss" in line:
                parts = line.strip().split("|")
                epoch = int(parts[0].split(":")[1].strip())
                tr_loss = float(parts[2].split(":")[1].strip())
                vl_loss = float(parts[3].split(":")[1].strip())
                
                epochs.append(epoch)
                train_loss.append(tr_loss)
                val_loss.append(vl_loss)
    plt.figure(figsize=(10,5))
    plt.plot(epochs,train_loss,label="Training Loss",marker="o")
    plt.plot(epochs,val_loss,label="Validation Loss",marker="x")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
        
plot_graph("./SeedGPT-13M/src/log.txt")