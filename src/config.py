import torch
import os

## HyperParameter

EMB_SIZE = 512
CONTEXT_LENGTH = 256
N_HEAD = 4
N_BLOCK = 6
VOCAB_SIZE = 2000
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
N_EPCOHS = 150
N_EVAL_ITRS = 1

## Data Path
INPUT_DIR_PATH = "./"
TOKENIZED_TRAINING_DATA_PATH="/LocalData/deepseek/dataset/cleaned_bookcorpus/encoded_training_data.zarr"
TOKENIZED_TEST_DATA_PATH = "/LocalData/deepseek/dataset/cleaned_bookcorpus/encoded_val_data.zarr/"
TOKENIZED_VAL_DATA_PATH = "/LocalData/deepseek/dataset/cleaned_bookcorpus/encoded_val_data.zarr/"
SAVE_MODEL_PATH = "/LocalData/deepseek/dataset/model/SeedGPT20M_Bookcorpus.pt"
PRESAVED_MODEL_PATH = "/LocalData/deepseek/dataset/model/SeedGPT20M_Bookcorpus_Base.pt"

#Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"