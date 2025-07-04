import torch
import os

## HyperParameter

EMB_SIZE = 512
CONTEXT_LENGTH = 256
N_HEAD = 12
N_BLOCK = 12
VOCAB_SIZE = 50304
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
N_EPCOHS = 200
N_EVAL_ITRS = 10

## Data Path
INPUT_DIR_PATH = "./"
TOKENIZED_TRAINING_DATA_PATH = {"/LocalData/deepseek/dataset/encoded_bookcorpus2_training_tokens.zarr",
                                "/LocalData/deepseek/dataset/encoded_subtitles_training_tokens.zarr",
                                "/LocalData/deepseek/dataset/encoded_openwebtext_training_tokens.zarr"
                                }

TOKENIZED_TEST_DATA_PATH = "./encoded_testing_tokens.zarr"
TOKENIZED_VAL_DATA_PATH = "./encoded_val_tokens.zarr"
SAVE_MODEL_PATH = "./model/SeedGPT50M.pt"

#Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"