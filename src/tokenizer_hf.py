from transformers import PreTrainedTokenizerFast
import argparse
import os

# --------------------------
# Build Huggingface tokenizer 
# --------------------------

# Path where tokenizer.json is saved
tokenizer_dir = "./SeedGPT-13M/save_tokenizer"
save_tokenizer = "./SeedGPT-13M/SeedTokenizer"

hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_dir, "tokenizer.json"),
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
    bos_token="[S]",
    eos_token="[/S]"
)

if not os.path.exists(save_tokenizer):
    os.makedirs(save_tokenizer,exist_ok=True)

# Save in Hugging Face format
hf_tokenizer.save_pretrained(save_tokenizer)
