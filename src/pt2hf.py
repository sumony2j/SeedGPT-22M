# File: convert_to_hf.py

from transformers import PreTrainedModel, PretrainedConfig
from transformers import PreTrainedTokenizerFast
import torch
import torch.nn as nn
from transformer import Transformer  # Your existing Transformer class
import os

# --------------------------
# Custom HuggingFace Config
# --------------------------
class HFTransformerConfig(PretrainedConfig):
    model_type = "hf_transformer"

    def __init__(self, vocab_size=30522, emb_size=512, max_seq=128, num_head=8, num_block=6, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.max_seq = max_seq
        self.num_head = num_head
        self.num_block = num_block


# ----------------------------------
# Wrap the Transformer for HF usage
# ----------------------------------
class HFTransformerModel(PreTrainedModel):
    config_class = HFTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Transformer(
            vocab_size=config.vocab_size,
            emb_size=config.emb_size,
            max_seq=config.max_seq,
            num_head=config.num_head,
            num_block=config.num_block,
        )

    def forward(self, input_ids, labels=None):
        logits, loss = self.model(input_ids, targets=labels)
        return {
            "logits": logits,
            "loss": loss,
        }
    def generate(self,input_ids,max_tokens,temp=1):
        out = self.model.generate(input_ids,max_tokens=max_tokens,temp=temp)
        return out

def remove_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove 'module.' prefix
        new_state_dict[name] = v
    return new_state_dict


# --------------------------------
# Load from .pt and save as HF
# --------------------------------
def convert_pt_to_hf(pt_path, save_dir, config_kwargs):
    config = HFTransformerConfig(**config_kwargs)
    model = HFTransformerModel(config)

    # Load only model weights from the DDP checkpoint
    checkpoint = torch.load(pt_path, map_location="cpu")
    #model.model.load_state_dict(checkpoint["model_state_dict"])
    #model.load_state_dict(remove_module_prefix(checkpoint["model_state_dict"]))
    model.model.load_state_dict(remove_module_prefix(checkpoint["model_state_dict"]))

    # Save as HuggingFace model
    model.save_pretrained(save_dir)
    config.save_pretrained(save_dir)
    print(f"Hugging Face-compatible model saved to: {save_dir}")



# ----------------------
# Example Usage
# ----------------------
if __name__ == "__main__":
    pt_model_path = "./SeedGPT-13M/SeedGPT20M_Stories_Final.pt"  # Replace with actual path
    save_directory = "./SeedGPT20M"

    config_params = {
        "vocab_size": 2000,
        "emb_size": 512,
        "max_seq": 256,
        "num_head": 4,
        "num_block": 6,
    }

    convert_pt_to_hf(pt_model_path, save_directory, config_params)
