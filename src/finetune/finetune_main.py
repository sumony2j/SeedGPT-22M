from transformers import AutoTokenizer
from trl import SFTTrainer
from transformers import TrainingArguments, Trainer
import os
from transformers import AutoModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING
import torch
import sys
import os

from pt2hf import HFTransformerConfig, HFTransformerModel  # Your custom config and model
from datasets import Dataset
import json

def load_conversations(path):
    messages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "conversations" in item:
                messages.append({"messages": item["conversations"]})
    return messages

data = load_conversations("./SeedGPT_20M/data/Conversation.jsonl")  # Path to your file

# Convert to Hugging Face dataset
train_ds = Dataset.from_list(data)


# Register custom config and model
CONFIG_MAPPING.register("hf_transformer", HFTransformerConfig)
MODEL_MAPPING.register(HFTransformerConfig, HFTransformerModel)


## Test Seed Tokenizer
tokenizer = AutoTokenizer.from_pretrained("./SeedGPT_20M/SeedTokenizer")
model = AutoModel.from_pretrained("./SeedGPT20M")



chat_template = """
{{ bos_token }}
{% for message in messages %}
{% if message["role"]=="user"%}
User : {{message["content"]}}
{% elif message["role"]=="assistant"%}
Assistant : {{message["content"]}}
{% endif %}
{% endfor %}
Assistant :
{{ eos_token }}
"""

tokenizer.chat_template = chat_template

"""
messages = [
    {"role": "user", "content": "What's 2 + 2?"},
    {"role": "assistant", "content": "4"}
]

encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
"""
args = TrainingArguments(output_dir="sft_model",
                         per_device_train_batch_size=2,
                         num_train_epochs=3,
                         save_steps=100)
trainer = Trainer(model=model,
                     args=args,
                     tokenizer=tokenizer,
                     train_dataset=train_ds)

trainer.train()

