import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer,AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from pt2hf import HFTransformerConfig, HFTransformerModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING

# Register your custom config + model
CONFIG_MAPPING.register("hf_transformer", HFTransformerConfig)
MODEL_MAPPING.register(HFTransformerConfig, HFTransformerModel)

#tokenizer = AutoTokenizer.from_pretrained("./SeedGPT_20M/SeedTokenizer")

class SFTDataset(Dataset):
    def __init__(self,tokenizer,context_len,input_file):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.conv = []
        with open(input_file,"r",encoding="utf-8") as f:
            for line in f:
                self.conv.append(json.loads(line)["conversations"])
    def __len__(self):
        return len(self.conv)
    def __getitem__(self, index):
        data = self.conv[index]
        tokens = []
        labels = [-100]
        for chance in data:
            role,content = chance["role"],chance["content"]
            prefix = self.tokenizer.encode(f"<S>{role}\n",add_special_tokens=False)
            content_ids = self.tokenizer.encode(content,add_special_tokens=False)
            suffix = self.tokenizer.encode("</S>\n",add_special_tokens=False)
            
            tokens.extend(prefix + content_ids + suffix)
            if role == "assistant":
                labels.extend([-100]*len(prefix))
                labels.extend(content_ids)
                labels.extend(suffix)
            else:
                labels.extend([-100]*(len(prefix)+len(content_ids)+len(suffix)))
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        tokens = tokens[:self.context_len+1] + [pad] * max(0,self.context_len+1 - len(tokens))
        labels = labels[:self.context_len+1] + [-100] * max(0,self.context_len+1 - len(labels))
        
        inputs = torch.tensor(tokens[:-1],dtype=torch.long)
        targets = torch.tensor(labels[1:],dtype=torch.long)    
        attention_mask = (inputs != pad).long()      
        
        return {
            "input_ids" : inputs,
            "labels" : targets,
            "attention_mask" : attention_mask
        }

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./SeedGPT_20M/SeedTokenizer")
model = AutoModel.from_pretrained("./SeedGPT20M")

# Optional: define ChatML-style template
tokenizer.chat_template = """
{{ bos_token }}
{% for message in messages %}
{% if message["role"] == "user" %}
User: {{ message["content"] }}
{% elif message["role"] == "assistant" %}
Assistant: {{ message["content"] }}
{% endif %}
{% endfor %}
{{ eos_token }}
"""

# Set pad token (important for batch training)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

# Load dataset
dataset = SFTDataset(
    tokenizer=tokenizer,
    context_len=64,
    input_file="./SeedGPT_20M/data/Conversation.jsonl"
)

# Training arguments
args = TrainingArguments(
    output_dir="./sft_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    learning_rate=2e-5,
    fp16=True,  # Use if your GPU supports it
    report_to="none",
    remove_unused_columns=False  # Important when using custom datasets
)

from transformers import Trainer

class NoAttentionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Strip `attention_mask` since your model doesn't support it
        inputs = {k: v for k, v in inputs.items() if k != "attention_mask"}
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

# Create Trainer
trainer = NoAttentionTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Start fine-tuning
trainer.train()

# Save model
model.save_pretrained("./sft_model")
tokenizer.save_pretrained("./sft_model")
        
