import os
import json
import torch
from transformers import AutoTokenizer
from src.transformer import Transformer

tokenizer = AutoTokenizer.from_pretrained("./SeedGPT_20M/SeedTokenizer")
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load("./SeedGPT-V3_Chat_finetuned/pytorch_model.bin", map_location=device)

config = checkpoint["config"]
model = Transformer(
    vocab_size=config["vocab_size"],
    emb_size=config["emb_size"],
    max_seq=config["max_seq"],
    num_head=config["num_head"],
    num_block=config["num_block"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

tokenizer.chat_template = """
{% for message in messages %}
{% if message["role"] == "user" %}
<S>user: {{ message["content"] }}</S>
{% elif message["role"] == "assistant" %}
<S>assistant: {{ message["content"] }}</S>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<S>assistant:
{% endif %}
"""

chat = [{"role": "user", "content": "Hi"}]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

# Generate response
with torch.no_grad():
    output = model.generate(inputs["input_ids"], max_tokens=10, temp=0.8)

# Decode
generated = output[0][inputs["input_ids"].shape[1]:]
print("🧠 Prompt:", chat[0]["content"])
print("🗣️ Assistant:", tokenizer.decode(generated, skip_special_tokens=True))
