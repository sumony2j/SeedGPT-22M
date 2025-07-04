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

model = AutoModel.from_pretrained("./sft_model")
tokenizer = AutoTokenizer.from_pretrained("./sft_model")

tokenizer.chat_template = """
{{ bos_token }}
{% for message in messages %}
{% if message['role'] == 'user' %}
User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}
{% endif %}
{% endfor %}
{{ eos_token }}
"""

messages = [
    {"role": "user", "content": "Hi ?"}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

output = model.generate(input_ids,max_tokens=10,temp=0.8)

response = tokenizer.decode(output[0].tolist(),skip_special_tokens=True)
print("Assistant:", response.strip())



