from transformers import AutoTokenizer
from transformers import AutoModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING
import torch
from pt2hf import HFTransformerConfig, HFTransformerModel  # Your custom config and model

# Register custom config and model
CONFIG_MAPPING.register("hf_transformer", HFTransformerConfig)
MODEL_MAPPING.register(HFTransformerConfig, HFTransformerModel)


## Test Seed Tokenizer
tokenizer = AutoTokenizer.from_pretrained("./SeedGPT_20M/SeedTokenizer")
model = AutoModel.from_pretrained("./SeedGPT20M")

print(tokenizer.pad_token_id)
print(tokenizer.decode(5))
"""
tokens = tokenizer("Sumon ")

input_data = torch.tensor(tokens.input_ids,dtype=torch.long)[None,:]

gen_text = model.generate(input_data,256,0.7)

output_txt = tokenizer.decode(gen_text[0].tolist(),skip_special_tokens=True)
output_txt = output_txt.replace("</S>", "").strip()

print(output_txt)
#print(tokenizer.decode(tokens.input_ids))
"""
