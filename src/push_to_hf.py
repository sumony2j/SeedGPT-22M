from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from .convert_to_hf_model import HFTransformerConfig, HFTransformerModel  # Your custom config and model
from huggingface_hub import notebook_login

# Register custom config and model
CONFIG_MAPPING.register("hf_transformer", HFTransformerConfig)
MODEL_FOR_CAUSAL_LM_MAPPING.register(HFTransformerConfig, HFTransformerModel)


## Test SeedGPT
tokenizer = AutoTokenizer.from_pretrained("../SeedGPT_HuggingFace/SeedTokenizer")
model = AutoModelForCausalLM.from_pretrained("../SeedGPT_HuggingFace/SeedGPT-V3")

repo_name = "singhsumony2j/SeedGPT-V3"

model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)