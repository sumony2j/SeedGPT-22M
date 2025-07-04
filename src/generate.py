from transformer import Transformer
import torch
from tokenizers import Tokenizer
import tiktoken
import argparse
import os
import re
import string
from transformers import PreTrainedTokenizerFast
import language_tool_python

import re
import string

def post_process_llm_output(text: str) -> str:
    # 1. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Replace improper single quotes with double quotes (for speech)
    text = re.sub(r"(?<=\s)'(?=\w)", '"', text)
    text = re.sub(r"'(?=\s|[.,!?])", '"', text)

    # 3. Remove duplicated words (e.g., "nodded, nodded")
    text = re.sub(r'\b(\w+)[, ]+\1\b', r'\1', text, flags=re.IGNORECASE)

    # 4. Add space after punctuation if missing
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)

    # 5. Capitalize first letter of each sentence
    def capitalize_sentences(t):
        sentences = re.split('([.!?] *)', t)
        return ''.join([s.capitalize() if i % 2 == 0 else s for i, s in enumerate(sentences)])

    text = capitalize_sentences(text)

    # 6. Fix unbalanced quotes (basic version)
    if text.count('"') % 2 != 0:
        text += '"'

    # 7. Remove space before punctuation
    text = re.sub(r'\s([?.!",])', r'\1', text)

    return text

def clean_with_language_tool(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)


tokenizer_dir = "./SeedGPT_20M/save_tokenizer"
tokenizer_file = os.path.join(tokenizer_dir,"tokenizer.json")

def remove_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove 'module.' prefix
        new_state_dict[name] = v
    return new_state_dict

def generate_text(model_path,input_text:str,max_len:int,device="cuda"):
    model = Transformer(vocab_size=2000,
                        emb_size=512, #212
                        max_seq=256, #64
                        num_head=4,
                        num_block=6).to(device=device)
    model_param = torch.load(model_path,map_location=torch.device(device=device))
    model.load_state_dict(remove_module_prefix(model_param["model_state_dict"]))
    
    #enc = tiktoken.get_encoding('r50k_base')
    enc = Tokenizer.from_file(tokenizer_file)
    #enc = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    #print(enc.special_tokens_map)

    
    #input_token = enc.encode(input_text,allowed_special={"<|endoftext|>"})
    input_token = enc.encode(input_text).ids
    #input_token = enc.encode(input_text)

    input_data = torch.tensor(input_token,
                              dtype=torch.long)[None,:].to(device=device)

    with torch.no_grad():
        model.eval()
        output = model.generate(input_data,max_tokens=max_len,temp=0.8)
        output_txt = enc.decode(output[0].tolist(),skip_special_tokens=True)
        output_txt = output_txt.replace("</S>", "").strip()
        #output = post_process_llm_output(output_txt)    
        return output_txt

#model_path = "./SeedGPT_20M/SeedGPT20M_Stories_Final.pt"
model_path = "./SeedGPT_20M/SeedGPT20M_Bookcorpus.pt"

txt = "The "
print("----LLM Generated----\n")
gen_txt = generate_text(model_path,txt,512,"cpu")
print(gen_txt)
#print("\n----Modified----\n")
#print(clean_with_language_tool(gen_txt))
