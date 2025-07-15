from .transformer import Transformer
import torch
from tokenizers import Tokenizer
import argparse
import os

def remove_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove 'module.' prefix
        new_state_dict[name] = v
    return new_state_dict

def generate_text(model_path,tokenizer_file,input_text:str,max_len:int,device="cuda"):
    model = Transformer(vocab_size=2000,
                        emb_size=512, #212
                        max_seq=256, #64
                        num_head=4,
                        num_block=6).to(device=device)
    model_param = torch.load(model_path,map_location=torch.device(device=device))
    model.load_state_dict(remove_module_prefix(model_param["model_state_dict"]))
    
    enc = Tokenizer.from_file(tokenizer_file)
   
    input_token = enc.encode(input_text).ids
   

    input_data = torch.tensor(input_token,
                              dtype=torch.long)[None,:].to(device=device)

    with torch.no_grad():
        model.eval()
        output = model.generate(input_data,max_tokens=max_len,temp=0.8)
        output_txt = enc.decode(output[0].tolist(),skip_special_tokens=True)
        output_txt = output_txt.replace("</S>", "").strip()   
        return output_txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with SeedGPT model.")
    parser.add_argument("--tokenizer_file", type=str, help="Path to the tokenizer JSON file",
                        default="./saved_tokenizer/tokenizer.json")
    parser.add_argument("--model_path", type=str, help="Path to the trained model checkpoint (.pt)",
                        default="./models/SeedGPT-V2.pt")
    parser.add_argument("--prompt", type=str, help="Input prompt text to start generation",
                        default="The report is ")
    parser.add_argument("--max_len", type=int, default=3000, help="Number of tokens to generate")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run inference on")

    args = parser.parse_args()

    print("---- LLM Generated ----\n")
    generated_text = generate_text(
        model_path=args.model_path,
        tokenizer_file=args.tokenizer_file,
        input_text=args.prompt,
        max_len=args.max_len,
        device=args.device
    )
    print(generated_text)

