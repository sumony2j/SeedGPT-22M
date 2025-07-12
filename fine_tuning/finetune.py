import os
import json
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer
from src.transformer import Transformer
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def remove_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict

class SFTDataset(Dataset):
    def __init__(self, tokenizer, context_len, input_file):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.conv = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                self.conv.append(json.loads(line))

    def __len__(self):
        return len(self.conv)

    def __getitem__(self, index):
        data = self.conv[index]
        tokens = []
        labels = []
        for chance in data:
            role, content = chance["role"], chance["content"]
            prefix = self.tokenizer(f"<S>{role}\n", add_special_tokens=False).input_ids
            content_ids = self.tokenizer(content, add_special_tokens=False).input_ids
            suffix = self.tokenizer("</S>\n", add_special_tokens=False).input_ids

            tokens.extend(prefix + content_ids + suffix)
            if role == "assistant":
                labels.extend([-100] * len(prefix))
                labels.extend(content_ids)
                labels.extend([-100]*len(suffix))
            else:
                labels.extend([-100] * (len(prefix) + len(content_ids) + len(suffix)))

        pad = self.tokenizer.pad_token_id
        tokens = tokens[:self.context_len + 1] + [pad] * max(0, self.context_len + 1 - len(tokens))
        labels = labels[:self.context_len + 1] + [-100] * max(0, self.context_len + 1 - len(labels))

        inputs = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(labels[1:], dtype=torch.long)

        return {
            "input_ids": inputs,
            "labels": targets
        }

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = setup()

    tokenizer = AutoTokenizer.from_pretrained("/LocalData/deepseek/dataset/LLM_Model/fine-tune/SeedTokenizer")
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Transformer(
        vocab_size=2000,
        emb_size=512,
        max_seq=256,
        num_head=4,
        num_block=6
    ).to(local_rank)


    map_loc = {f"cuda:0" : f"cuda:{int(os.environ['LOCAL_RANK'])}"}
    checkpoint = torch.load("./SeedGPT_20M/SeedGPT20M_Stories.pt", map_location=map_loc)
    model.load_state_dict(remove_module_prefix(checkpoint["model_state_dict"]))
    model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)

    dataset = SFTDataset(
        tokenizer=tokenizer,
        context_len=256,
        input_file="/LocalData/deepseek/dataset/LLM_Model/fine-tune/lmsys_chat.jsonl"
    )
    sampler = DistributedSampler(dataset=dataset, shuffle=True, num_replicas=world_size, rank=rank)
    sft_dataloader = DataLoader(dataset, batch_size=512, sampler=sampler)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    SFT_N_EPOCHS = 200
    model.train()
    print(f"[Rank {rank}] 🚀 Starting SFT for {SFT_N_EPOCHS} epochs...")

    for epoch in range(SFT_N_EPOCHS):
        sampler.set_epoch(epoch)
        total_sft_loss = 0
        sft_batch_count = 0
        loop = tqdm(sft_dataloader, desc=f"Epoch {epoch+1}", disable=rank != 0)

        for batch in loop:
            input_ids = batch["input_ids"].to(local_rank)
            labels = batch["labels"].to(local_rank)
                                                             
            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(input_ids, targets=labels)
            
            if loss is not None and isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()
                total_sft_loss += loss.item()

            sft_batch_count += 1
            if sft_batch_count % max(1, len(sft_dataloader)//1) == 0 and rank == 0:
                print(f"SFT Epoch {epoch+1}/{SFT_N_EPOCHS}, Batch {sft_batch_count}/{len(loop)}, Loss: {loss.item():.4f}")

        if rank == 0:
            avg_epoch_sft_loss = total_sft_loss / sft_batch_count if sft_batch_count > 0 else float("nan")
            print(f"✅ End of Epoch {epoch+1} | Average SFT Loss: {avg_epoch_sft_loss:.4f}")

    if rank == 0:
        save_path = "./SeedGPT-V3_Chat_finetuned"
        os.makedirs(save_path, exist_ok=True)
        torch.save({
            "model_state_dict": model.module.state_dict(),
            "config": {
                "vocab_size": 2000,
                "emb_size": 512,
                "max_seq": 256,
                "num_head": 4,
                "num_block": 6
            }
        }, os.path.join(save_path, "pytorch_model.bin"))
        tokenizer.save_pretrained(save_path)
        print("🎉 SFT training finished and model saved.")

    cleanup()

if __name__ == "__main__":
    main()
