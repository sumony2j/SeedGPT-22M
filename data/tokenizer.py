from transformers import AutoTokenizer
from tokenizers import Tokenizer,models,pre_tokenizers,trainers,decoders,normalizers
import os
import json
import random
import argparse



def load_txt_from_jsonl(file_path):
    """_summary_

    Args:
        file_path (_type_): jsonl file
    Yields:
        _type_: texts which need to be tokenized
    """
    with open(file=file_path,mode="r",encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            try:
                yield json.loads(line)["text"]
            except json.JSONDecodeError as e:
                print(f"[Warning] Skipping bad line {i}: {e}")
                continue
            except KeyError:
                print(f"[Warning] 'text' key missing at line {i}")
                continue



tokenier = Tokenizer(model=models.BPE(unk_token="[UNK]"))
tokenier.normalizer = normalizers.Sequence([normalizers.NFC()])
tokenier.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

special_tokens = ["[UNK]","[S]","[/S]","[SEP]","[MASK]","[PAD]","[CLS]"]
trainer = trainers.BpeTrainer(vocab_size=2000,
                              show_progress=True,
                              special_tokens=special_tokens,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

## Load the texts

texts = load_txt_from_jsonl("./training_data.jsonl")
tokenier.train_from_iterator(texts,trainer=trainer)

## Decoder

tokenier.decoder = decoders.ByteLevel()

## Save the tokenizer
tokenizer_dir = "./save_tokenizer"
os.makedirs(tokenizer_dir,exist_ok=True)

tokenier.save(os.path.join(tokenizer_dir,"tokenizer.json"))
print("Tokenizer training completed and saved successfully.")