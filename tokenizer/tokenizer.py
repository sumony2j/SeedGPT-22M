from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, normalizers
import os
import json
import argparse

def load_txt_from_jsonl(file_path):
    """Generator that yields 'text' fields from a JSONL file."""
    with open(file=file_path, mode="r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            try:
                yield json.loads(line)["text"]
            except json.JSONDecodeError as e:
                print(f"[Warning] Skipping bad line {i}: {e}")
                continue
            except KeyError:
                print(f"[Warning] 'text' key missing at line {i}")
                continue

def train_tokenizer(data_file, vocab_size, output_dir):
    """Train a BPE tokenizer and save it."""
    tokenizer = Tokenizer(model=models.BPE(unk_token="[UNK]"))

    # Normalizer: NFC to handle Unicode characters
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])

    # Pre-tokenizer: ByteLevel
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens = ["[UNK]", "[S]", "[/S]", "[SEP]", "[MASK]", "[PAD]", "[CLS]"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # Load texts from JSONL file
    texts = load_txt_from_jsonl(data_file)

    # Train
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Add decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    print(f"✅ Tokenizer training completed. Saved to: {output_dir}/tokenizer.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from JSONL data.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to JSONL file with 'text' field.")
    parser.add_argument("--vocab_size", type=int, default=2000, help="Vocabulary size.")
    parser.add_argument("--output_dir", type=str, default="./save_tokenizer", help="Directory to save the tokenizer.")

    args = parser.parse_args()

    train_tokenizer(args.data_file, args.vocab_size, args.output_dir)
