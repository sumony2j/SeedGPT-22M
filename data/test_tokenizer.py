import os
from tokenizers import Tokenizer

tokenizer_dir = "./SeedGPT-13M/save_tokenizer"
tokenizer_file = os.path.join(tokenizer_dir,"tokenizer.json")

load_tokenizer = Tokenizer.from_file(tokenizer_file)

encoded = load_tokenizer.encode("Hello World")

# Print tokens
print("Tokens are :",encoded.tokens)

token_ids = encoded.ids

# Print unique token IDs
unique_token_ids = list(set(token_ids))
unique_token_ids = token_ids
print("Unique token IDs:", unique_token_ids)

# Print decode from raw token IDs
decoded_text2 = load_tokenizer.decode(encoded.ids)
print("Decoded from token IDs:", decoded_text2)
