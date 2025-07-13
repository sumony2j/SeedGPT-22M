# 🌱 SeedGPT-22M - Lightweight Transformer Language Models for Efficient Text Generation

![SeedGPT Architecture](./Architecture.webp)

**SeedGPT** is a family of compact language models designed from scratch using the PyTorch and Hugging Face ecosystem. With parameter counts around 22M, these models serve as an educational yet practical toolkit for building and deploying LLMs (Large Language Models) on resource-constrained hardware. It supports multiple fine-tuned variants and provides a ready-to-use Streamlit-based chat UI.

---

## 📚 Table of Contents

- [Overview](#overview)
- [LLM Architecture](#llm-architecture)
- [Model Variants](#model-variants)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Web UI Demo](#web-ui-demo)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Distributed Training](#distributed-training)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)
- [References](#references)

---

## 🧠 Overview

SeedGPT demonstrates how to build a language model from scratch and integrate it into a functional UI. Inspired by modern LLM principles and open-source models, SeedGPT supports:

- 🛠️ Custom transformer architecture
- 🎛️ Fine-tuning and chat-style generation
- ⚡ Real-time inference via GPU
- 🖥️ Minimal resource footprint (~22M parameters)

---

## 🏗️ LLM Architecture

SeedGPT uses a custom architecture implemented via:

- `HFTransformerConfig`: Configuration for the transformer.
- `HFTransformerModel`: Custom decoder-only Transformer model class.
- HuggingFace-compatible `AutoModelForCausalLM` integration.

**Architecture Overview** (as illustrated in `Architecture.webp`):

- Transformer decoder blocks
- Positional embeddings
- Causal self-attention mechanism
- LayerNorm & residuals
- Output linear projection for token generation

These components are registered with Hugging Face’s `transformers` via:

```python
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
CONFIG_MAPPING.register("hf_transformer", HFTransformerConfig)
MODEL_FOR_CAUSAL_LM_MAPPING.register(HFTransformerConfig, HFTransformerModel)
```

---

## 🧬 Model Variants

| Model         | Params | Dataset | Purpose | Repo |
|---------------|--------|---------|---------|------|
| **SeedGPT-V1** | 22M   | Refined BookCorpus | General text generation | [HF Link](https://huggingface.co/singhsumony2j/SeedGPT-V1) |
| **SeedGPT-V2** | 22M   | TinyStories + Stories Dataset | Story generation | [HF Link](https://huggingface.co/singhsumony2j/SeedGPT-V2) |
| **SeedGPT-V3** | 22M   | LMSYS Chat English | Chat-style conversations | [HF Link](https://huggingface.co/singhsumony2j/SeedGPT-V3) |

---

## ✨ Features

- ✅ Streamlit-powered chat UI with session memory
- ✅ Temperature & token-length sliders
- ✅ Multiple fine-tuned model options
- ✅ Hugging Face integration
- ✅ Supports GPU acceleration via CUDA
- ✅ Token template for structured prompt handling

---

## 📦 Installation

### Requirements

Ensure Python 3.13+ is installed.

### Install dependencies

```bash
git clone https://github.com/your-username/seedgpt
cd seedgpt
pip install -r requirments.txt
```

---

## 🚀 Usage

### Launch Web Chat UI

```bash
streamlit run web.py
```

Interact with the model using a chatbot interface where you can select the model, adjust generation temperature, and control maximum token length.

---

## 🖥️ Web UI Demo

The Streamlit app (`web.py`) offers:

- Sidebar controls:
  - 🔥 Temperature (0.2 to 1.0)
  - 🔠 Max Tokens (10 to 4096)
  - 🧠 Model selector
- Interactive message history
- Real-time model responses
- Clear chat option

All models are automatically loaded from Hugging Face based on your selection.

---

## ⚙️ Configuration

All model and tokenizer logic resides in the `web.py` file:

```python
tokenizer = AutoTokenizer.from_pretrained(f"singhsumony2j/{model_type}")
model = AutoModelForCausalLM.from_pretrained(f"singhsumony2j/{model_type}")
model.to(device)
```

Message formatting is handled using a Jinja-style chat template for consistent role separation.

---

## 📄 Dependencies

List from `requirments.txt`:

```txt
pandas
torch
datasets
transformers
tokenizers
tiktoken
modelscope
zarr==2.10
numcodecs
tqdm
streamlit
streamlit_chat
```

Install with:

```bash
pip install -r requirments.txt
```

---

## 🧪 Distributed Training

Use `run.sh` to launch multi-node distributed training:

```bash
bash run.sh
```

This script launches 6 processes per node across 3 nodes using `torchrun`:

```bash
torchrun --nproc_per_node=6 --nnodes=3 --node_rank=0 --master_addr="" --master_port=12345 -m src.main
```

> ⚠️ Make sure to fill in your `--master_addr` with the master node IP before running.

NCCL environment variables are pre-configured for optimal network training performance.

---

## 💬 Examples

Try prompts like:

- `"Tell me a bedtime story about a space dinosaur."`
- `"Explain quantum physics simply."`
- `"Simulate a conversation between a detective and a suspect."`

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA device not available` | Ensure compatible CUDA and GPU driver is installed |
| `Model not loading` | Check internet access for downloading from Hugging Face |
| `Streamlit crashes` | Use Python 3.8+, install all requirements |
| `Chat not generating response` | Increase max token or lower temperature |

---

## 👥 Contributors

- **Sumony Singh** – [Hugging Face Profile](https://huggingface.co/singhsumony2j)
- Thanks to open-source contributors behind Hugging Face, Streamlit, and PyTorch.

---

## 🪪 License

This project is licensed under the **MIT License**.

---

## 📚 References

- [Original Blog: Building a 2B Parameter LLM](https://levelup.gitconnected.com/building-a-2-billion-parameter-llm-from-scratch-using-python-1325cb05d6fb)
- [TinyStories Dataset](https://shorturl.at/F1ZvX)
- [LMSYS Chat English](https://shorturl.at/PZANz)
- [Refined BookCorpus](https://shorturl.at/FezgK)

---
