#!/bin/bash

sudo apt install -y git git-lfs

git lfs install

## Dataset on which the CUSTOM Tokeinzer was trained
git clone https://www.modelscope.cn/datasets/OmniData/Pile-OpenWebText2.git 

## Datasets on which SeedGPT-V2 was trained
git clone https://huggingface.co/datasets/roneneldan/TinyStories 
git clone https://huggingface.co/datasets/delphi-suite/stories

## Dataset on which SeedGPT-V1 was trained
curl -L -o ./refined-bookcorpus-dataset.zip https://www.kaggle.com/api/v1/datasets/download/nishantsingh96/refined-bookcorpus-dataset
 
## Dataset on which SeedGPT-V2 was fine-tuned (SeedGPT-V3)
curl -L -o ~/Downloads/lmsys-chat-1m.zip https://www.kaggle.com/api/v1/datasets/download/gmhost/lmsys-chat-1m