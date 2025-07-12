#!/bin/bash

sudo apt install -y git git-lfs

git lfs install

git clone https://huggingface.co/datasets/roneneldan/TinyStories

git clone https://huggingface.co/datasets/delphi-suite/stories

curl -L -o ./refined-bookcorpus-dataset.zip https://www.kaggle.com/api/v1/datasets/download/nishantsingh96/refined-bookcorpus-dataset
 
curl -L -o ~/Downloads/lmsys-chat-1m.zip https://www.kaggle.com/api/v1/datasets/download/gmhost/lmsys-chat-1m