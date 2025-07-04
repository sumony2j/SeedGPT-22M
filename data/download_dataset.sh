#!/bin/bash

sudo apt install -y git git-lfs

git lfs install

git clone https://www.modelscope.cn/datasets/OmniData/Pile-OpenWebText2.git

git clone https://www.modelscope.cn/datasets/OmniData/Pile-BookCorpus2.git

git clone https://www.modelscope.cn/datasets/OmniData/Pile-OpenSubtitles.git

git clone https://www.modelscope.cn/datasets/prithivMLmods/OpenWeb383K

git clone https://www.modelscope.cn/datasets/PrimeIntellect/c4-tiny

git clone https://www.modelscope.cn/datasets/AI-ModelScope/TinyStories

git clone https://huggingface.co/datasets/iohadrubin/wikitext-103-raw-v1

git clone https://huggingface.co/datasets/wufuheng/wikitext-103-v1-5p

git clone https://huggingface.co/datasets/roneneldan/TinyStories (This)

git clone https://huggingface.co/datasets/vblagoje/cc_news

git clone https://huggingface.co/datasets/jxie/bookcorpus

git clone https://huggingface.co/datasets/delphi-suite/stories

curl -L -o ./refined-bookcorpus-dataset.zip https://www.kaggle.com/api/v1/datasets/download/nishantsingh96/refined-bookcorpus-dataset



