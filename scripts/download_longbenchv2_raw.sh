#!/bin/bash

# Create the data directory and subfolder for LongBench-v2
mkdir -p data/longbenchv2

# Download the data.json file from Hugging Face, following redirects
curl -L -o data/longbenchv2/data.json https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/main/data.json

echo "Download completed. File saved to data/longbenchv2/data.json"
