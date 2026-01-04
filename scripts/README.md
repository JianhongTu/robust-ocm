

# Scripts

This directory contains utility scripts for the robust-ocm project.

## Dataset Generation

### Needle in a Haystack (NIAH) Dataset Generation

The `generate_niah.sh` script generates synthetic NIAH datasets by inserting secret key-value pairs into documents and creating retrieval questions.

```bash
# Basic usage with defaults
./scripts/generate_niah.sh

# Custom configuration
./scripts/generate_niah.sh \
  --input data/longbenchv2/data.json \
  --output data/niah/data.json \
  --blacklist blacklist.txt \
  --multiplier 2 \
  --seed 42

# View help
./scripts/generate_niah.sh --help
```

See [src/robust_ocm/adv/README.md](../src/robust_ocm/adv/README.md) for more details on the NIAH CLI tool.

## Environment Preparation

### Environment Preparation

#### Deepseek OCR
```bash
# Install the latest version of vLLM
uv pip install -U vllm
```

#### Qwen3-VL
```bash
pip install accelerate
pip install qwen-vl-utils==0.0.14
# Install the latest version of vLLM 'vllm>=0.11.0'
uv pip install -U vllm
```