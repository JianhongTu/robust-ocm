# Robust OCM

A toolkit for robust Optical Character Recognition (OCR) and model evaluation, designed to work with LongBench-v2 and similar document understanding benchmarks.

## Installation

### Using UV (Recommended)

```bash
# Install the package in development mode
uv pip install -e .
```

## Project Structure

```
robust-ocm/
├── pyproject.toml          # Project configuration
├── README.md               # This file
├── blacklist.txt           # IDs of problematic samples (top 25% longest)
├── src/                    # Source code
│   └── robust_ocm/
│       ├── __init__.py
│       └── render/
│           ├── __init__.py
│           ├── render.py          # Main TextRenderer class
│           ├── pdf_generator.py   # PDF generation logic
│           ├── bbox_extractor.py  # Bounding box extraction
│           ├── image_processor.py # Image processing
│           ├── config.py          # Configuration management
│           ├── cli.py             # render CLI
│           ├── visualize.py       # viz CLI
│           ├── analyze.py         # analyze CLI
│           ├── convert_to_omnidoc.py # convert-to-omnidoc CLI
│           └── README.md          # Module documentation
├── scripts/                # Legacy scripts
│   ├── batch_inference.py  # Batch OCR inference with DeepSeek-OCR
│   └── word2png_function.py
├── config/                 # Configuration files
│   ├── config_en.json
│   └── config_zh.json
└── data/                   # Data directory
    └── longbenchv2/
        ├── data.json
        └── STRUCTURE.md
```

## Usage

### Generate the OCR splits:

```bash
# Render clean OCR data
conda run -n robust-ocm render --blacklist blacklist_short.txt --task ocr

# Generate adversarial DPI downscale split
conda run -n robust-ocm python -m robust_ocm.adv.adv_render --perturbation-type dpi_downscale --dpi 56 --blacklist ./blacklist_short.txt --task ocr

# Generate adversarial DPI upscale split
conda run -n robust-ocm python -m robust_ocm.adv.adv_render --perturbation-type dpi_downscale --dpi 192 --blacklist ./blacklist_short.txt --task ocr

# Generate adversarial dense text split
conda run -n robust-ocm python -m robust_ocm.adv.adv_render --perturbation-type dense_text --font-size 7 --blacklist ./blacklist_short.txt --task ocr

# Generate adversarial binarization split
conda run -n robust-ocm python -m robust_ocm.adv.adv_cli --perturbation-type binarization_thresholding --threshold 128 --input-dir ./data/ocr/longbenchv2_img/images/ --task ocr

# Generate adversarial lossy encoding split
conda run -n robust-ocm python -m robust_ocm.adv.adv_cli --perturbation-type lossy_encoding --format jpeg --quality 1 --input-dir ./data/ocr/longbenchv2_img/images/ --task ocr

# Upscale low DPI images
conda run -n robust-ocm python scripts/upscale_images.py --input-dir ./data/ocr/adv_dpi_56/images --output-dir ./data/ocr/adv_upscale/images --low-dpi 56 --high-dpi 96
```

### Generate NIAH (Needle in a Haystack) Dataset:

```bash
# Step 1: Generate NIAH dataset with secrets inserted between sentences
conda run -n robust-ocm python -m robust_ocm.adv.niah_cli \
  --input data/longbenchv2/data.json \
  --output data/niah/data.json \
  --blacklist blacklist_final.txt \
  --multiplier 5 \
  --seed 42

# Step 2: Render NIAH dataset to images
conda run -n robust-ocm render \
  --data-json data/niah/data.json \
  --blacklist blacklist_short.txt \
  --task niah

# Step 3: Validate the generated dataset
conda run -n robust-ocm python scripts/validate_niah.py data/niah/data.json
```

### Run NIAH (Needle in a Haystack) Inference:

```bash
# Start vLLM server first (example with Qwen3-VL-8B)
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 8 \
    --dtype auto \
    --trust-remote-code

# Run NIAH inference on all subdirectories (multiple splits)
micromamba run -n test python scripts/vqa/niah_inference.py \
    --input data/niah \
    --output results/qwen3 \
    --config config/vqa/qwen3-vl.json \
    --base_url http://localhost:8000/v1 \
    --max_workers 128

# This will automatically:
# - Scan all subdirectories in data/niah
# - Process each subdirectory with data.json and images/
# - Generate separate results: results/qwen3/{subdir}_niah_eval.json

# Run with different models
micromamba run -n test python scripts/vqa/niah_inference.py \
    --input data/niah \
    --output results/internvl \
    --config config/vqa/internvl-3.5.json \
    --base_url http://localhost:8000/v1 \
    --max_workers 128
```