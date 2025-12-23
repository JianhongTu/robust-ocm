# Robust OCM

A toolkit for robust Optical Character Recognition (OCR) and model evaluation, designed to work with LongBench-v2 and similar document understanding benchmarks.

## Installation

### Using UV (Recommended)

```bash
# Install the package in development mode
uv pip install -e .

# Or install with development dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### Render LongBench-v2 Dataset to Images

```bash
# Process all samples
robust-ocm-render --config config/config_en.json

# Process limited number of samples for testing
robust-ocm-render --limit 10 --config config/config_en.json

# Resume interrupted processing
robust-ocm-render --recover --config config/config_en.json

# Custom paths
robust-ocm-render \
    --data-json data/longbenchv2/data.json \
    --config config/config_en.json \
    --output-dir data/longbenchv2_img/images \
    --output-jsonl data/longbenchv2_img/processed_output.jsonl \
    --limit 100
```

### Python API

```python
from word2png_function import text_to_images, batch_process_to_images

# Convert single text to images
images = text_to_images(
    text="Your document text here...",
    output_dir="./output",
    config_path="config/config_en.json",
    unique_id="my_document"
)

# Batch process dataset
batch_process_to_images(
    json_path="data/longbenchv2/data.json",
    output_dir="data/longbenchv2_img/images",
    output_jsonl_path="data/longbenchv2_img/processed_output.jsonl",
    config_path="config/config_en.json",
    limit=100
)
```

## Configuration

Create a configuration file (e.g., `config/config_en.json`) with the following settings:

```json
{
    "page-size": "595,842",
    "dpi": 72,
    "margin-x": 10,
    "margin-y": 10,
    "font-path": "/path/to/your/font.ttf",
    "font-size": 12,
    "line-height": 14,
    "font-color": "#000000",
    "alignment": "LEFT",
    "horizontal-scale": 1.0,
    "first-line-indent": 0,
    "left-indent": 0,
    "right-indent": 0,
    "space-after": 0,
    "space-before": 0,
    "border-width": 0,
    "border-padding": 0,
    "page-bg-color": "#FFFFFF",
    "para-bg-color": "#FFFFFF",
    "auto-crop-width": true,
    "auto-crop-last-page": true
}
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/your-org/robust-ocm.git
cd robust-ocm

# Install with uv
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black scripts/
isort scripts/

# Lint code
flake8 scripts/
```

## Project Roadmap

### Dataset Preparation
- [x] Download LongBenchv2
- [x] Encode LongBenchv2 into images
- [ ] Recognize word-level and block-level bounding boxes
- [ ] Encode into a format compatible with OmniDocBench evaluation
- [ ] Create corruption split on both block and word levels

### Inference Preparation
- [ ] Set up those models to process vision inputs
- [ ] Evaluate transcription accuracy with OmniDocBench
- [ ] Evaluate QA accuracy with (maybe) LM-Eval-Harness

### Experimentation
- [ ] Set up a pipeline for inference & get scores
- [ ] Report baseline OCR scores
- [ ] Report OCR scores on adversarial splits
- [ ] Report OCR scores on the corruption split
- [ ] Report baseline QA scores
- [ ] Report QA scores on the corruption split

### Additionals
- [ ] Text-based adversarial attack
- [ ] Image-based adversarial attacks
- [ ] Simple multi-agent baseline
- [ ] Use an OCR model as context selection
- [ ] Extract the necessary context
- [ ] Sends to a language model to extract the final answer

## Project Structure

```
robust-ocm/
├── pyproject.toml          # Project configuration
├── README.md               # This file
├── scripts/                # Source code
│   └── word2png_function.py
├── config/                 # Configuration files
│   ├── config_en.json
│   └── config_zh.json
└── data/                   # Data directory
    └── longbenchv2/
        ├── data.json
        └── STRUCTURE.md
```
