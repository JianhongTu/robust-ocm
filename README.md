# Robust OCM

A toolkit for robust Optical Character Recognition (OCR) and model evaluation, designed to work with LongBench-v2 and similar document understanding benchmarks.

## Installation

### Using UV (Recommended)

```bash
# Install the package in development mode
uv pip install -e .
```

## Project Roadmap

### Dataset Preparation
- [x] Download LongBenchv2
- [x] Encode LongBenchv2 into images
- [x] Recognize word-level and block-level bounding boxes
- [x] Encode into a format compatible with OmniDocBench evaluation
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

### Converting to OmniDocBench Format

To convert the line_bbox.jsonl file to OmniDocBench format for evaluation:

```bash
# Convert all documents (page 0 only)
convert-to-omnidoc --input data/longbenchv2_img/line_bbox.jsonl --output data/longbenchv2_img/omnidoc_format.json

# Convert first 100 documents, page 1 only
convert-to-omnidoc --input data/longbenchv2_img/line_bbox.jsonl --output data/longbenchv2_img/omnidoc_format.json --limit 100 --page 1

# Convert with custom page selection
convert-to-omnidoc --input data/longbenchv2_img/line_bbox.jsonl --output data/longbenchv2_img/omnidoc_format.json --page 2
```

### Rendering Documents

To render LongBench-v2 documents to images:

```bash
# Process all samples with line-level bbox extraction
render --data-json ./data/longbenchv2/data.json

# Skip samples listed in blacklist.txt (recommended to avoid blocking samples)
render --blacklist ./blacklist.txt

# Process only 10 samples with word-level extraction
render --limit 10 --extraction-level word

# Use custom configuration
render --config ./config/config_en.json

# Resume interrupted processing
render --recover
```

### Visualization and Analysis

```bash
# Visualize results
viz --input data/longbenchv2_img/line_bbox.jsonl

# Analyze document statistics
analyze --input data/longbenchv2_img/line_bbox.jsonl
```
