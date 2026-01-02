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

conda run -n robust-ocm render --blacklist blacklist_short.txt --task ocr

conda run -n robust-ocm python -m robust_ocm.adv.adv_render --perturbation-type dpi_downscale --dpi 48 --blacklist ./blacklist_short.txt --task ocr

conda run -n robust-ocm python -m robust_ocm.adv.adv_render --perturbation-type dense_text --font-size 7 --blacklist ./blacklist_short.txt --task ocr

conda run -n robust-ocm python -m robust_ocm.adv.adv_cli --perturbation-type binarization_thresholding --threshold 128 --input-dir ./data/ocr/longbenchv2_img/images/ --task ocr

conda run -n robust-ocm python -m robust_ocm.adv.adv_cli --perturbation-type lossy_encoding --format jpeg --quality 1 --input-dir ./data/ocr/longbenchv2_img/images/ --task ocr

conda run -n robust-ocm python scripts/upscale_images.py --input-dir ./data/ocr/adv_dpi_48/images --output-dir ./data/ocr/adv_upscale/images --low-dpi 48 --high-dpi 200

```