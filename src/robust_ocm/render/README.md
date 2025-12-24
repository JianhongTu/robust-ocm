# Robust OCM Render Module

A modular Python package for converting text to images with bounding box extraction, designed for optical compression model benchmarking.

## Overview

The render module provides three main functionalities:
1. **Text Rendering** - Convert text contexts to PNG images with configurable styling
2. **Bounding Box Extraction** - Extract precise bounding boxes at word or line level
3. **Analysis & Visualization** - Analyze output statistics and create visual overlays

## Installation

```bash
# Install from source (development mode)
cd /path/to/robust_ocm
pip install -e .

# Or install from PyPI (when published)
pip install robust-ocm
```

## Quick Start

### 1. Render Text to Images

```bash
# Basic usage - render all samples with line-level bboxes
render

# Render with specific configuration
render --config config/config_en.json --extraction-level line

# Limit to N samples
render --limit 10

# Resume interrupted processing
render --recover
```

### 2. Visualize Bounding Boxes

```bash
# Visualize first 5 samples, first page only
viz --input data/longbenchv2_img/line_bbox.jsonl --limit 5 --pages 1

# Visualize specific sample
viz --input data/longbenchv2_img/line_bbox.jsonl --sample-id 66fcffd9bb02136c067c94c5

# Custom output directory (default: data/longbenchv2_img/viz)
viz --input data/longbenchv2_img/line_bbox.jsonl --output-dir ./my_visualizations
```

### 3. Analyze Output

```bash
# Basic analysis
analyze --input data/longbenchv2_img/line_bbox.jsonl

# Detailed statistics per sample
analyze --input data/longbenchv2_img/line_bbox.jsonl --detailed

# Export statistics to JSON
analyze --input data/longbenchv2_img/line_bbox.jsonl --export-stats stats.json
```

## Command Reference

### robust-render

Main command for converting text to images with bounding box extraction.

**Arguments:**
- `--data-json`: Path to LongBench-v2 data.json file (default: `./data/longbenchv2/data.json`)
- `--config`: Path to configuration file (default: `./config/config_en.json`)
- `--output-dir`: Directory to save images (default: `./data/longbenchv2_img/images`)
- `--output-jsonl`: Path to save processed output (default: `./data/longbenchv2_img/line_bbox.jsonl`)
- `--limit N`: Process only N samples
- `--extraction-level`: `word` or `line` (default: `line`)
- `--processes N`: Number of parallel processes (default: 8)
- `--recover`: Resume interrupted processing

**Examples:**
```bash
# Render 10 samples with word-level bboxes
render --limit 10 --extraction-level word

# Use custom paths
render --data-json ./my_data.json --output-dir ./my_images --config ./my_config.json
```

### robust-viz

Create visual overlays of bounding boxes on generated images.

**Arguments:**
- `--input`: Path to line_bbox.jsonl file (required)
- `--image-dir`: Directory containing images (auto-detected if not provided)
- `--output-dir`: Directory to save visualizations (default: `./data/longbenchv2_img/viz`)
- `--limit N`: Process N samples only
- `--sample-id`: Process specific sample ID
- `--pages N`: Limit to first N pages of each sample
- `--overwrite`: Overwrite existing visualizations

**Examples:**
```bash
# Visualize all samples
viz --input data/longbenchv2_img/line_bbox.jsonl

# Quick check of first sample
viz --input data/longbenchv2_img/line_bbox.jsonl --limit 1 --pages 1
```

### robust-analyze

Analyze rendered output and provide statistics.

**Arguments:**
- `--input`: Path to processed_output.jsonl file (required)
- `--sample-id`: Analyze specific sample ID
- `--detailed`: Show per-sample statistics
- `--export-stats`: Export statistics to JSON file

**Examples:**
```bash
# Full analysis with details
robust-analyze --input processed_output.jsonl --detailed

# Export for further processing
robust-analyze --input processed_output.jsonl --export-stats analysis_results.json
```

## Configuration

The render module uses JSON configuration files. Key parameters:

```json
{
    "page-size": "595,842",
    "dpi": 72,
    "margin-x": 10,
    "margin-y": 10,
    "font-path": "./config/Verdana.ttf",
    "font-size": 9,
    "line-height": 10,
    "font-color": "#000000",
    "alignment": "LEFT",
    "horizontal-scale": 1.0,
    "auto-crop-width": true,
    "auto-crop-last-page": true
}
```

### Configuration Fields

- **Page Settings**:
  - `page-size`: Page dimensions in points (width,height)
  - `dpi`: Resolution for rendering
  - `margin-x`, `margin-y`: Page margins

- **Font Settings**:
  - `font-path`: Path to font file (required)
  - `font-size`: Font size in points
  - `line-height`: Line spacing
  - `font-color`: Text color (hex)

- **Layout**:
  - `alignment`: Text alignment (LEFT/CENTER/RIGHT/JUSTIFY)
  - `horizontal-scale`: Horizontal scaling factor
  - `auto-crop-width`: Crop empty space on right
  - `auto-crop-last-page`: Crop empty space on last page
  - `newline-markup`: HTML markup for newlines (default: "<br/>")
  - `remove-line-breaks`: Remove line breaks before rendering (default: false)

## Output Format

### JSONL Output Structure

Each line in the output JSONL contains only essential information:
```json
{
  "unique_id": "66fcffd9bb02136c067c94c5",
  "image_paths": [
    "/path/to/d0_66fcffd9_page_001.png",
    "/path/to/d0_66fcffd9_page_002.png",
    "..."
  ],
  "bboxes": [
    [
      [x0, y0, x1, y1, "text line 1"],
      [x0, y0, x1, y1, "text line 2"],
      "..."
    ],
    [
      [x0, y0, x1, y1, "text line 1 on page 2"],
      "..."
    ]
  ]
}
```

**Note**: Only essential fields (`unique_id`, `image_paths`, `bboxes`) are saved to reduce file size. Original metadata like domain, difficulty, etc., is not included in the output.

### Bounding Box Format

Each bounding box is represented as `[x0, y0, x1, y1, text]`:
- `x0, y0`: Top-left corner coordinates (pixels)
- `x1, y1`: Bottom-right corner coordinates (pixels)
- `text`: The text content within the box

## Python API

You can also use the module directly in Python:

```python
from robust_ocm.render import TextRenderer

# Initialize renderer
renderer = TextRenderer(config_path="config/config_en.json")

# Render text to images with bboxes
result = renderer.render_text(
    text="Hello World!\nThis is a test.",
    output_dir="./output",
    extraction_level="line"
)

# Access results
print(f"Generated {len(result['image_paths'])} images")
print(f"Found {len(result['bboxes'])} pages of bboxes")
```

## Module Structure

```
src/robust_ocm/render/
├── __init__.py          # Package initialization
├── render.py            # Main TextRenderer class
├── pdf_generator.py     # PDF generation logic
├── bbox_extractor.py    # Bounding box extraction
├── image_processor.py   # Image processing
├── config.py           # Configuration management
├── cli.py              # robust-render CLI
├── visualize.py        # robust-viz CLI
├── analyze.py          # robust-analyze CLI
└── README.md           # This file
```

## Performance Tips

1. **Parallel Processing**: Use multiple processes for large datasets
   ```bash
   robust-render --processes 16
   ```

2. **Extraction Level**: 
   - Use `line` for faster processing and fewer bboxes
   - Use `word` for fine-grained analysis (more bboxes)

3. **Memory Management**: Process in batches for very large datasets
   ```bash
   robust-render --batch-size 100
   ```

## Troubleshooting

### Font Not Found
```
ValueError: Font file not found: ../config/Verdana.ttf
```
Solution: Use absolute paths or ensure the font path is relative to the working directory.

### Memory Issues
For large datasets, reduce the number of parallel processes:
```bash
robust-render --processes 4
```

### Empty Images
Check that the input text is not empty and the configuration is valid.

## Perturbation Types

The render module supports various perturbation types for robustness testing and evaluation of optical compression models. These perturbations can be applied to test model performance under different degradation conditions.

### Text-based Perturbations

- **Font Weight** - Vary the thickness of characters (light, normal, bold, black)
- **Kerning Collisions** - Adjust spacing between character pairs to create visual artifacts
- **Homoglyph Substitution** - Replace characters with visually similar ones (e.g., 'rn' → 'm', '0' → 'O')
- **Line-Height Compression** - Reduce vertical spacing between lines to create crowding effects

### Image-based Perturbations

- **JPEG Compression** - Apply JPEG compression with varying quality levels to simulate lossy compression artifacts
- **Binarization Thresholding** - Convert grayscale images to binary with different threshold values
- **Random Noise** - Add Gaussian, salt-and-pepper, or speckle noise to images
- **Blur** - Apply Gaussian, motion, or defocus blur with varying intensities
- **Pixelation** - Downsample and upsample images to create blocky artifacts

These perturbations can be combined and parameterized to create comprehensive test suites for evaluating model robustness across different degradation scenarios.

## Examples

### Complete Workflow
```bash
# 1. Render dataset
render --config config/config_en.json --limit 100

# 2. Quick visualization check
viz --input data/longbenchv2_img/line_bbox.jsonl --limit 3 --pages 1

# 3. Analyze results
analyze --input data/longbenchv2_img/line_bbox.jsonl --detailed

# 4. Export statistics for reporting
analyze --input data/longbenchv2_img/line_bbox.jsonl --export-stats report.json
```

### Custom Configuration
```bash
# Create high-resolution output
render --config config/high_res.json --extraction-level word

# Visualize with custom output
viz --input data/longbenchv2_img/line_bbox.jsonl --output-dir ./custom_viz --overwrite
```

## Dependencies

- PyMuPDF - PDF text extraction
- ReportLab - PDF generation
- pdf2image - PDF to image conversion
- Pillow - Image processing
- NumPy - Numerical operations
- tqdm - Progress bars

## License

MIT License - see LICENSE file for details.