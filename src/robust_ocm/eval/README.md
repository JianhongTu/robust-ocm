# OCR Evaluation Module

This module provides tools for evaluating OCR predictions against ground truth using metrics from the OmniDocBench benchmark.

## Features

- **Character Error Rate (CER)**: Measures the percentage of character-level errors
- **BLEU Score**: Measures n-gram overlap between prediction and ground truth
- **Text Normalization**: Automatic text preprocessing and normalization
- **Simple Layout Support**: Direct filename matching for simple layouts

## Installation

The evaluation module requires the following dependencies:

```bash
pip install Levenshtein sacrebleu sacremoses beautifulsoup4 pylatexenc
```

## Usage

### Basic Usage

Evaluate predictions against ground truth:

```bash
python -m robust_ocm.eval \
    --gt data/longbenchv2_img/OmniDocBench_concatenated.json \
    --pred data/pred/dpsk
```

### Save Results

Save evaluation results to a JSON file:

```bash
python -m robust_ocm.eval \
    --gt data/longbenchv2_img/OmniDocBench_concatenated.json \
    --pred data/pred/dpsk \
    --output results/dpsk_evaluation.json
```

### Disable Normalization

Evaluate without text normalization:

```bash
python -m robust_ocm.eval \
    --gt data/longbenchv2_img/OmniDocBench_concatenated.json \
    --pred data/pred/dpsk \
    --no-normalize
```

### Verbose Output

Enable verbose logging for debugging:

```bash
python -m robust_ocm.eval \
    --gt data/longbenchv2_img/OmniDocBench_concatenated.json \
    --pred data/pred/dpsk \
    --verbose
```

## Command-Line Arguments

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--gt` | `-g` | Yes | - | Path to ground truth JSON file |
| `--pred` | `-p` | Yes | - | Path to prediction directory containing .md files |
| `--output` | `-o` | No | None | Path to output JSON file for saving results |
| `--no-normalize` | - | No | False | Disable text normalization before metric calculation |
| `--verbose` | `-v` | No | False | Enable verbose logging |

## Evaluation Protocol

### 1. Data Format

**Ground Truth (JSON):**
- Format: JSON file with an array of page entries
- Each entry contains:
  - `page_info`: Metadata including `image_path` (e.g., `66ebb0e55a08c7b9b35ddd6a_page_001.png`)
  - `layout_dets`: Array with `text` field containing the ground truth content

**Predictions (Markdown):**
- Format: Directory containing `.md` files
- Filename format: `{page_id}.md` (e.g., `66ebb0e55a08c7b9b35ddd6a_page_001.md`)
- Content: Predicted OCR text in markdown format

### 2. Matching Strategy

For simple layouts, predictions are matched with ground truth based on filename:
- Extract `page_id` from prediction filename (e.g., `66ebb0e55a08c7b9b35ddd6a_page_001.md` → `66ebb0e55a08c7b9b35ddd6a_page_001`)
- Match with ground truth entry where `image_path` contains the same `page_id`

### 3. Text Normalization

Before calculating metrics, text is normalized using the following steps:

1. **Remove Markdown Fences**: Remove ```markdown, ```html, ```latex fences
2. **Convert Inline LaTeX**: Convert inline math formulas ($...$ or \(...\)) to Unicode
3. **Full-width to Half-width**: Convert full-width characters to half-width
4. **Replace Repeated Characters**: Standardize consecutive spaces and underscores
5. **Whitespace Normalization**: Replace multiple whitespace with single space

### 4. Metrics Calculation

All metrics are calculated per page and then aggregated:

- **Mean**: Average score across all pages
- **Std**: Standard deviation across all pages
- **Min**: Minimum score across all pages
- **Max**: Maximum score across all pages
- **Count**: Number of pages evaluated

## Output Format

### Console Output

```
================================================================================
EVALUATION RESULTS
================================================================================

Summary Statistics:
--------------------------------------------------------------------------------

CER:
  Mean:   5.6100
  Std:    2.3400
  Min:    1.2000
  Max:    12.5000
  Count:  100

BLEU:
  Mean:   85.3200
  ...

================================================================================

Per-Page Results:
--------------------------------------------------------------------------------
Page ID                                             CER     BLEU
--------------------------------------------------------------------------------
66ebb0e55a08c7b9b35ddd6a_page_001                5.61    85.32
66ebb0e55a08c7b9b35ddd6a_page_002                3.42    87.12
...
================================================================================
```

### JSON Output

```json
{
  "metrics": {
    "cer": [5.61, 3.42, ...],
    "bleu": [85.32, 87.12, ...]
  },
  "per_page": {
    "66ebb0e55a08c7b9b35ddd6a_page_001": {
      "cer": 5.61,
      "bleu": 85.32
    },
    ...
  },
  "summary": {
    "cer_mean": 5.61,
    "cer_std": 2.34,
    "cer_min": 1.20,
    "cer_max": 12.50,
    "cer_count": 100,
    ...
  }
}
```

## Examples

### Example 1: Evaluate DeepSeek-OCR Predictions

```bash
python -m robust_ocm.eval \
    --gt data/longbenchv2_img/OmniDocBench_concatenated.json \
    --pred data/pred/dpsk \
    --output results/dpsk_evaluation.json
```

### Example 2: Evaluate Qwen3-VL Predictions

```bash
python -m robust_ocm.eval \
    --gt data/longbenchv2_img/OmniDocBench_concatenated.json \
    --pred data/pred/qwen3 \
    --output results/qwen3_evaluation.json
```

### Example 3: Compare Multiple Models

```bash
# Evaluate model A
python -m robust_ocm.eval \
    --gt data/longbenchv2_img/OmniDocBench_concatenated.json \
    --pred data/pred/model_a \
    --output results/model_a.json

# Evaluate model B
python -m robust_ocm.eval \
    --gt data/longbenchv2_img/OmniDocBench_concatenated.json \
    --pred data/pred/model_b \
    --output results/model_b.json

# Compare results (using jq)
jq -s '.[0].summary + .[1].summary' results/model_a.json results/model_b.json
```

## API Usage

You can also use the evaluation module programmatically:

```python
from robust_ocm.eval import load_ground_truth, load_predictions, calculate_all_metrics

# Load data
ground_truths = load_ground_truth("data/longbenchv2_img/OmniDocBench_concatenated.json")
predictions = load_predictions("data/pred/dpsk")

# Calculate metrics
results = calculate_all_metrics(
    predictions=predictions,
    ground_truths=ground_truths,
    normalize=True,
)

# Print summary
print(results["summary"])
```

## Troubleshooting

### Missing Predictions

If you see warnings about missing predictions:

```
WARNING: Missing predictions for 5 ground truth pages
```

This means some ground truth pages don't have corresponding prediction files. Check that:
- Prediction filenames match ground truth page IDs
- All prediction files are in the correct directory

### High CER Values

If CER values are unexpectedly high (e.g., > 80%), check:
- Prediction files contain correct text (not empty or corrupted)
- Prediction files match the correct ground truth pages
- Text normalization is working as expected

### Import Errors

If you encounter import errors:

```bash
# Install missing dependencies
pip install Levenshtein sacrebleu sacremoses beautifulsoup4 pylatexenc
```

## References

- [OmniDocBench](https://github.com/OmniData-Org/OmniDocBench): Comprehensive document understanding benchmark
- [BLEU Score](https://github.com/mjpost/sacrebleu): BLEU metric implementation
- [Levenshtein Distance](https://github.com/maxbachmann/Levenshtein): Edit distance calculation

## License

This module is part of the robust_ocm project.