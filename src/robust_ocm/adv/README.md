# Adversarial Perturbations Module

This module provides functionality to generate adversarial splits of the dataset by applying various perturbations to text and image data.

## Used Corruption Splits
- Vanilla: No corruption; clean images rendered at 11pt and 200 dpi.
- Dense Text: Reduced font size and line height; rendered at 7pt
- Blurry: Upscaled from images rendered at 42 dpi to 200 dpi.
- Lossy Encoding: Compress with Jpeg with 1 quality setting.
- DPI Downscaling: Rendered at 72 dpi.
- Binarization: 1-bit color by thresholding/
- Resampling Kernel: Downscaling to half size with nearest method.

## Structure
- `perturbations.py`: Base classes and registry system.
- `text_perturbations.py`: Text-based perturbation implementations.
- `image_perturbations.py`: Image-based perturbation implementations.
- `niah_cli.py`: CLI tool for generating Needle in a Haystack (NIAH) synthetic datasets.

## TODO List of Perturbations to Implement

### Text-based / Config Perturbations (in `text_perturbations.py`)
- [x] Dense Text: Combine reduced font size and tight line spacing to create very compact text layout.
- [x] DPI / Resolution Downscaling: Modify rendering DPI to simulate lower resolution scans/captures.

### Image-based Perturbations (in `image_perturbations.py`)
- [x] Blur: Upscale images rendered at a low DPI to create blurry effect using bilinear.
- [x] Binarization: Convert images to binary using thresholding.
- [x] Lossy Encoding: Apply JPEG or WebP compression with configurable quality levels.
- [x] Resampling Kernel: Apply different resampling filters (nearest, bilinear, bicubic, lanczos) when resizing.

## Usage

The perturbations are designed to be modular. Each perturbation type is implemented as a class inheriting from a base `Perturbation` class. New perturbations can be easily added by subclassing and registering them in the perturbation registry.

To apply a perturbation:

```python
from robust_ocm.adv import apply_perturbation

# Apply a specific perturbation
perturbed_data = apply_perturbation(data, 'dense_text', font_size=8)
```

## Adding New Perturbations

1. Create a new class inheriting from `Perturbation`.
2. Implement the `apply` method.
3. Register the class using the `@register_perturbation` decorator.
4. Add the class to the appropriate file (`text_perturbations.py` or `image_perturbations.py`).

Example:

```python
@register_perturbation('new_perturbation')
class NewPerturbation(Perturbation):
    def apply(self, data, **kwargs):
        # Implementation here
        pass
```

## Commands to Generate Each Corruption

The following commands can be used to generate adversarial splits for each implemented perturbation. Ensure you are in the project root directory and have activated the appropriate environment (e.g., `micromamba activate robust_ocm`).

### Text-based / Config Perturbations

- **Dense Text** (combines reduced font size and tight line spacing):
  ```bash
  # Generate dense text with 8-point font (line-height = 9)
  python -m robust_ocm.adv.adv_render --perturbation-type dense_text --font-size 8
  
  # Generate dense text with 11-point font (line-height = 12)
  python -m robust_ocm.adv.adv_render --perturbation-type dense_text --font-size 11
  ```

- **DPI / Resolution Downscaling**:
  ```bash
  # Generate low-resolution images (72 DPI)
  python -m robust_ocm.adv.adv_render --perturbation-type dpi_downscale --dpi 72
  
  # Generate very low-resolution images (48 DPI)
  python -m robust_ocm.adv.adv_render --perturbation-type dpi_downscale --dpi 48
  ```

- **Tofu (Missing Characters)**:
  ```bash
  # Generate images with Verdana font (may cause tofu for unsupported characters)
  python -m robust_ocm.adv.adv_render --perturbation-type tofu
  ```

### Image-based Perturbations

- **Random Noise**:
  ```bash
  python -m robust_ocm.adv.adv_cli --perturbation-type random_noise --noise-type gaussian --intensity 0.3
  ```

- **Blur**:
  ```bash
  python -m robust_ocm.adv.adv_cli --perturbation-type blur --radius 1.25
  ```

- **Binarization**:
  ```bash
  python -m robust_ocm.adv.adv_cli --perturbation-type binarization_thresholding --threshold 128
  ```

- **Resampling Kernel**:
  ```bash
  python -m robust_ocm.adv.adv_cli --perturbation-type resampling_kernel --method nearest --scale 0.8
  ```

- **Lossy Encoding (JPEG)**:
  ```bash
  python -m robust_ocm.adv.adv_cli --perturbation-type lossy_encoding --format jpeg --quality 30
  ```

- **Lossy Encoding (WebP)**:
  ```bash
  python -m robust_ocm.adv.adv_cli --perturbation-type lossy_encoding --format webp --quality 40
  ```

Note: Adjust parameters as needed for your use case. Use `--limit N` to process only the first N samples for testing.

## Upscaling Script

The upscaling script (`scripts/upscale_images.py`) allows you to simulate the effect of capturing documents at low DPI and then upscaling to a higher DPI. This naturally introduces blur and loss of detail, simulating real-world scanning scenarios.

### Usage

```bash
python scripts/upscale_images.py --input-dir <input_dir> --output-dir <output_dir> --low-dpi <low_dpi> --high-dpi <high_dpi> --method <method>
```

### Arguments

- `--input-dir`: Input directory containing images
- `--output-dir`: Output directory for upscaled images
- `--low-dpi`: Original DPI of the images (e.g., 72)
- `--high-dpi`: Target DPI for upscaling (e.g., 200)
- `--method`: Upsampling method (`nearest`, `bilinear`, `bicubic`, `lanczos`)

### Examples

```bash
# Upscale from 72 DPI to 200 DPI using bilinear upscaling
python scripts/upscale_images.py --input-dir data/adv_dpi_72/images --output-dir data/adv_dpi_72_upscaled_200/images --low-dpi 72 --high-dpi 200 --method bilinear

# Upscale from 48 DPI to 200 DPI using bicubic upscaling
python scripts/upscale_images.py --input-dir data/adv_dpi_48/images --output-dir data/adv_dpi_48_upscaled_200/images --low-dpi 48 --high-dpi 200 --method bicubic
```

### How It Works

The script simulates a real-world pipeline:
1. **Downscale**: Reduces image dimensions to simulate low DPI capture
2. **Upscale**: Resizes back to original dimensions using the specified method

This naturally introduces blur and loss of detail, similar to what happens when low-quality scans are upscaled.

## Needle in a Haystack (NIAH) CLI

The NIAH CLI tool (`niah_cli.py`) generates synthetic datasets for testing retrieval capabilities in long-context scenarios. It inserts secret key-value pairs into documents and creates questions to retrieve them.

### Features

- Inserts 5 secret key-value pairs across different regions (0-20%, 20-40%, 40-60%, 60-80%, 80-100%) of each document
- Predefined categories: number, animal, flower, color, city, fruit, element, planet, instrument, gemstone
- Supports blacklist filtering to exclude specific instances
- Configurable multiplier to generate multiple questions per document
- Questions and answers always stored as lists for consistency

### Usage

```bash
python -m robust_ocm.adv.niah_cli --input <input_file> --output <output_file> [options]
```

### Arguments

- `--input`: Path to input ground truth JSON file (default: `data/longbenchv2/data.json`)
- `--output`: Path to output NIAH dataset JSON file (default: `data/niah/data.json`)
- `--blacklist`: Path to blacklist file containing instance IDs to exclude (optional)
- `--multiplier`: Number of questions to generate per instance (default: 1)
- `--num-secrets`: Number of secret key-value pairs to insert per instance (default: 5)
- `--seed`: Random seed for reproducibility (default: 42)

### Examples

```bash
# Basic usage with default settings
python -m robust_ocm.adv.niah_cli

# With blacklist filtering
python -m robust_ocm.adv.niah_cli --blacklist blacklist.txt

# Generate 3 questions per instance
python -m robust_ocm.adv.niah_cli --multiplier 3

# Custom input/output paths with blacklist
python -m robust_ocm.adv.niah_cli \
  --input data/longbenchv2/data.json \
  --output data/niah/data.json \
  --blacklist blacklist.txt \
  --multiplier 2 \
  --seed 42
```

### Output Format

Each NIAH instance contains:
- `_id`: Original instance ID
- `domain`: "Needle in a Haystack"
- `sub_domain`: "Secret Retrieval"
- `questions`: List of questions (e.g., "What is the secret animal in the above context?")
- `answers`: List of corresponding answers
- `context`: Modified context with inserted secrets
- `secrets`: Dictionary of all inserted key-value pairs
- `original_question`: Original question from the source dataset
- `original_answer`: Original answer from the source dataset

### Secret Insertion Format

Secrets are inserted as: `"The secret {key} is {value}."`

Example: `"The secret animal is penguin."`</content>
<parameter name="filePath">/home/jianhongtu/codes/robust_ocm/src/robust_ocm/adv/README.md