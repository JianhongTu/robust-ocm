# Adversarial Perturbations Module

This module provides functionality to generate adversarial splits of the dataset by applying various perturbations to text and image data.

## Structure
- `perturbations.py`: Base classes and registry system.
- `text_perturbations.py`: Text-based perturbation implementations.
- `image_perturbations.py`: Image-based perturbation implementations.

## TODO List of Perturbations to Implement

### Text-based Perturbations (in `text_perturbations.py`)
- [] Font Weight: Vary the font weight by switching to bold font files (implemented, requires bold font variants).
- [] Kerning Collisions: Simulate tighter spacing by reducing font size (implemented as placeholder).
- [] Homoglyph Substitution: Replace characters with visually similar Unicode characters.
- [] Line-Height Compression: Reduce line spacing in rendered output.
- [] Tofu: Force tofu rendering by using Verdana font.

### Image-based Perturbations (in `image_perturbations.py`)
- [x] JPEG Compression: Apply JPEG compression artifacts at various quality levels (implemented).
- [x] Binarization Thresholding: Convert images to binary using different thresholds (implemented).
- [x] Random Noise: Add random noise (e.g., Gaussian, salt-and-pepper) to images (implemented).
- [x] Blur: Apply blurring effects (e.g., Gaussian blur) with varying intensities (implemented).

## Usage

The perturbations are designed to be modular. Each perturbation type is implemented as a class inheriting from a base `Perturbation` class. New perturbations can be easily added by subclassing and registering them in the perturbation registry.

To apply a perturbation:

```python
from robust_ocm.adv import apply_perturbation

# Apply a specific perturbation
perturbed_data = apply_perturbation(data, 'font_weight', **params)
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

### Text-based Perturbations

- **Font Weight** (requires bold font variants):
  ```bash
  python -m robust_ocm.adv.adv_render --perturbation-type font_weight --weight bold --output-dir data/adv_bold
  ```

- **Kerning Collisions**:
  ```bash
  python -m robust_ocm.adv.adv_render --perturbation-type kerning_collisions --collision-factor 0.1 --output-dir data/adv_kerning_collisions
  ```

- **Homoglyph Substitution**:
  ```bash
  python -m robust_ocm.adv.adv_render --perturbation-type homoglyph_substitution --substitution-rate 0.3 --output-dir data/adv_homoglyph_0.3
  ```

- **Line-Height Compression**:
  ```bash
  python -m robust_ocm.adv.adv_render --perturbation-type line_height_compression --compression-factor 0.8 --output-dir data/adv_line_height_compression
  ```

- **Tofu**:
  ```bash
  python -m robust_ocm.adv.adv_render --perturbation-type tofu --output-dir data/adv_tofu
  ```

### Image-based Perturbations

- **JPEG Compression**:
  ```bash
  python -m robust_ocm.adv.adv_cli --perturbation-type jpeg_compression --quality 10
  ```

- **Binarization Thresholding**:
  ```bash
  python -m robust_ocm.adv.adv_cli --perturbation-type binarization_thresholding --threshold 128
  ```

- **Random Noise**:
  ```bash
  python -m robust_ocm.adv.adv_cli --perturbation-type random_noise --noise-type gaussian --intensity 0.3
  ```

- **Blur**:
  ```bash
  python -m robust_ocm.adv.adv_cli --perturbation-type blur --radius 1.25
  ```

Note: Adjust parameters as needed for your use case. Use `--limit N` to process only the first N samples for testing.</content>
<parameter name="filePath">/home/jianhongtu/codes/robust_ocm/src/robust_ocm/adv/README.md