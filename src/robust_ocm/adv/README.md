# Adversarial Perturbations Module

This module provides functionality to generate adversarial splits of the dataset by applying various perturbations to text and image data.

## Structure
- `perturbations.py`: Base classes and registry system.
- `text_perturbations.py`: Text-based perturbation implementations.
- `image_perturbations.py`: Image-based perturbation implementations.

## TODO List of Perturbations to Implement

### Text-based Perturbations (in `text_perturbations.py`)
- [x] Font Weight: Vary the font weight by switching to bold font files (implemented, requires bold font variants).
- [x] Kerning Collisions: Simulate tighter spacing by reducing font size (implemented as placeholder).
- [x] Homoglyph Substitution: Replace characters with visually similar Unicode characters (implemented).
- [x] Line-Height Compression: Reduce line spacing in rendered output (implemented).

### Image-based Perturbations (in `image_perturbations.py`)
- [x] JPEG Compression: Apply JPEG compression artifacts at various quality levels (implemented).
- [x] Binarization Thresholding: Convert images to binary using different thresholds (implemented).
- [x] Random Noise: Add random noise (e.g., Gaussian, salt-and-pepper) to images (implemented).
- [x] Blur: Apply blurring effects (e.g., Gaussian blur) with varying intensities (implemented).
- [x] Pixelation: Reduce image resolution by pixelating (implemented).

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
```</content>
<parameter name="filePath">/home/jianhongtu/codes/robust_ocm/src/robust_ocm/adv/README.md