from .perturbations import register_perturbation, Perturbation
from PIL import Image, ImageFilter
import numpy as np
import io
import random
from typing import Any

# Image-based perturbations
# Assuming data is a PIL Image

@register_perturbation('jpeg_compression')
class JPEGCompressionPerturbation(Perturbation):
    def apply(self, data, quality: int = 50, **kwargs):
        """Apply JPEG compression artifacts."""
        if not isinstance(data, Image.Image):
            raise TypeError("Data must be a PIL Image")
        buffer = io.BytesIO()
        data.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

@register_perturbation('binarization_thresholding')
class BinarizationThresholdingPerturbation(Perturbation):
    def apply(self, data, threshold: int = 128, **kwargs):
        """Convert image to binary using thresholding."""
        if not isinstance(data, Image.Image):
            raise TypeError("Data must be a PIL Image")
        # Convert to grayscale if not already
        if data.mode != 'L':
            data = data.convert('L')
        # Apply threshold
        return data.point(lambda p: 255 if p > threshold else 0).convert('1')

@register_perturbation('random_noise')
class RandomNoisePerturbation(Perturbation):
    def apply(self, data, noise_type: str = 'gaussian', intensity: float = 0.1, **kwargs):
        """Add random noise to the image."""
        if not isinstance(data, Image.Image):
            raise TypeError("Data must be a PIL Image")
        img_array = np.array(data)
        if noise_type == 'gaussian':
            noise = np.random.normal(0, intensity * 128, img_array.shape).astype(np.int16)
            noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif noise_type == 'salt_and_pepper':
            # Salt and pepper noise
            prob = intensity
            random_matrix = np.random.rand(*img_array.shape[:2])
            noisy_array = img_array.copy()
            noisy_array[random_matrix < prob/2] = 0  # pepper
            noisy_array[random_matrix > 1 - prob/2] = 255  # salt
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        return Image.fromarray(noisy_array)

@register_perturbation('blur')
class BlurPerturbation(Perturbation):
    def apply(self, data, blur_type: str = 'gaussian', radius: float = 0.5, **kwargs) -> Any:
        """Apply blurring to the image."""
        if not isinstance(data, Image.Image):
            raise TypeError("Data must be a PIL Image")
        if blur_type == 'gaussian':
            return data.filter(ImageFilter.GaussianBlur(radius))
        else:
            raise ValueError(f"Unsupported blur type: {blur_type}")