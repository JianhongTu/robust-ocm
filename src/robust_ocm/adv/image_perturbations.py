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
        """
        Apply JPEG compression artifacts to the image.
        
        This simulates lossy compression that occurs when saving images in JPEG format.
        Lower quality values introduce more compression artifacts (blocking, ringing, etc.).
        
        Args:
            data: PIL Image to compress
            quality: JPEG quality level (1-100, where 100 is highest quality/least compression)
        
        Returns:
            Compressed PIL Image with JPEG artifacts
        
        Note:
            - Quality 90-100: High quality, minimal artifacts
            - Quality 70-90: Good quality, some artifacts visible
            - Quality 50-70: Medium quality, visible artifacts
            - Quality 30-50: Low quality, significant artifacts
            - Quality 1-30: Very low quality, severe artifacts
        """
        if not isinstance(data, Image.Image):
            raise TypeError("Data must be a PIL Image")
        buffer = io.BytesIO()
        data.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

@register_perturbation('webp_compression')
class WebPCompressionPerturbation(Perturbation):
    def apply(self, data, quality: int = 50, **kwargs):
        """
        Apply WebP compression artifacts to the image.
        
        This simulates lossy compression that occurs when saving images in WebP format.
        WebP uses different compression algorithms than JPEG, resulting in different artifact patterns.
        
        Args:
            data: PIL Image to compress
            quality: WebP quality level (1-100, where 100 is highest quality/least compression)
        
        Returns:
            Compressed PIL Image with WebP artifacts
        
        Note:
            - Quality 90-100: High quality, minimal artifacts
            - Quality 70-90: Good quality, some artifacts visible
            - Quality 50-70: Medium quality, visible artifacts
            - Quality 30-50: Low quality, significant artifacts
            - Quality 1-30: Very low quality, severe artifacts
        
        WebP compression typically produces better quality than JPEG at the same file size,
        with different artifact characteristics (less blocking, more smoothing/blurring).
        """
        if not isinstance(data, Image.Image):
            raise TypeError("Data must be a PIL Image")
        buffer = io.BytesIO()
        data.save(buffer, format='WEBP', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

@register_perturbation('lossy_encoding')
class LossyEncodingPerturbation(Perturbation):
    def apply(self, data, format: str = 'jpeg', quality: int = 50, **kwargs):
        """
        Apply lossy encoding compression to the image.
        
        This unified perturbation supports multiple lossy image formats.
        
        Args:
            data: PIL Image to compress
            format: Compression format ('jpeg' or 'webp')
            quality: Quality level (1-100, where 100 is highest quality/least compression)
        
        Returns:
            Compressed PIL Image with compression artifacts
        
        Note:
            JPEG artifacts: Blocking, ringing, color bleeding
            WebP artifacts: Smoothing, blurring, less blocking than JPEG
        """
        if not isinstance(data, Image.Image):
            raise TypeError("Data must be a PIL Image")
        
        format_upper = format.upper()
        if format_upper not in ['JPEG', 'WEBP']:
            raise ValueError(f"Unsupported format: {format}. Choose from 'jpeg' or 'webp'")
        
        buffer = io.BytesIO()
        data.save(buffer, format=format_upper, quality=quality)
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

@register_perturbation('resampling_kernel')
class ResamplingKernelPerturbation(Perturbation):
    def apply(self, data, method: str = 'nearest', scale: float = 0.8, **kwargs) -> Any:
        """
        Apply resampling kernel by resizing the image with different resampling filters.
        
        Args:
            data: PIL Image to resample
            method: Resampling method ('nearest', 'bilinear', 'bicubic', 'lanczos')
            scale: Scale factor for resizing (e.g., 0.5 for 50% size, 0.8 for 80% size)
        
        Returns:
            Resampled PIL Image
        """
        if not isinstance(data, Image.Image):
            raise TypeError("Data must be a PIL Image")
        
        # Map method names to PIL resampling filters
        resampling_map = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS
        }
        
        if method not in resampling_map:
            raise ValueError(f"Unsupported resampling method: {method}. Choose from {list(resampling_map.keys())}")
        
        # Calculate new dimensions
        original_width, original_height = data.size
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize with the specified resampling filter
        return data.resize((new_width, new_height), resampling_map[method])