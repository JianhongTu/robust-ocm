#!/usr/bin/env python
"""
Script to upscale images from low DPI to high DPI using bilinear resampling.
"""

import os
import argparse
from PIL import Image
from tqdm import tqdm


def upscale_images(input_dir, output_dir, low_dpi=72, high_dpi=200, method='bilinear'):
    """
    Upscale images from low DPI to high DPI.
    
    Args:
        input_dir: Directory containing low DPI images
        output_dir: Directory to save upscaled images
        low_dpi: Original DPI of images
        high_dpi: Target DPI
        method: Upscaling method ('nearest', 'bilinear', 'bicubic', 'lanczos')
    """
    # Map method names to PIL resampling filters
    resampling_map = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }
    
    if method not in resampling_map:
        raise ValueError(f"Unsupported method: {method}. Choose from {list(resampling_map.keys())}")
    
    # Calculate upscale factor
    scale_factor = high_dpi / low_dpi
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Upscaling from {low_dpi} DPI to {high_dpi} DPI (scale factor: {scale_factor:.2f}x)")
    print(f"Using {method} resampling")
    
    # Process images
    for filename in tqdm(image_files, desc="Upscaling images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Open image
            img = Image.open(input_path)
            original_width, original_height = img.size
            
            # Calculate new dimensions
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Upscale image
            upscaled = img.resize((new_width, new_height), resampling_map[method])
            
            # Save upscaled image
            upscaled.save(output_path)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"Done! Upscaled images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Upscale images from low DPI to high DPI")
    parser.add_argument('--input-dir', default='data/adv_dpi_72/images',
                        help='Input directory containing low DPI images')
    parser.add_argument('--output-dir', default='data/adv_dpi_72_upscaled_200/images',
                        help='Output directory for upscaled images')
    parser.add_argument('--low-dpi', type=int, default=72,
                        help='Original DPI of images (default 72)')
    parser.add_argument('--high-dpi', type=int, default=200,
                        help='Target DPI (default 200)')
    parser.add_argument('--method', default='bilinear',
                        choices=['nearest', 'bilinear', 'bicubic', 'lanczos'],
                        help='Upscaling method (default bilinear)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of images to process')
    
    args = parser.parse_args()
    
    upscale_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        low_dpi=args.low_dpi,
        high_dpi=args.high_dpi,
        method=args.method
    )


if __name__ == '__main__':
    main()