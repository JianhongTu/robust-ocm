#!/usr/bin/env python3
"""
CLI for generating adversarial splits of the dataset using perturbations.
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from robust_ocm.adv import apply_perturbation

def is_image_file(filepath):
    """Check if file is an image based on extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    return filepath.suffix.lower() in image_extensions

def process_images(input_dir, output_dir, perturbation_type, **kwargs):
    """Process all images in input_dir and save perturbed versions to output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    for root, dirs, files in os.walk(input_path):
        for file in files:
            file_path = Path(root) / file
            if not is_image_file(file_path):
                continue

            # Calculate relative path
            relative_path = file_path.relative_to(input_path)
            output_file_path = output_path / relative_path

            # Ensure output directory exists
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                # Load image
                with Image.open(file_path) as img:
                    # Apply perturbation
                    perturbed_img = apply_perturbation(img, perturbation_type, **kwargs)
                    # Save perturbed image
                    perturbed_img.save(output_file_path)
                    print(f"Processed: {relative_path}")
            except Exception as e:
                print(f"Error processing {relative_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate adversarial splits using perturbations.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/longbenchv2_img/images",
        help="Input directory containing images (default: data/longbenchv2_img/images)"
    )
    parser.add_argument(
        "--perturbation",
        type=str,
        required=True,
        help="Perturbation type to apply (e.g., jpeg_compression, blur, etc.)"
    )
    parser.add_argument(
        "--param",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Parameters for the perturbation (can be used multiple times)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory"
    )

    args = parser.parse_args()

    # Parse parameters
    params = {}
    if args.param:
        for key, value in args.param:
            # Try to convert to appropriate type
            try:
                # Try int
                params[key] = int(value)
            except ValueError:
                try:
                    # Try float
                    params[key] = float(value)
                except ValueError:
                    # Keep as string
                    params[key] = value

    # Set up output directory
    output_base = f"av_{args.perturbation}"
    output_images_dir = Path(output_base) / "images"

    if output_images_dir.exists() and not args.overwrite:
        print(f"Output directory {output_images_dir} already exists. Use --overwrite to overwrite.")
        return

    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        "perturbation_type": args.perturbation,
        "parameters": params,
        "input_directory": args.input,
        "output_directory": str(output_images_dir)
    }

    metadata_path = Path(output_base) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Applying perturbation '{args.perturbation}' with parameters: {params}")
    print(f"Input: {args.input}")
    print(f"Output: {output_images_dir}")
    print(f"Metadata saved to: {metadata_path}")

    # Process images
    process_images(args.input, str(output_images_dir), args.perturbation, **params)

    print("Processing complete!")

if __name__ == "__main__":
    main()