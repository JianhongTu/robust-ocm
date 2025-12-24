import argparse
import os
import json
from PIL import Image
from robust_ocm.adv import apply_perturbation
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate adversarial splits by applying perturbations to images.")
    parser.add_argument('--input-dir', default='data/longbenchv2_img/images', help='Input directory containing images')
    parser.add_argument('--perturbation-type', required=True, choices=[
        'jpeg_compression', 'binarization_thresholding', 'random_noise', 'blur', 'pixelation'
    ], help='Type of perturbation to apply')
    parser.add_argument('--output-dir', help='Output directory. If not specified, uses av_{perturbation_type}/images')
    
    # Perturbation-specific parameters
    parser.add_argument('--quality', type=int, default=50, help='Quality for JPEG compression (0-100)')
    parser.add_argument('--threshold', type=int, default=128, help='Threshold for binarization (0-255)')
    parser.add_argument('--noise-type', default='gaussian', choices=['gaussian', 'salt_and_pepper'], help='Noise type for random noise')
    parser.add_argument('--intensity', type=float, default=0.1, help='Intensity for random noise (0-1)')
    parser.add_argument('--radius', type=float, default=0.5, help='Radius for Gaussian blur')
    parser.add_argument('--pixel-size', type=int, default=10, help='Pixel size for pixelation')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of images to process (for testing)')
    
    args = parser.parse_args()
    
    if not args.output_dir:
        args.output_dir = f'data/av_{args.perturbation_type}/images'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect parameters based on perturbation type
    params = {}
    if args.perturbation_type == 'jpeg_compression':
        params['quality'] = args.quality
    elif args.perturbation_type == 'binarization_thresholding':
        params['threshold'] = args.threshold
    elif args.perturbation_type == 'random_noise':
        params['noise_type'] = args.noise_type
        params['intensity'] = args.intensity
    elif args.perturbation_type == 'blur':
        params['radius'] = args.radius
    elif args.perturbation_type == 'pixelation':
        params['pixel_size'] = args.pixel_size
    
    # Save metadata
    metadata = {
        'perturbation_type': args.perturbation_type,
        'params': params,
        'input_dir': args.input_dir,
        'output_dir': args.output_dir
    }
    metadata_path = os.path.join(args.output_dir, '..', 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Process images
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
    files_to_process = [f for f in os.listdir(args.input_dir) if f.lower().endswith(image_extensions)]
    if args.limit:
        files_to_process = files_to_process[:args.limit]
    
    processed_count = 0
    for filename in tqdm(files_to_process, desc="Processing images"):
        img_path = os.path.join(args.input_dir, filename)
        try:
            img = Image.open(img_path)
            perturbed = apply_perturbation(img, args.perturbation_type, **params)
            output_path = os.path.join(args.output_dir, filename)
            perturbed.save(output_path)
            processed_count += 1
        except Exception as e:
            print(f'Error processing {filename}: {e}')
    
    print(f'Done! Processed {processed_count} images. Metadata saved to {metadata_path}')

if __name__ == '__main__':
    main()