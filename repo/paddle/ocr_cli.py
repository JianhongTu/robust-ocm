#!/usr/bin/env python3
"""
Generic OCR CLI Tool

A flexible command-line tool for extracting text and bounding boxes from images
using PaddleOCR. Supports both word-level and block-level extraction with
configurable input/output options.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from paddleocr import PaddleOCR


class OCRExtractor:
    """Extract text and bounding boxes using PaddleOCR"""
    
    def __init__(self, lang='en', verbose=False, visualize=False, viz_output_dir='output', extraction_level='word'):
        """
        Initialize PaddleOCR with configurable detection settings
        
        Args:
            lang: Language code ('ch', 'en', etc.)
            verbose: Enable verbose output
            visualize: Enable visualization output
            viz_output_dir: Directory for visualization outputs
            extraction_level: 'word' for word-level or 'block' for block-level extraction
        """
        self.verbose = verbose
        self.visualize = visualize
        self.viz_output_dir = Path(viz_output_dir)
        self.extraction_level = extraction_level
        
        if self.visualize:
            self.viz_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Visualization output directory: {self.viz_output_dir}")
        
        # Configure PaddleOCR based on extraction level
        ocr_config = {
            'use_doc_orientation_classify': False,
            'use_doc_unwarping': False,
            'use_textline_orientation': False,
            'lang': lang,
        }
        
        if extraction_level == 'word':
            ocr_config['return_word_box'] = True
        
        self.ocr = PaddleOCR(**ocr_config)
        
    def process_image(self, image_path: str) -> Optional[Dict]:
        """
        Process a single image and return OCR results
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing image info and extraction results, or None on error
        """
        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            return None
            
        try:
            result = self.ocr.predict(input=image_path)
            
            if not result:
                if self.verbose:
                    print(f"No OCR result for: {image_path}")
                return None
            
            res = result[0]
            
            # Extract based on extraction level
            if self.extraction_level == 'word':
                # Extract word-level poly and text
                word_boxes = res.get('text_word_boxes') or []
                word_texts = res.get('text_word') or []
                
                # Flatten to parallel arrays
                polys = []
                texts = []
                confidences = []
                for line_words, line_boxes in zip(word_texts, word_boxes):
                    for word_text, word_box in zip(line_words, line_boxes):
                        if word_text and word_text.strip():
                            # Convert numpy to list
                            if isinstance(word_box, np.ndarray):
                                word_box = word_box.tolist()
                            polys.append(word_box)
                            texts.append(word_text.strip())
                            confidences.append(1.0)  # PaddleOCR doesn't provide word-level confidence
                
                result_data = {
                    'image_path': image_path,
                    'image_name': os.path.basename(image_path),
                    'extraction_level': 'word',
                    'word_count': len(texts),
                    'polys': polys,
                    'texts': texts,
                    'confidences': confidences
                }
            else:
                # Extract block-level poly and text
                block_boxes = res.get('rec_polys') or []
                block_texts = res.get('rec_texts') or []
                block_confs = res.get('rec_scores') or []
                
                # Flatten to parallel arrays
                polys = []
                texts = []
                confidences = []
                for block_text, block_box, block_conf in zip(block_texts, block_boxes, block_confs):
                    if block_text and block_text.strip():
                        # Convert numpy to list
                        if isinstance(block_box, np.ndarray):
                            block_box = block_box.tolist()
                        elif isinstance(block_box, str):
                            # Handle string representation of numpy arrays
                            import re
                            # Extract numbers from string like "[[14 16]\n ...]"
                            numbers = re.findall(r'\d+', block_box)
                            if numbers:
                                # Convert to list of ints, reshape if needed
                                nums = [int(n) for n in numbers]
                                # For bounding boxes, we expect 8 numbers (4 points x 2 coords)
                                if len(nums) >= 8:
                                    block_box = [[nums[0], nums[1]], [nums[2], nums[3]], 
                                                [nums[4], nums[5]], [nums[6], nums[7]]]
                                else:
                                    continue
                            else:
                                continue
                        polys.append(block_box)
                        texts.append(block_text.strip())
                        confidences.append(float(block_conf))
                
                result_data = {
                    'image_path': image_path,
                    'image_name': os.path.basename(image_path),
                    'extraction_level': 'block',
                    'block_count': len(texts),
                    'polys': polys,
                    'texts': texts,
                    'confidences': confidences
                }
            
            # Save visualization if enabled
            if self.visualize:
                for res_obj in result:
                    if self.verbose:
                        res_obj.print()
                    # Save visualization image
                    res_obj.save_to_img(str(self.viz_output_dir))
            
            return result_data
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def process_directory(self, 
                         input_dir: str,
                         recursive: bool = False,
                         limit: Optional[int] = None,
                         image_extensions: List[str] = None) -> List[Dict]:
        """
        Process all images in a directory
        
        Args:
            input_dir: Path to input directory
            recursive: Whether to search subdirectories recursively
            limit: Maximum number of images to process (None for all)
            image_extensions: List of supported image extensions
            
        Returns:
            List of results from processed images
        """
        if not os.path.exists(input_dir):
            print(f"Error: Directory not found: {input_dir}")
            return []
        
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        
        # Find all image files
        image_paths = []
        input_path = Path(input_dir)
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for ext in image_extensions:
            for img_path in input_path.glob(pattern + ext):
                image_paths.append(str(img_path))
            for img_path in input_path.glob(pattern + ext.upper()):
                image_paths.append(str(img_path))
        
        image_paths = sorted(list(set(image_paths)))  # Remove duplicates and sort
        
        if not image_paths:
            print(f"No images found in directory: {input_dir}")
            return []
        
        print(f"Found {len(image_paths)} images in directory")
        
        if limit:
            image_paths = image_paths[:limit]
            print(f"Processing first {len(image_paths)} images (limit applied)")
        
        results = []
        for i, img_path in enumerate(image_paths, 1):
            print(f"Processing [{i}/{len(image_paths)}]: {os.path.basename(img_path)}")
            result = self.process_image(img_path)
            if result:
                results.append(result)
        
        return results
    
    def process_file_list(self, 
                         file_list_path: str,
                         limit: Optional[int] = None) -> List[Dict]:
        """
        Process images listed in a text file (one path per line)
        
        Args:
            file_list_path: Path to text file containing image paths
            limit: Maximum number of images to process (None for all)
            
        Returns:
            List of results from processed images
        """
        if not os.path.exists(file_list_path):
            print(f"Error: File list not found: {file_list_path}")
            return []
        
        try:
            with open(file_list_path, 'r') as f:
                image_paths = [line.strip() for line in f if line.strip()]
            
            print(f"Loaded {len(image_paths)} image paths from file list")
            
            if limit:
                image_paths = image_paths[:limit]
                print(f"Processing first {len(image_paths)} images (limit applied)")
            
            results = []
            for i, img_path in enumerate(image_paths, 1):
                print(f"Processing [{i}/{len(image_paths)}]: {os.path.basename(img_path)}")
                result = self.process_image(img_path)
                if result:
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error reading file list: {e}")
            return []


def save_results(results: List[Dict], output_path: str, format_type: str = 'json'):
    """
    Save results to file
    
    Args:
        results: List of OCR results
        output_path: Output file path
        format_type: Output format ('json', 'csv', 'txt')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type.lower() == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    elif format_type.lower() == 'csv':
        import csv
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['image_name', 'extraction_level', 'text', 'confidence', 'polygon'])
            # Write data
            for result in results:
                image_name = result['image_name']
                extraction_level = result['extraction_level']
                texts = result.get('texts', [])
                confidences = result.get('confidences', [])
                polys = result.get('polys', [])
                
                for text, conf, poly in zip(texts, confidences, polys):
                    writer.writerow([image_name, extraction_level, text, conf, str(poly)])
    
    elif format_type.lower() == 'txt':
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"Image: {result['image_name']}\n")
                f.write(f"Extraction level: {result['extraction_level']}\n")
                
                if result.get('extraction_level') == 'word':
                    f.write(f"Word count: {result['word_count']}\n")
                else:
                    f.write(f"Block count: {result['block_count']}\n")
                
                f.write("Results:\n")
                texts = result.get('texts', [])
                confidences = result.get('confidences', [])
                polys = result.get('polys', [])
                
                for i, (text, conf, poly) in enumerate(zip(texts, confidences, polys), 1):
                    f.write(f"  {i}. '{text}' (conf: {conf:.3f}) at {poly}\n")
                f.write("\n" + "="*60 + "\n\n")
    
    print(f"\nResults saved to: {output_path}")
    print(f"Total images processed: {len(results)}")
    
    # Print summary statistics
    if results:
        extraction_level = results[0].get('extraction_level', 'word')
        if extraction_level == 'word':
            total_items = sum(r.get('word_count', 0) for r in results)
            item_type = 'words'
        else:
            total_items = sum(r.get('block_count', 0) for r in results)
            item_type = 'blocks'
        
        print(f"Total {item_type} detected: {total_items}")
        avg_items = total_items / len(results)
        print(f"Average {item_type} per image: {avg_items:.1f}")


def print_sample_result(result: Dict):
    """Print a sample result for debugging"""
    print("\n" + "="*60)
    print("Sample Result:")
    print("="*60)
    print(f"Image: {result['image_name']}")
    print(f"Extraction level: {result.get('extraction_level', 'word')}")
    
    if result.get('extraction_level') == 'word':
        print(f"Word count: {result['word_count']}")
        item_type = "words"
    else:
        print(f"Block count: {result['block_count']}")
        item_type = "blocks"
    
    print(f"\nFirst 5 {item_type}:")
    texts = result.get('texts', [])
    confidences = result.get('confidences', [])
    polys = result.get('polys', [])
    for i in range(min(5, len(texts))):
        print(f"  {i+1}. '{texts[i]}' (conf: {confidences[i]:.3f}) at {polys[i][:2]}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generic OCR CLI Tool for Text and Bounding Box Extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image (word-level)
  python ocr_cli.py --image path/to/image.png
  
  # Process single image (block-level)
  python ocr_cli.py --image path/to/image.png --extraction-level block
  
  # Process all images in directory
  python ocr_cli.py --directory ./images --recursive
  
  # Process images from file list
  python ocr_cli.py --file-list image_paths.txt
  
  # Process with visualization and custom output
  python ocr_cli.py --image path/to/image.png --visualize --output-dir ./results
  
  # Process with Chinese language model
  python ocr_cli.py --directory ./images --lang ch --limit 10
  
  # Save results in different formats
  python ocr_cli.py --directory ./images --format csv
  python ocr_cli.py --directory ./images --format txt
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', '-i', 
                           help='Path to single image file')
    input_group.add_argument('--directory', '-d',
                           help='Path to directory containing images')
    input_group.add_argument('--file-list', '-f',
                           help='Path to text file containing image paths (one per line)')
    
    # Processing options
    parser.add_argument('--recursive', '-r',
                       action='store_true',
                       help='Search subdirectories recursively (only with --directory)')
    parser.add_argument('--limit', '-l', 
                       type=int,
                       help='Maximum number of images to process')
    parser.add_argument('--extraction-level', '-e',
                       default='word',
                       choices=['word', 'block'],
                       help='Extraction level: word-level or block-level (default: word)')
    parser.add_argument('--lang', 
                       default='en',
                       choices=['ch', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka', 'latin', 'arabic', 'cyrillic', 'devanagari'],
                       help='Language for OCR (default: en)')
    
    # Output options
    parser.add_argument('--output-dir', '-o', 
                       default='output',
                       help='Output directory for results (default: output)')
    parser.add_argument('--output-name', '-n',
                       default='ocr_results',
                       help='Output filename without extension (default: ocr_results)')
    parser.add_argument('--format', '-F',
                       default='json',
                       choices=['json', 'csv', 'txt'],
                       help='Output format (default: json)')
    
    # Other options
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--show-sample', '-s',
                       action='store_true',
                       help='Print sample result after processing')
    parser.add_argument('--visualize', '--viz',
                       action='store_true',
                       help='Save visualization images with bounding boxes')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.recursive and not args.directory:
        parser.error("--recursive can only be used with --directory")
    
    # Initialize extractor
    print(f"Initializing PaddleOCR (language: {args.lang}, extraction level: {args.extraction_level})...")
    if args.visualize:
        print(f"Visualization enabled: images will be saved to {args.output_dir}")
    
    extractor = OCRExtractor(
        lang=args.lang, 
        verbose=args.verbose,
        visualize=args.visualize,
        viz_output_dir=args.output_dir,
        extraction_level=args.extraction_level
    )
    
    results = []
    
    # Process input
    if args.image:
        print(f"\nProcessing single image: {args.image}")
        result = extractor.process_image(args.image)
        if result:
            results.append(result)
    
    elif args.directory:
        print(f"\nProcessing directory: {args.directory}")
        if args.recursive:
            print("Recursive search enabled")
        if args.limit:
            print(f"Limit: {args.limit} images")
        results = extractor.process_directory(args.directory, recursive=args.recursive, limit=args.limit)
    
    elif args.file_list:
        print(f"\nProcessing file list: {args.file_list}")
        if args.limit:
            print(f"Limit: {args.limit} images")
        results = extractor.process_file_list(args.file_list, limit=args.limit)
    
    # Save results
    if results:
        # Construct output filename
        output_filename = f"{args.output_name}.{args.format}"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Save results
        save_results(results, output_path, args.format)
        
        # Show sample if requested
        if args.show_sample and results:
            print_sample_result(results[0])
    else:
        print("\nNo results to save. Please check your input.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())