#!/usr/bin/env python3
"""
Single GPU DeepSeek-OCR Inference Script

A simplified, production-ready script for running DeepSeek-OCR inference
on a single GPU using vLLM. Designed for robustness and ease of use.

Usage:
    python ds_single_gpu_inference.py --input data/longbenchv2_img/images --output data/pred/dpsk
"""

import os
import sys
import argparse
import glob
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
from PIL import Image
import re
import json

# Add repo path for imports
repo_path = Path(__file__).parent / "repo/DeepSeek-OCR-master/DeepSeek-OCR-vllm"
sys.path.insert(0, str(repo_path))

from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inference.log')
    ]
)
logger = logging.getLogger(__name__)


class SingleGPUInference:
    """Single GPU inference wrapper for DeepSeek-OCR"""
    
    def __init__(
        self,
        model_path: str = 'deepseek-ai/DeepSeek-OCR',
        max_concurrency: int = 100,
        gpu_memory_utilization: float = 0.9,
        max_tokens: int = 8192,
        temperature: float = 0.0
    ):
        """
        Initialize vLLM model for single GPU inference
        
        Args:
            model_path: Path to DeepSeek-OCR model
            max_concurrency: Maximum number of concurrent sequences
            gpu_memory_utilization: GPU memory fraction to use (0.0-1.0)
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
        """
        self.model_path = model_path
        self.max_concurrency = max_concurrency
        self.gpu_memory_utilization = gpu_memory_utilization
        
        logger.info(f"Initializing vLLM with model: {model_path}")
        logger.info(f"Max concurrency: {max_concurrency}, GPU memory: {gpu_memory_utilization*100:.1f}%")
        
        self.llm = LLM(
            model=model_path,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor],
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_concurrency,
            trust_remote_code=True,
        )
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
            ),
            skip_special_tokens=False,
        )
        
        logger.info("vLLM initialization complete")
    
    def infer_batch(
        self,
        image_paths: List[str],
        prompt: str = '<image>\n<|grounding|>Convert the document to markdown.'
    ) -> List[Tuple[str, str]]:
        """
        Run inference on a batch of images
        
        Args:
            image_paths: List of image file paths
            prompt: Prompt to use for OCR
            
        Returns:
            List of (image_path, ocr_result) tuples
        """
        model_input = []
        valid_paths = []
        
        # Load and validate images
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                model_input.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": img}
                })
                valid_paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
                continue
        
        if not model_input:
            logger.warning("No valid images in batch")
            return []
        
        # Run inference
        try:
            outputs = self.llm.generate(model_input, self.sampling_params)
            
            # Extract results
            results = []
            for output, path in zip(outputs, valid_paths):
                content = output.outputs[0].text
                results.append((path, content))
            
            return results
            
        except Exception as e:
            logger.error(f"Inference failed for batch: {e}")
            return []
    
    def infer(
        self,
        image_paths: List[str],
        batch_size: int = 64,
        prompt: str = '<image>\n<|grounding|>Convert the document to markdown.',
        progress_bar: bool = True
    ) -> List[Tuple[str, str]]:
        """
        Run inference on all images with batching
        
        Args:
            image_paths: List of all image paths to process
            batch_size: Number of images per batch (default: 64, recommended: 64-128)
            prompt: Prompt to use for OCR
            progress_bar: Show progress bar
            
        Returns:
            List of (image_path, ocr_result) tuples
        """
        all_results = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(image_paths)} images in {total_batches} batches")
        
        iterator = range(0, len(image_paths), batch_size)
        if progress_bar:
            iterator = tqdm(iterator, desc="Processing batches", unit="batch")
        
        for i in iterator:
            batch_paths = image_paths[i:i + batch_size]
            batch_results = self.infer_batch(batch_paths, prompt)
            all_results.extend(batch_results)
            
            if progress_bar:
                iterator.set_postfix({"processed": len(all_results)})
        
        logger.info(f"Completed inference: {len(all_results)} results")
        return all_results


def get_image_paths(input_path: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    Get all image paths from directory
    
    Args:
        input_path: Directory containing images
        extensions: List of file extensions (default: common image formats)
        
    Returns:
        Sorted list of image paths
    """
    if extensions is None:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_path, ext)))
        image_paths.extend(glob.glob(os.path.join(input_path, ext.upper())))
    
    return sorted(set(image_paths))


def clean_formula(text: str) -> str:
    """Clean formula formatting in OCR output"""
    formula_pattern = r'\\\[(.*?)\\\]'
    
    def process_formula(match):
        formula = match.group(1)
        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        formula = formula.strip()
        return r'\[' + formula + r'\]'
    
    cleaned_text = re.sub(formula_pattern, process_formula, text)
    return cleaned_text


def clean_ocr_content(content: str) -> str:
    """
    Clean and normalize OCR output
    
    Args:
        content: Raw OCR output
        
    Returns:
        Cleaned content
    """
    # Clean formulas
    content = clean_formula(content)
    
    # Remove reference and detection tags
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        content = content.replace(match[0], '')
    
    # Normalize line breaks
    content = content.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')
    
    # Remove center tags
    content = content.replace('<center>', '').replace('</center>', '')
    
    return content.strip()


def save_results(results: List[Tuple[str, str]], output_dir: str) -> Tuple[int, int]:
    """
    Save OCR results to markdown files
    
    Args:
        results: List of (image_path, ocr_result) tuples
        output_dir: Directory to save results
        
    Returns:
        Tuple of (saved_count, failed_count)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    failed_count = 0
    failed_samples = []
    
    for image_path, content in results:
        try:
            # Clean content
            cleaned_content = clean_ocr_content(content)
            
            # Save to markdown file
            mmd_path = output_dir / (Path(image_path).stem + '.md')
            with open(mmd_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            saved_count += 1
            
        except Exception as e:
            logger.error(f"Failed to save result for {image_path}: {e}")
            failed_count += 1
            failed_samples.append({
                "path": image_path,
                "error": str(e)
            })
    
    # Save failed samples list
    if failed_samples:
        failed_path = output_dir / "failed_samples.json"
        with open(failed_path, 'w', encoding='utf-8') as f:
            json.dump(failed_samples, f, indent=2, ensure_ascii=False)
        logger.warning(f"Failed to save {failed_count} samples. See {failed_path}")
    
    logger.info(f"Saved {saved_count} results, failed {failed_count}")
    return saved_count, failed_count


def main():
    parser = argparse.ArgumentParser(
        description='Single GPU DeepSeek-OCR inference with vLLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (recommended for A100 GPU)
    python ds_single_gpu_inference.py --input data/longbenchv2_img/images --output data/pred/dpsk
    
    # Limit to 100 images for testing
    python ds_single_gpu_inference.py --input data/longbenchv2_img/images --output data/pred/dpsk --limit 100
    
    # Large batch size for maximum GPU utilization (A100 40GB)
    python ds_single_gpu_inference.py --input data/longbenchv2_img/images --output data/pred/dpsk --batch-size 128 --max-concurrency 128
    
    # Medium batch size (RTX 3090/4090 24GB)
    python ds_single_gpu_inference.py --input data/longbenchv2_img/images --output data/pred/dpsk --batch-size 64 --max-concurrency 64 --gpu-memory 0.85
    
    # Use custom prompt
    python ds_single_gpu_inference.py --input data/longbenchv2_img/images --output data/pred/dpsk --prompt "Extract text from image"
    
    # Specify GPU device
    python ds_single_gpu_inference.py --input data/longbenchv2_img/images --output data/pred/dpsk --gpu 1
    
Performance Tips:
    - For optimal GPU utilization, set batch_size close to max_concurrency
    - A100 40GB: batch_size=128, max_concurrency=128
    - A100 80GB: batch_size=200, max_concurrency=200
    - RTX 3090/4090: batch_size=64, max_concurrency=64
    - V100: batch_size=80, max_concurrency=80
        """
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input directory containing images')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for OCR results')
    
    # Optional arguments
    parser.add_argument('--limit', '-l', type=int, default=None,
                       help='Limit number of images to process')
    parser.add_argument('--batch-size', '-b', type=int, default=2048,
                       help='Batch size for inference (default: 64, recommended: 64-128 for optimal GPU utilization)')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device ID (default: 0)')
    parser.add_argument('--model-path', type=str, default='deepseek-ai/DeepSeek-OCR',
                       help='Path to DeepSeek-OCR model')
    parser.add_argument('--max-concurrency', type=int, default=100,
                       help='Maximum concurrent sequences (default: 100)')
    parser.add_argument('--gpu-memory', type=float, default=0.9,
                       help='GPU memory utilization 0.0-1.0 (default: 0.9)')
    parser.add_argument('--max-tokens', type=int, default=8192,
                       help='Maximum output tokens (default: 8192)')
    parser.add_argument('--prompt', type=str,
                       default='<image>\n<|grounding|>Convert the document to markdown.',
                       help='Prompt for OCR task')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    
    args = parser.parse_args()
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logger.info(f"Using GPU: {args.gpu}")
    
    # Validate input path
    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    if not os.path.isdir(args.input):
        logger.error(f"Input path is not a directory: {args.input}")
        sys.exit(1)
    
    # Get image paths
    logger.info(f"Scanning images in {args.input}")
    image_paths = get_image_paths(args.input)
    
    if not image_paths:
        logger.error(f"No images found in {args.input}")
        sys.exit(1)
    
    # Apply limit if specified
    if args.limit:
        image_paths = image_paths[:args.limit]
        logger.info(f"Limited to {len(image_paths)} images")
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Initialize inference engine
    try:
        inference = SingleGPUInference(
            model_path=args.model_path,
            max_concurrency=args.max_concurrency,
            gpu_memory_utilization=args.gpu_memory,
            max_tokens=args.max_tokens,
        )
        
        # Check batch size vs max_concurrency
        if args.batch_size < args.max_concurrency * 0.5:
            logger.warning(
                f"batch_size ({args.batch_size}) is much smaller than max_concurrency ({args.max_concurrency}). "
                f"Consider increasing batch_size to {args.max_concurrency} for better GPU utilization."
            )
        else:
            logger.info(
                f"Batch configuration: batch_size={args.batch_size}, max_concurrency={args.max_concurrency} "
                f"(Good GPU utilization expected)"
            )
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        sys.exit(1)
    
    # Run inference
    start_time = time.time()
    
    try:
        results = inference.infer(
            image_paths=image_paths,
            batch_size=args.batch_size,
            prompt=args.prompt,
            progress_bar=not args.no_progress
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)
    
    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.2f}s ({len(results)/inference_time:.2f} img/s)")
    
    # Save results
    logger.info(f"Saving results to {args.output}")
    saved_count, failed_count = save_results(results, args.output)
    
    # Summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total images: {len(image_paths)}")
    logger.info(f"Successfully processed: {len(results)}")
    logger.info(f"Successfully saved: {saved_count}")
    logger.info(f"Failed to save: {failed_count}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average speed: {len(results)/total_time:.2f} img/s")
    logger.info(f"Output directory: {args.output}")
    logger.info("=" * 60)
    
    if failed_count > 0:
        logger.warning(f"Check {args.output}/failed_samples.json for details")


if __name__ == "__main__":
    main()