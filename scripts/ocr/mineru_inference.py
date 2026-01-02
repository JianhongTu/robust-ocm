#!/usr/bin/env python3
"""
MinerU2.5 OCR inference script with concurrent async processing.

To start vLLM server for MinerU2.5:
vllm serve opendatalab/MinerU2.5-2509-1.2B \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --trust-remote-code \
    --logits_processors mineru_vl_utils:MinerULogitsProcessor

Then run this script:
micromamba run -n test python scripts/ocr/mineru_inference.py \
    --input data/ocr \
    --output data/pred/mineru \
    --max_workers 64
"""

import io
import asyncio
import os
import sys
import argparse
from pathlib import Path
from PIL import Image
from tqdm.asyncio import tqdm as async_tqdm
from concurrent.futures import ThreadPoolExecutor

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from mineru_vl_utils import MinerUClient
from mineru_vl_utils import MinerULogitsProcessor


async def process_image(client, image_file, image_dir, result_dir, executor):
    """
    Process a single image with MinerU and save results.
    
    Args:
        client: MinerUClient instance
        image_file: Name of the image file to process
        image_dir: Directory containing the image
        result_dir: Directory to save the result
        executor: ThreadPoolExecutor for non-blocking file I/O
    
    Returns:
        Status message string
    """
    try:
        output_path = os.path.join(result_dir, os.path.splitext(image_file)[0] + ".md")
        if os.path.exists(output_path):
            return f"⏭ 跳过已存在: {image_file}"

        image_path = os.path.join(image_dir, image_file)
        
        # Read image using executor to avoid blocking
        loop = asyncio.get_event_loop()
        def read_image():
            with open(image_path, "rb") as f:
                return Image.open(io.BytesIO(f.read()))
        
        image = await loop.run_in_executor(executor, read_image)
        
        # Extract content using MinerU
        extracted_blocks = await client.aio_two_step_extract(image)
        result = "\n".join([block['content'] for block in extracted_blocks])
        
        # Save result using executor to avoid blocking
        def write_result():
            with open(output_path, "w", encoding='utf-8') as f:
                f.write(result)
        
        await loop.run_in_executor(executor, write_result)
        
        return f"✓ 成功处理: {image_file}"
    
    except Exception as e:
        return f"✗ 处理失败: {image_file}, 错误: {str(e)}"


async def process_subdirectory(client, subdir, parent_input, base_output, max_workers, executor):
    """
    Process all images in a subdirectory.
    
    Args:
        client: MinerUClient instance
        subdir: Subdirectory name
        parent_input: Parent input directory
        base_output: Base output directory
        max_workers: Maximum concurrent workers
        executor: ThreadPoolExecutor for non-blocking file I/O
    
    Returns:
        Tuple of (total, completed, failed, stats_dict)
    """
    image_dir = os.path.join(parent_input, subdir, 'images')
    result_dir = os.path.join(base_output, subdir)
    os.makedirs(result_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_files = [f for f in os.listdir(image_dir)
                  if os.path.isfile(os.path.join(image_dir, f)) and
                  any(f.lower().endswith(ext) for ext in image_extensions)]
    
    # Check for existing files
    existing_count = 0
    pending_files = []
    for image_file in image_files:
        output_path = os.path.join(result_dir, os.path.splitext(image_file)[0] + ".md")
        if os.path.exists(output_path):
            existing_count += 1
        else:
            pending_files.append(image_file)
    
    if len(pending_files) == 0:
        return len(image_files), 0, 0, {
            'total': len(image_files),
            'existing': existing_count,
            'new': 0,
            'completed': 0,
            'failed': 0
        }
    
    # Create async tasks with semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_workers)
    
    async def bounded_process(image_file):
        async with semaphore:
            return await process_image(client, image_file, image_dir, result_dir, executor)
    
    # Process all pending files
    tasks = [bounded_process(f) for f in pending_files]
    results = await async_tqdm.gather(*tasks, desc=f"Processing {subdir}")
    
    # Count results
    completed = sum(1 for r in results if "✓ 成功处理" in r)
    failed = sum(1 for r in results if "✗" in r)
    
    stats = {
        'total': len(image_files),
        'existing': existing_count,
        'new': len(pending_files),
        'completed': completed,
        'failed': failed
    }
    
    return len(image_files), completed, failed, stats


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='MinerU2.5 OCR inference with concurrent async processing')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Parent directory containing subdirectories with images folders')
    
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for OCR results')
    
    parser.add_argument('--max_workers', type=int,
                       default=64,
                       help='Maximum number of concurrent async tasks')
    
    return parser.parse_args()


async def main():
    args = parse_args()
    
    parent_input = args.input
    base_output = args.output
    
    os.makedirs(base_output, exist_ok=True)
    
    print("Initializing MinerU2.5 inference engine...")
    
    # Initialize async LLM and client
    async_llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model="opendatalab/MinerU2.5-2509-1.2B",
            logits_processors=[MinerULogitsProcessor]
        )
    )
    
    client = MinerUClient(
        backend="vllm-async-engine",
        vllm_async_llm=async_llm,
    )
    
    print("MinerU2.5 engine initialized\n")
    
    # Find subdirs with images
    image_subdirs = []
    for item in os.listdir(parent_input):
        item_path = os.path.join(parent_input, item)
        if os.path.isdir(item_path):
            images_path = os.path.join(item_path, 'images')
            if os.path.isdir(images_path):
                image_subdirs.append(item)
    
    if not image_subdirs:
        print("No subdirectories with 'images' folder found.")
        sys.exit(1)
    
    # Scan all subdirectories
    print(f"Scanning all subdirectories for images...")
    subdir_stats = {}
    total_images = 0
    total_skipped = 0
    
    for subdir in image_subdirs:
        image_dir = os.path.join(parent_input, subdir, 'images')
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        image_files = [f for f in os.listdir(image_dir)
                      if os.path.isfile(os.path.join(image_dir, f)) and
                      any(f.lower().endswith(ext) for ext in image_extensions)]
        
        result_dir = os.path.join(base_output, subdir)
        os.makedirs(result_dir, exist_ok=True)
        
        existing_count = 0
        for image_file in image_files:
            output_path = os.path.join(result_dir, os.path.splitext(image_file)[0] + ".md")
            if os.path.exists(output_path):
                existing_count += 1
                total_skipped += 1
        
        new_count = len(image_files) - existing_count
        total_images += len(image_files)
        
        print(f"Subdirectory {subdir}: {len(image_files)} images, {existing_count} already processed, {new_count} pending")
    
    if total_images - total_skipped == 0:
        print("All files already processed!")
        async_llm.shutdown()
        sys.exit(0)
    
    print(f"\nTotal pending: {total_images - total_skipped} files, already skipped: {total_skipped} files")
    print(f"Starting concurrent async processing (max_workers={args.max_workers})...\n")
    
    # Process all subdirectories
    total_completed = 0
    total_failed = 0
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for subdir in image_subdirs:
            total, completed, failed, stats = await process_subdirectory(
                client, subdir, parent_input, base_output, args.max_workers, executor
            )
            subdir_stats[subdir] = stats
            total_completed += completed
            total_failed += failed
    
    # Print detailed statistics
    print(f"\nProcessing completed statistics (by subdirectory):")
    print("-" * 80)
    for subdir in image_subdirs:
        stats = subdir_stats[subdir]
        print(f"{subdir:30} | Total: {stats['total']:4} | Done: {stats['existing']:4} | Success: {stats['completed']:4} | Failed: {stats['failed']:4}")
    print("-" * 80)
    print(f"{'Total':30} | Total: {total_images:4} | Done: {total_skipped:4} | Success: {total_completed:4} | Failed: {total_failed:4}")
    
    print(f"\nResults saved in: {base_output}")
    
    # Shutdown the async LLM
    async_llm.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
