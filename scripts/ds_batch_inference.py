#!/usr/bin/env python3
"""
Batch OCR Inference CLI using DeepSeek-OCR with vLLM and Ray Data

Usage:
    python batch_inference.py
"""

import os
import sys
import argparse
import hashlib
import glob
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import re
import ray
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# Add repo path for imports
repo_path = Path(__file__).parent / "repo/DeepSeek-OCR-master/DeepSeek-OCR-vllm"
sys.path.insert(0, str(repo_path))

from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

# Constants configuration
# TODO: change modes
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 6  # max:9; If your GPU memory is small, it is recommended to set it to 6.
MAX_CONCURRENCY = 100  # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = 64  # image pre-process (resize/padding) workers
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'  # change to your model path

# Ray Data configuration
BATCH_SIZE = 32  # vLLM request batch size (GPU memory bound)
PREFETCH_BATCHES = 2  # Overlap IO/decode with GPU
MAX_IN_FLIGHT = 4  # Max pending actor calls (backpressure)
RAY_CPUS = 8  # CPU cores for Ray Data preprocessing
ENABLE_CACHE = False  # Enable vLLM mm processor cache

# TODO: change INPUT_PATH
# .pdf: run_dpsk_ocr_pdf.py;
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py;
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py

INPUT_PATH = 'data/longbenchv2_img/images'
OUTPUT_PATH = 'data/pred/dpsk'

PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

def expand_path(path):
    """Expand user home directory in path"""
    return os.path.expanduser(path)


@ray.remote(num_gpus=1)
class VLLMWorker:
    """Ray actor for vLLM inference on GPU - decodes images from paths"""
    
    def __init__(self, model_path: str, max_concurrency: int, enable_cache: bool = False):
        """Initialize vLLM model on GPU"""
        from vllm import LLM, SamplingParams
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
        
        print(f"Initializing vLLM worker with model: {model_path}")
        
        self.llm = LLM(
            model=model_path,
            enable_prefix_caching=enable_cache,
            mm_processor_cache_gb=4 if enable_cache else 0,
            logits_processors=[NGramPerReqLogitsProcessor],
            gpu_memory_utilization=0.9,
            max_num_seqs=max_concurrency,
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
            ),
            skip_special_tokens=False,
        )
        self.enable_cache = enable_cache
    
    def generate_batch_from_paths(self, batch_data: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Generate OCR for a batch of image paths (decode inside actor)"""
        model_input = []
        paths = []
        uuids = []
        
        for item in batch_data:
            path = item['path']
            prompt = item['prompt']
            uuid = item.get('uuid')
            
            try:
                # Decode image inside the GPU actor
                img = Image.open(path).convert('RGB')
                
                # Prepare request with optional UUID for caching
                request_data = {
                    "prompt": prompt,
                    "multi_modal_data": {"image": img}
                }
                
                # Add UUID for caching if enabled
                if self.enable_cache and uuid:
                    request_data["multi_modal_uuids"] = {"image": uuid}
                
                model_input.append(request_data)
                paths.append(path)
                uuids.append(uuid)
                
            except Exception as e:
                print(f"Warning: Failed to load {path} in worker: {e}")
                continue
        
        if not model_input:
            return []
        
        # Run inference
        outputs = self.llm.generate(model_input, self.sampling_params)
        
        # Extract results
        results = []
        for output, path in zip(outputs, paths):
            content = output.outputs[0].text
            results.append((path, content))
        
        return results


def create_ray_dataset_paths_only(input_path: str, limit: Optional[int] = None) -> ray.data.Dataset:
    """Create Ray Data pipeline with paths only (no image loading)"""
    input_path = expand_path(input_path)
    
    # Support multiple image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_path, ext)))
        image_paths.extend(glob.glob(os.path.join(input_path, ext.upper())))
    
    image_paths = sorted(set(image_paths))
    
    if limit:
        image_paths = image_paths[:limit]
    
    print(f"Found {len(image_paths)} images in {input_path}")
    
    # Create Ray Dataset with paths and metadata only
    data = []
    for path in image_paths:
        uuid = hashlib.md5(str(path).encode()).hexdigest()
        data.append({
            "path": path,
            "uuid": uuid,
            "prompt": PROMPT,
        })
    
    ds = ray.data.from_pandas(data)
    return ds

def clean_formula(text):

    formula_pattern = r'\\\[(.*?)\\\]'
    
    def process_formula(match):
        formula = match.group(1)

        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        
        formula = formula.strip()
        
        return r'\[' + formula + r'\]'

    cleaned_text = re.sub(formula_pattern, process_formula, text)
    
    return cleaned_text

def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)


    # mathes_image = []
    mathes_other = []
    for a_match in matches:
        mathes_other.append(a_match[0])
    return matches, mathes_other


def process_streaming_batches_with_wait(ds: ray.data.Dataset, workers: List[Any], output_dir: str) -> None:
    """Process batches with bounded dispatch using ray.wait for optimal backpressure"""
    output_dir = expand_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Stream batches from Ray Data (paths only)
    batch_iter = ds.iter_batches(
        batch_size=BATCH_SIZE,
        prefetch_batches=PREFETCH_BATCHES,
        batch_format="numpy",
    )
    
    print(f"Starting streaming processing with {len(workers)} workers...")
    
    # Track pending futures and their metadata
    pending_refs = []  # Just the ObjectRefs
    pending_metadata = []  # Corresponding metadata (batch_size, etc.)
    
    processed_count = 0
    submitted_count = 0
    worker_idx = 0
    start_time = time.time()
    
    # Convert iterator to list for better control
    batches = list(batch_iter)
    total_batches = len(batches)
    total_images = ds.count()
    
    print(f"Processing {total_images} images in {total_batches} batches...")
    
    batch_idx = 0
    
    while batch_idx < total_batches or pending_refs:
        # Submit new batches until we hit the in-flight limit
        while batch_idx < total_batches and len(pending_refs) < MAX_IN_FLIGHT:
            batch = batches[batch_idx]
            
            # Convert batch to list of dicts for worker
            batch_data = []
            for i in range(len(batch["path"])):
                batch_data.append({
                    "path": batch["path"][i],
                    "prompt": batch["prompt"][i],
                    "uuid": batch["uuid"][i],
                })
            
            # Submit to worker (round-robin)
            worker = workers[worker_idx % len(workers)]
            future_ref = worker.generate_batch_from_paths.remote(batch_data)
            
            pending_refs.append(future_ref)
            pending_metadata.append({
                "batch_size": len(batch_data),
                "batch_idx": batch_idx,
                "worker_idx": worker_idx % len(workers),
            })
            
            submitted_count += len(batch_data)
            worker_idx += 1
            batch_idx += 1
            
            print(f"Submitted batch {batch_idx}/{total_batches} ({len(batch_data)} images)")
        
        # Wait for at least one batch to complete
        if pending_refs:
            ready_refs, remaining_refs = ray.wait(pending_refs, num_returns=1, timeout=0.1)
            
            # Process completed batches
            for ref in ready_refs:
                # Find the metadata for this ref
                idx = pending_refs.index(ref)
                metadata = pending_metadata[idx]
                
                try:
                    results = ray.get(ref)
                    save_batch_results(results, output_dir)
                    processed_count += len(results)
                    
                    # Print progress with metrics
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    print(f"Completed batch {metadata['batch_idx']+1}/{total_batches}. "
                          f"Processed {processed_count}/{total_images} images ({rate:.2f} img/s)")
                    
                except Exception as e:
                    print(f"Error processing batch {metadata['batch_idx']}: {e}")
                
                # Remove from pending lists
                pending_refs.remove(ref)
                pending_metadata.pop(idx)
        
        # Small sleep to prevent busy waiting
        if pending_refs and batch_idx >= total_batches:
            time.sleep(0.01)
    
    total_time = time.time() - start_time
    print(f"Completed processing {processed_count} images in {total_time:.2f}s ({processed_count/total_time:.2f} img/s)")


def save_batch_results(results: List[Tuple[str, str]], output_dir: str) -> None:
    """Save batch results to markdown files"""
    for path, content in results:
        try:
            # Clean content
            content = clean_formula(content)
            matches_ref, mathes_other = re_match(content)
            for a_match_other in mathes_other:
                content = content.replace(a_match_other, '').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')
            
            # Save to file
            mmd_path = Path(output_dir) / (Path(path).stem + '.md')
            with open(mmd_path, 'w', encoding='utf-8') as afile:
                afile.write(content)
                
        except Exception as e:
            print(f"Error saving result for {path}: {e}")


def process_single_gpu_baseline(input_path: str, output_path: str, limit: Optional[int] = None) -> None:
    """Single GPU baseline without Ray - chunked processing"""
    print("Running single GPU baseline (no Ray)...")
    
    # Get image paths
    input_path = expand_path(input_path)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_path, ext)))
        image_paths.extend(glob.glob(os.path.join(input_path, ext.upper())))
    
    image_paths = sorted(set(image_paths))
    if limit:
        image_paths = image_paths[:limit]
    
    print(f"Found {len(image_paths)} images")
    
    # Initialize vLLM
    print("Initializing vLLM...")
    llm = LLM(
        model=MODEL_PATH,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
        gpu_memory_utilization=0.9,
        max_num_seqs=MAX_CONCURRENCY,
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},
        ),
        skip_special_tokens=False,
    )
    
    # Process in chunks
    chunk_size = BATCH_SIZE
    output_dir = expand_path(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    start_time = time.time()
    
    for i in range(0, len(image_paths), chunk_size):
        chunk_paths = image_paths[i:i + chunk_size]
        
        # Load images for this chunk
        images = []
        valid_paths = []
        for path in chunk_paths:
            try:
                img = Image.open(path).convert('RGB')
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue
        
        if not images:
            continue
        
        # Create batch inputs
        model_input = []
        for img in images:
            model_input.append({
                "prompt": PROMPT,
                "multi_modal_data": {"image": img}
            })
        
        # Generate
        outputs = llm.generate(model_input, sampling_params)
        
        # Save results
        for output, path in zip(outputs, valid_paths):
            content = output.outputs[0].text
            content = clean_formula(content)
            matches_ref, mathes_other = re_match(content)
            for a_match_other in mathes_other:
                content = content.replace(a_match_other, '').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')
            
            mmd_path = Path(output_dir) / (Path(path).stem + '.md')
            with open(mmd_path, 'w', encoding='utf-8') as afile:
                afile.write(content)
        
        processed_count += len(images)
        
        # Print progress and metrics
        elapsed = time.time() - start_time
        rate = processed_count / elapsed if elapsed > 0 else 0
        print(f"Processed {processed_count}/{len(image_paths)} images ({rate:.2f} img/s)")
    
    total_time = time.time() - start_time
    print(f"Completed! Processed {processed_count} images in {total_time:.2f}s ({processed_count/total_time:.2f} img/s)")





def main():
    parser = argparse.ArgumentParser(
        description='Batch OCR inference using DeepSeek-OCR with optimized scaling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single GPU baseline (no Ray) - best for 1 GPU
    python batch_inference.py --baseline
    
    # Multi-GPU with Ray Data - scales from 1 to N GPUs
    python batch_inference.py --num-workers 2
    
    # Tune performance parameters
    python batch_inference.py --batch-size 64 --max-in-flight 8
    
    # Enable caching for repeated images
    python batch_inference.py --enable-cache
        """
    )
    
    parser.add_argument('--limit', type=int, help='Maximum number of images to process')
    parser.add_argument('--input', type=str, help='Override input directory path')
    parser.add_argument('--output', type=str, help='Override output directory path')
    parser.add_argument('--prompt', type=str, help='Override prompt')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID (default: 0)')
    
    # Mode selection
    parser.add_argument('--baseline', action='store_true', 
                       help='Use single GPU baseline (no Ray) - recommended for 1 GPU')
    parser.add_argument('--num-workers', type=int, default=1,
                       help='Number of GPU workers for multi-GPU mode (default: 1)')
    
    # Performance tuning parameters
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, 
                       help=f'Batch size for vLLM (default: {BATCH_SIZE})')
    parser.add_argument('--prefetch-batches', type=int, default=PREFETCH_BATCHES, 
                       help=f'Prefetch batches (default: {PREFETCH_BATCHES})')
    parser.add_argument('--max-in-flight', type=int, default=MAX_IN_FLIGHT, 
                       help=f'Max in-flight requests (default: {MAX_IN_FLIGHT})')
    parser.add_argument('--ray-cpus', type=int, default=RAY_CPUS, 
                       help=f'CPU cores for Ray Data (default: {RAY_CPUS})')
    parser.add_argument('--enable-cache', action='store_true', 
                       help='Enable vLLM mm processor cache (useful for repeated images)')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Update global config from args
    global BATCH_SIZE, PREFETCH_BATCHES, MAX_IN_FLIGHT, RAY_CPUS, ENABLE_CACHE, PROMPT
    BATCH_SIZE = args.batch_size
    PREFETCH_BATCHES = args.prefetch_batches
    MAX_IN_FLIGHT = args.max_in_flight
    RAY_CPUS = args.ray_cpus
    ENABLE_CACHE = args.enable_cache
    
    # Use configuration constants (or override with command line arguments)
    input_path = args.input if args.input else INPUT_PATH
    output_path = args.output if args.output else OUTPUT_PATH
    if args.prompt:
        PROMPT = args.prompt
    
    print(f"Configuration: batch_size={BATCH_SIZE}, max_in_flight={MAX_IN_FLIGHT}, "
          f"enable_cache={ENABLE_CACHE}")
    
    # Choose processing mode
    if args.baseline or args.num_workers == 1:
        print("Running single GPU baseline (no Ray)...")
        process_single_gpu_baseline(input_path, output_path, limit=args.limit)
    else:
        print(f"Running multi-GPU mode with {args.num_workers} workers...")
        \n        # Initialize Ray\n        if not ray.is_initialized():\n            ray.init()\n        \n        print(f\"Ray initialized with {ray.cluster_resources()}\")\n        \n        # Create GPU workers\n        workers = []\n        for i in range(args.num_workers):\n            worker = VLLMWorker.remote(\n                model_path=MODEL_PATH,\n                max_concurrency=MAX_CONCURRENCY,\n                enable_cache=ENABLE_CACHE,\n            )\n            workers.append(worker)\n        \n        print(f\"Created {len(workers)} GPU workers\")\n        \n        # Create Ray Data pipeline (paths only)\n        ds = create_ray_dataset_paths_only(input_path, limit=args.limit)\n        \n        if ds.count() == 0:\n            print(\"No images to process. Exiting.\")\n            return\n        \n        # Process with streaming batches using ray.wait\n        process_streaming_batches_with_wait(ds, workers, output_path)\n        \n        # Shutdown Ray\n        ray.shutdown()\n    \n    print(\"Done!\")


if __name__ == "__main__":
    main()
