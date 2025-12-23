#!/usr/bin/env python3
"""
Batch OCR Inference CLI using DeepSeek-OCR with vLLM

Usage:
    python batch_inference.py
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import re

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


def load_images(input_path, limit=None):
    """Load images from input directory"""
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
    
    # Load images
    images = []
    valid_paths = []
    
    for img_path in tqdm(image_paths, desc="Loading images"):
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
    
    return images, valid_paths

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


def create_batch_inputs(images, prompt, crop_mode=True):
    """Create batch inputs for vLLM with image preprocessing"""
    model_input = []
        
    for image in images:        
        model_input.append({
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        })
    
    return model_input


def save_outputs(outputs, image_paths, output_dir):
    """Save model outputs to markdown files"""
    output_dir = expand_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    for output, image in zip(outputs, image_paths):

        content = output.outputs[0].text

        content = clean_formula(content)
        matches_ref, mathes_other = re_match(content)
        for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
            content = content.replace(a_match_other, '').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')
        
        mmd_path = Path(output_dir) / (Path(image).stem + '.md')

        with open(mmd_path, 'w', encoding='utf-8') as afile:
            afile.write(content)
    
    print(f"Saved {len(outputs)} outputs to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch OCR inference using DeepSeek-OCR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process images with default settings
    python batch_inference.py
    
    # Process limited number of images
    python batch_inference.py --limit 10
    
    # Override input/output paths
    python batch_inference.py --input /path/to/images --output /path/to/output
        """
    )
    
    parser.add_argument('--limit', type=int, help='Maximum number of images to process')
    parser.add_argument('--input', type=str, help='Override input directory path')
    parser.add_argument('--output', type=str, help='Override output directory path')
    parser.add_argument('--prompt', type=str, help='Override prompt')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Use configuration constants (or override with command line arguments)
    input_path = args.input if args.input else INPUT_PATH
    output_path = args.output if args.output else OUTPUT_PATH
    prompt = args.prompt if args.prompt else PROMPT
    model_path = MODEL_PATH
    crop_mode = CROP_MODE
    max_concurrency = MAX_CONCURRENCY
    
    # Load images
    images, image_paths = load_images(input_path, limit=args.limit)
    
    if not images:
        print("No images to process. Exiting.")
        return
    
    # Initialize model
    print(f"Initializing model: {model_path}")

    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR",
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
        gpu_memory_utilization=0.9,
        max_num_seqs=max_concurrency,
    )
    
    
    # Create batch inputs
    print("Creating batch inputs...")
    model_input = create_batch_inputs(images, prompt, crop_mode=crop_mode)
    
    # Set sampling parameters
    sampling_param = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        # ngram logit processor args
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
        ),
        skip_special_tokens=False,
    )
    
    # Run inference
    print(f"Running inference on {len(images)} images...")
    outputs = llm.generate(model_input, sampling_param)
    
    # Save outputs
    save_outputs(outputs, image_paths, output_path)
    
    print("Done!")


if __name__ == "__main__":
    main()
