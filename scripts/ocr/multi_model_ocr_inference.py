#!/usr/bin/env python3
"""
Multi-model OCR inference script with concurrent processing.
Supports: DeepSeek-OCR, Qwen3-VL, InternVL3.5, and Kimi-VL

To start vLLM server for a specific model:

# DeepSeek-OCR (Multiple GPUs):
vllm serve deepseek-ai/DeepSeek-OCR \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 8 \
    --dtype auto \
    --trust-remote-code \
    --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0

# Qwen3-VL-8B:
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 8 \
    --dtype auto \
    --trust-remote-code

# InternVL3.5-8B:
vllm serve OpenGVLab/InternVL3_5-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 8 \
    --dtype auto \
    --trust-remote-code

# Kimi-VL-A3B:
conda run -n test vllm serve moonshotai/Kimi-VL-A3B-Instruct \
  --served-model-name kimi-vl \
  --trust-remote-code \
  --data-parallel-size 8 \
  --max-num-batched-tokens 32768 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 64}'

# GLM-4.1V-9B:
vllm serve zai-org/GLM-4.1V-9B-Thinking \
    --data-parallel-size 8 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm

# Glyph:
vllm serve zai-org/Glyph \
    --data-parallel-size 8 \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --reasoning-parser glm45 \
    --limit-mm-per-prompt.video 0

Then run this script:
micromamba run -n test python scripts/ocr/multi_model_ocr_inference.py \
    --input data/ocr \
    --output data/pred/ds \
    --model deepseek-ocr \
    --base_url http://localhost:8000/v1 \
    --max_workers 1024
"""

from openai import OpenAI, APIConnectionError
import base64
import os
import sys
import argparse
import concurrent.futures
from tqdm import tqdm
import re
import json
import glob


def encode_image(image_path):
    """
    Encode the image file to base64 string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_model_config(config_path: str) -> dict:
    """
    Load model configuration from a JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
    
    Returns:
        Dictionary containing model configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ['model_name', 'prompt', 'max_tokens']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in config file: {config_path}")
    
    return config


def process_image(client, image_file, image_dir, result_dir, model_name, model_config, presence_penalty=0.0):
    """
    Generic image processing function that works with any model configuration.
    
    Args:
        client: OpenAI client instance
        image_file: Name of the image file to process
        image_dir: Directory containing the image
        result_dir: Directory to save the result
        model_name: Name of the model (for DeepSeek-OCR specific cleaning)
        model_config: Model configuration dict with model_name, prompt, max_tokens, etc.
        presence_penalty: Presence penalty parameter
    """
    try:
        output_path = os.path.join(result_dir, os.path.splitext(image_file)[0] + ".md")
        if os.path.exists(output_path):
            return f"⏭ 跳过已存在: {image_file}"

        image_path = os.path.join(image_dir, image_file)
        base64_image = encode_image(image_path)
        data_url = f"data:image/jpeg;base64,{base64_image}"

        # Build request parameters
        request_params = {
            'model': model_config['model_name'],
            'messages': [{
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': model_config['prompt'],
                    },
                    {
                        'type': 'image_url',
                        'image_url': {'url': data_url},
                    }
                ],
            }],
            'max_tokens': model_config['max_tokens'],
        }

        # Add temperature if specified
        if model_config['temperature'] is not None:
            request_params['temperature'] = model_config['temperature']

        # Add timeout if specified
        if model_config['timeout'] is not None:
            request_params['timeout'] = model_config['timeout']

        # Add presence penalty
        request_params['presence_penalty'] = presence_penalty

        # Add extra_body if specified
        if model_config['extra_body'] is not None:
            request_params['extra_body'] = model_config['extra_body']

        # Make API request
        response = client.chat.completions.create(**request_params)
        result = response.choices[0].message.content

        # Clean result for DeepSeek-OCR
        if 'deepseek' in model_name.lower() and 'ocr' in model_name.lower():
            result = clean_ocr_content(result)
        # Clean result for Glyph (remove thinking process)
        elif 'glyph' in model_name.lower():
            result = clean_glyph_content(result)

        # Save result
        with open(output_path, "w", encoding='utf-8') as f:
            print(result, file=f)

        return f"✓ 成功处理: {image_file}"
    except APIConnectionError as e:
        return f"✗ 连接超时: {image_file}, 错误: {str(e)}"
    except Exception as e:
        return f"✗ 处理失败: {image_file}, 错误: {str(e)}"


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


def re_match(text):
    """
    Extract matches from OCR output
    """
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    filtered_matches = []
    for a_match in matches:
        filtered_matches.append(a_match[0])
    return matches, filtered_matches


def clean_ocr_content(content: str) -> str:
    """
    Clean and normalize OCR output (DeepSeek-OCR specific)
    """
    content = clean_formula(content)
    matches_ref, filtered_matches = re_match(content)
    for a_match in filtered_matches:
        content = content.replace(a_match, '')
    content = content.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')
    return content.strip()


def clean_glyph_content(content: str) -> str:
    """
    Clean and normalize OCR output for Glyph model.
    Removes the thinking process by keeping only content after </think> tag.
    """
    # Find the </think> tag and extract everything after it
    think_tag = "</think>"
    if think_tag in content:
        # Get everything after </think>
        content = content.split(think_tag, 1)[1]
    
    # Clean up extra whitespace
    content = content.strip()
    return content


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Multi-model OCR inference with concurrent processing')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Parent directory containing subdirectories with images folders')
    
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for OCR results')
    
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to model configuration JSON file (e.g., config/ocr/deepseek-ocr.json)')
    
    parser.add_argument('--base_url', type=str,
                       default='http://localhost:8000/v1',
                       help='API base URL')
    
    parser.add_argument('--api_key', type=str,
                       default=None,
                       help='API key (optional for local vLLM)')
    
    parser.add_argument('--max_workers', type=int,
                       default=512,
                       help='Number of concurrent workers')
    
    parser.add_argument('--presence_penalty', type=float,
                       default=None,
                       help='Presence penalty for repetition control (0.0 to 2.0). Overrides config file value if specified.')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    parent_input = args.input
    base_output = args.output
    
    os.makedirs(base_output, exist_ok=True)

    # Load model configuration from JSON file
    try:
        model_config = load_model_config(args.config)
        model_name = model_config['model_name']
        print(f"Loaded configuration from: {args.config}")
        print(f"Model: {model_name}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Determine presence_penalty: CLI overrides config, default to 0.0
    presence_penalty = args.presence_penalty if args.presence_penalty is not None else model_config.get('presence_penalty', 0.0)
    print(f"Using presence_penalty: {presence_penalty}")

    # Create OpenAI client
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key if args.api_key else "dummy",
    )

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

    # First step: Scan all subdirectories and build unified task queue
    print(f"\nScanning all subdirectories for images...")
    all_tasks = []
    subdir_stats = {}
    total_images = 0
    total_skipped = 0

    for subdir in image_subdirs:
        image_dir = os.path.join(parent_input, subdir, 'images')
        result_dir = os.path.join(base_output, subdir)
        os.makedirs(result_dir, exist_ok=True)

        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        image_files = [f for f in os.listdir(image_dir)
                      if os.path.isfile(os.path.join(image_dir, f)) and
                      any(f.lower().endswith(ext) for ext in image_extensions)]
        
        total_images += len(image_files)

        # Check for existing files, only add new files to task queue
        existing_count = 0
        new_count = 0
        for image_file in image_files:
            output_path = os.path.join(result_dir, os.path.splitext(image_file)[0] + ".md")
            if os.path.exists(output_path):
                existing_count += 1
                total_skipped += 1
            else:
                all_tasks.append((image_file, image_dir, result_dir, subdir))
                new_count += 1

        subdir_stats[subdir] = {
            'total': len(image_files),
            'existing': existing_count,
            'new': new_count,
            'completed': 0,
            'failed': 0
        }
        
        print(f"Subdirectory {subdir}: {len(image_files)} images, {existing_count} already processed, {new_count} pending")

    if len(all_tasks) == 0:
        print("All files already processed!")
        sys.exit(0)

    print(f"\nTotal pending: {len(all_tasks)} files, already skipped: {total_skipped} files")
    print(f"Starting concurrent processing (max_workers={args.max_workers})...\n")
    
    # Second step: Submit all tasks to thread pool
    total_completed = 0
    total_failed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_image, client, image_file, image_dir, result_dir, model_name, model_config, presence_penalty): (image_file, subdir)
            for image_file, image_dir, result_dir, subdir in all_tasks
        }
        
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_tasks), desc="Overall progress"):
            try:
                result = future.result()
                image_file, subdir = futures[future]
                results.append((result, subdir))
                
                if "✓ 成功处理" in result:
                    total_completed += 1
                    subdir_stats[subdir]['completed'] += 1
                elif "✗" in result:
                    total_failed += 1
                    subdir_stats[subdir]['failed'] += 1
            except Exception as exc:
                total_failed += 1
                image_file, subdir = futures[future]
                subdir_stats[subdir]['failed'] += 1
                results.append((f"✗ Exception: {str(exc)}", subdir))
    
    # Third step: Print detailed statistics
    print(f"\nProcessing completed statistics (by subdirectory):")
    print("-" * 80)
    for subdir in image_subdirs:
        stats = subdir_stats[subdir]
        print(f"{subdir:30} | Total: {stats['total']:4} | Done: {stats['existing']:4} | Success: {stats['completed']:4} | Failed: {stats['failed']:4}")
    print("-" * 80)
    print(f"{'Total':30} | Total: {total_images:4} | Done: {total_skipped:4} | Success: {total_completed:4} | Failed: {total_failed:4}")
    
    print(f"\nResults saved in: {base_output}")
    
    # Print failure details if any
    if total_failed > 0:
        print(f"\nFailure details (total {total_failed}):")
        for result, subdir in results:
            if "✗" in result:
                print(f"  [{subdir}] {result}")
