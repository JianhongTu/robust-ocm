#!/usr/bin/env python3
"""
Multi-model Multiple Choice VQA inference script with concurrent processing.
Evaluates vision-language models on multiple choice questions from longbenchv2 dataset.

To start vLLM server for a specific model:

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

Then run this script:
micromamba run -n test python scripts/vqa/mc_inference.py \
    --data-json data/longbenchv2/data.json \
    --images-dir data/vqa \
    --output results/qwen3 \
    --config config/vqa/qwen3-vl.json \
    --base_url http://localhost:8000/v1 \
    --max_workers 128

The script will automatically:
- Load data.json from the specified path
- Scan images-dir for subdirectories with images/ folders
- Process each subdirectory (different perturbations of the same data)
- Save results to results/qwen3/{subdir}_mc_eval.json
"""

from openai import OpenAI, APIConnectionError
import base64
import os
import sys
import argparse
import concurrent.futures
from tqdm import tqdm
import json
import tiktoken
from typing import Dict, List, Any, Tuple


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
    required_fields = ['model_name', 'max_tokens']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in config file: {config_path}")
    
    return config


def load_mc_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load multiple choice dataset from JSON file.
    
    Args:
        data_path: Path to the data.json file
    
    Returns:
        List of multiple choice instances
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def load_blacklist(blacklist_path: str) -> set:
    """
    Load blacklist of instance IDs to skip.
    
    Args:
        blacklist_path: Path to the blacklist file (one ID per line)
    
    Returns:
        Set of instance IDs to skip
    """
    if not os.path.exists(blacklist_path):
        raise FileNotFoundError(f"Blacklist file not found: {blacklist_path}")
    
    blacklist_ids = set()
    with open(blacklist_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                blacklist_ids.add(line)
    
    return blacklist_ids


def format_mc_prompt(question: str, choices: Dict[str, str], prompt_template: str = None) -> str:
    """
    Format the multiple choice question prompt using a template if provided.
    
    Args:
        question: The question to ask
        choices: Dictionary with keys 'A', 'B', 'C', 'D' and their corresponding text
        prompt_template: Optional template string with {question} and {choices} placeholders
    
    Returns:
        Formatted prompt string
    """
    # Format choices
    choices_text = "\n".join([f"{key}. {value}" for key, value in sorted(choices.items())])
    
    if prompt_template:
        return prompt_template.format(question=question, choices=choices_text)
    else:
        # Default prompt format
        return f"{question}\n\n{choices_text}\n\n"


def clean_thinking_tokens(model_output: str) -> str:
    """
    Remove thinking tokens from model output.
    If </think> tag is present, truncate everything before and including it.
    
    Args:
        model_output: The model's output string
    
    Returns:
        Cleaned output string
    """
    if "</think>" in model_output:
        # Find the position of </think> and truncate everything before it
        think_end_pos = model_output.find("</think>") + len("</think>")
        return model_output[think_end_pos:].strip()
    return model_output


def extract_answer(model_output: str) -> str:
    """
    Extract the answer choice (A, B, C, or D) from model output.
    
    Args:
        model_output: The model's output string
    
    Returns:
        Extracted answer choice or empty string if not found
    """
    model_output = model_output.strip().upper()
    
    # Try to find A, B, C, or D in the output
    # Look for patterns like "A", "A.", "A)", "(A)", "Answer: A", etc.
    import re
    
    # Pattern 1: First letter A-D
    match = re.search(r'\b([A-D])\b', model_output)
    if match:
        return match.group(1)
    
    # Pattern 2: In parentheses or with punctuation
    match = re.search(r'[(（]([A-D])[)）]', model_output)
    if match:
        return match.group(1)
    
    # Pattern 3: After keywords
    match = re.search(r'(?:answer|choice|option)[:\s]*([A-D])', model_output, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # If nothing found, return the first character if it's A-D
    if len(model_output) > 0 and model_output[0] in 'ABCD':
        return model_output[0]
    
    return ""


def check_answer_correct(model_output: str, ground_truth: str) -> bool:
    """
    Check if the model output contains the correct answer.
    
    Args:
        model_output: The model's output string
        ground_truth: The expected answer (A, B, C, or D)
    
    Returns:
        True if the answer is correct, False otherwise
    """
    extracted = extract_answer(model_output)
    return extracted == ground_truth.upper()


def process_question(
    client: OpenAI,
    instance_id: str,
    question: str,
    choices: Dict[str, str],
    answer: str,
    image_paths: List[str],
    images_dir: str,
    model_config: dict,
    presence_penalty: float = 0.0,
    context: str = None,
    text_only_mode: bool = False
) -> Dict[str, Any]:
    """
    Process a single multiple choice question and evaluate the answer.
    
    Args:
        client: OpenAI client instance
        instance_id: ID of the instance
        question: The question to ask
        choices: Dictionary of answer choices
        answer: The ground truth answer
        image_paths: List of image file names for this instance
        images_dir: Directory containing the images
        model_config: Model configuration dict
        presence_penalty: Presence penalty parameter
        context: Raw text context (used in text-only mode)
        text_only_mode: If True, use context text instead of images
    
    Returns:
        Dictionary with question results
    """
    try:
        # Format the prompt
        prompt = format_mc_prompt(question, choices, model_config.get('prompt_template'))
        
        # Build message content based on mode
        if text_only_mode:
            # Text-only mode: include context text instead of images
            if context:
                full_prompt = f"Context:\n{context}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            message_content = [{
                'type': 'text',
                'text': full_prompt,
            }]
        else:
            # Image mode: prepare image data URLs
            image_contents = []
            for image_path in image_paths:
                full_path = os.path.join(images_dir, image_path)
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"Image not found: {full_path}")
                
                base64_image = encode_image(full_path)
                data_url = f"data:image/jpeg;base64,{base64_image}"
                image_contents.append({
                    'type': 'image_url',
                    'image_url': {'url': data_url},
                })
            
            # Build message content (images first, then question)
            message_content = image_contents + [{
                'type': 'text',
                'text': prompt,
            }]
        
        # Build request parameters
        request_params = {
            'model': model_config['model_name'],
            'messages': [{
                'role': 'user',
                'content': message_content,
            }],
            'max_tokens': model_config['max_tokens'],
        }
        
        # Add temperature if specified
        if model_config.get('temperature') is not None:
            request_params['temperature'] = model_config['temperature']
        
        # Add timeout if specified
        if model_config.get('timeout') is not None:
            request_params['timeout'] = model_config['timeout']
        
        # Add presence penalty
        request_params['presence_penalty'] = presence_penalty
        
        # Add extra_body if specified
        if model_config.get('extra_body') is not None:
            request_params['extra_body'] = model_config['extra_body']
        
        # Make API request
        response = client.chat.completions.create(**request_params)
        model_output = response.choices[0].message.content
        
        # Clean thinking tokens if present
        cleaned_output = clean_thinking_tokens(model_output)
        
        # Extract answer and check if correct
        extracted_answer = extract_answer(cleaned_output)
        is_correct = check_answer_correct(cleaned_output, answer)
        
        return {
            'instance_id': instance_id,
            'question': question,
            'choices': choices,
            'ground_truth': answer,
            'prompt': prompt,
            'model_output': model_output,
            'extracted_answer': extracted_answer,
            'correct': is_correct,
            'num_images': len(image_paths),
            'status': 'success'
        }
    
    except APIConnectionError as e:
        return {
            'instance_id': instance_id,
            'question': question,
            'choices': choices,
            'ground_truth': answer,
            'model_output': None,
            'extracted_answer': None,
            'correct': False,
            'num_images': len(image_paths),
            'status': 'connection_error',
            'error': str(e)
        }
    except Exception as e:
        return {
            'instance_id': instance_id,
            'question': question,
            'choices': choices,
            'ground_truth': answer,
            'model_output': None,
            'extracted_answer': None,
            'correct': False,
            'num_images': len(image_paths),
            'status': 'error',
            'error': str(e)
        }


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Multi-model Multiple Choice VQA inference with concurrent processing')
    
    parser.add_argument('--data-json', type=str, required=True,
                       help='Path to data.json file (e.g., data/longbenchv2/data.json)')
    
    parser.add_argument('--images-dir', type=str,
                       default=None,
                       help='Parent directory containing subdirectories with images/ folders (e.g., data/vqa). Not required in text-only mode.')
    
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for result JSON files')
    
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to model configuration JSON file (e.g., config/vqa/qwen3-vl.json)')
    
    parser.add_argument('--base_url', type=str,
                       default='http://localhost:8000/v1',
                       help='API base URL')
    
    parser.add_argument('--api_key', type=str,
                       default=None,
                       help='API key (optional for local vLLM)')
    
    parser.add_argument('--max_workers', type=int,
                       default=128,
                       help='Number of concurrent workers')
    
    parser.add_argument('--presence_penalty', type=float,
                       default=None,
                       help='Presence penalty for repetition control (0.0 to 2.0). Overrides config file value if specified.')
    
    parser.add_argument('--limit', type=int,
                       default=None,
                       help='Limit processing to N instances per subdirectory (for testing)')
    
    parser.add_argument('--blacklist', type=str,
                       default=None,
                       help='Path to blacklist file containing instance IDs to skip (one per line)')
    
    parser.add_argument('--text-only', action='store_true',
                       help='Use text context instead of images for inference (for baseline comparison)')
    
    return parser.parse_args()


def categorize_context_length(token_count: int) -> str:
    """
    Categorize context length into bins.
    
    Args:
        token_count: Number of tokens in context
    
    Returns:
        Category string (e.g., '8k-16k')
    """
    if token_count < 8000:
        return '<8k'
    elif token_count < 16000:
        return '8k-16k'
    elif token_count < 24000:
        return '16k-24k'
    elif token_count < 32000:
        return '24k-32k'
    elif token_count < 48000:
        return '32k-48k'
    elif token_count < 64000:
        return '48k-64k'
    else:
        return '>=64k'


def analyze_by_context_length(results: List[Dict[str, Any]], mc_data: List[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
    """
    Analyze accuracy by context length.
    
    Args:
        results: List of result dictionaries from inference
        mc_data: Original MC dataset
        tokenizer: Tiktoken tokenizer for counting tokens
    
    Returns:
        Dictionary with analysis results
    """
    # Create instance lookup
    instance_lookup = {inst['_id']: inst for inst in mc_data}
    
    # Initialize stats structure
    length_bins = ['<8k', '8k-16k', '16k-24k', '24k-32k', '32k-48k', '48k-64k', '>=64k']
    
    # Stats by length
    length_stats = {length: {'total': 0, 'correct': 0} for length in length_bins}
    
    # Tokenize contexts (cache to avoid re-tokenizing)
    context_token_counts = {}
    for inst in mc_data:
        inst_id = inst['_id']
        context = inst.get('context', '')
        if isinstance(context, str):
            token_count = len(tokenizer.encode(context, disallowed_special=()))
            context_token_counts[inst_id] = token_count
    
    # Process each result
    for result in results:
        inst_id = result['instance_id']
        is_correct = result.get('correct', False)
        
        # Get instance and token count
        instance = instance_lookup.get(inst_id)
        if not instance:
            continue
        
        token_count = context_token_counts.get(inst_id, 0)
        length_bin = categorize_context_length(token_count)
        
        # Update stats
        length_stats[length_bin]['total'] += 1
        if is_correct:
            length_stats[length_bin]['correct'] += 1
    
    # Calculate accuracies
    for length in length_bins:
        total = length_stats[length]['total']
        if total > 0:
            length_stats[length]['accuracy'] = length_stats[length]['correct'] / total * 100
        else:
            length_stats[length]['accuracy'] = None
    
    return {
        'by_context_length': length_stats
    }


def analyze_by_difficulty_and_length(results: List[Dict[str, Any]], mc_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze accuracy by difficulty and length.
    
    Args:
        results: List of result dictionaries from inference
        mc_data: Original MC dataset
    
    Returns:
        Dictionary with analysis results
    """
    # Create instance lookup
    instance_lookup = {inst['_id']: inst for inst in mc_data}
    
    # Initialize stats structure
    difficulty_levels = ['easy', 'medium', 'hard']
    length_levels = ['short', 'medium', 'long']
    
    # Stats by difficulty
    difficulty_stats = {diff: {'total': 0, 'correct': 0} for diff in difficulty_levels}
    
    # Stats by length
    length_stats = {length: {'total': 0, 'correct': 0} for length in length_levels}
    
    # Stats by difficulty x length
    combined_stats = {
        diff: {length: {'total': 0, 'correct': 0} for length in length_levels}
        for diff in difficulty_levels
    }
    
    # Process each result
    for result in results:
        inst_id = result['instance_id']
        is_correct = result.get('correct', False)
        
        # Get instance
        instance = instance_lookup.get(inst_id)
        if not instance:
            continue
        
        difficulty = instance.get('difficulty', 'unknown')
        length = instance.get('length', 'unknown')
        
        if difficulty not in difficulty_levels or length not in length_levels:
            continue
        
        # Update stats
        difficulty_stats[difficulty]['total'] += 1
        if is_correct:
            difficulty_stats[difficulty]['correct'] += 1
        
        length_stats[length]['total'] += 1
        if is_correct:
            length_stats[length]['correct'] += 1
        
        combined_stats[difficulty][length]['total'] += 1
        if is_correct:
            combined_stats[difficulty][length]['correct'] += 1
    
    # Calculate accuracies
    for diff in difficulty_levels:
        total = difficulty_stats[diff]['total']
        if total > 0:
            difficulty_stats[diff]['accuracy'] = difficulty_stats[diff]['correct'] / total * 100
        else:
            difficulty_stats[diff]['accuracy'] = None
    
    for length in length_levels:
        total = length_stats[length]['total']
        if total > 0:
            length_stats[length]['accuracy'] = length_stats[length]['correct'] / total * 100
        else:
            length_stats[length]['accuracy'] = None
    
    for diff in difficulty_levels:
        for length in length_levels:
            total = combined_stats[diff][length]['total']
            if total > 0:
                combined_stats[diff][length]['accuracy'] = combined_stats[diff][length]['correct'] / total * 100
            else:
                combined_stats[diff][length]['accuracy'] = None
    
    return {
        'by_difficulty': difficulty_stats,
        'by_length': length_stats,
        'by_difficulty_and_length': combined_stats
    }


if __name__ == "__main__":
    args = parse_args()
    
    # Validate arguments
    if not args.text_only and not args.images_dir:
        print("Error: --images-dir is required unless --text-only mode is enabled")
        sys.exit(1)
    
    data_json_path = args.data_json
    images_parent_dir = args.images_dir
    base_output = args.output
    
    # Create output directory
    os.makedirs(base_output, exist_ok=True)

    # Load model configuration
    try:
        model_config = load_model_config(args.config)
        model_name = model_config['model_name']
        print(f"Loaded configuration from: {args.config}")
        print(f"Model: {model_name}")
        print(f"Mode: {'Text-only (using raw context)' if args.text_only else 'Image-based (using OCR images)'}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Determine presence_penalty: CLI overrides config, default to 0.0
    presence_penalty = args.presence_penalty if args.presence_penalty is not None else model_config.get('presence_penalty', 0.0)
    print(f"Using presence_penalty: {presence_penalty}")
    
    # Load MC data
    if not os.path.exists(data_json_path):
        print(f"Error: data.json not found at {data_json_path}")
        sys.exit(1)
    
    try:
        mc_data = load_mc_data(data_json_path)
        print(f"Loaded {len(mc_data)} instances from: {data_json_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Apply blacklist if specified
    if args.blacklist is not None:
        try:
            blacklist_ids = load_blacklist(args.blacklist)
            print(f"Loaded blacklist with {len(blacklist_ids)} IDs from: {args.blacklist}")
            
            original_count = len(mc_data)
            mc_data = [inst for inst in mc_data if inst['_id'] not in blacklist_ids]
            filtered_count = original_count - len(mc_data)
            
            print(f"Filtered out {filtered_count} blacklisted instances ({len(mc_data)} remaining)")
        except Exception as e:
            print(f"Error loading blacklist: {e}")
            sys.exit(1)
    
    # Apply limit if specified
    if args.limit is not None:
        mc_data = mc_data[:args.limit]
        print(f"Limited to {len(mc_data)} instances for testing")
    
    # Create OpenAI client
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key if args.api_key else "dummy",
    )
    
    # Find subdirectories with images/ or run in text-only mode
    if args.text_only:
        # Text-only mode: process as a single run without subdirectories
        print(f"\nRunning in text-only mode (no images, using raw context)...")
        subdirs_to_process = ['text_only']
    else:
        # Find subdirectories with images/
        print(f"\nScanning for subdirectories with images/ in {images_parent_dir}...")
        subdirs_to_process = []
        
        for item in os.listdir(images_parent_dir):
            item_path = os.path.join(images_parent_dir, item)
            if os.path.isdir(item_path):
                images_path = os.path.join(item_path, 'images')
                
                if os.path.isdir(images_path):
                    subdirs_to_process.append(item)
    
    if not subdirs_to_process:
        print("No subdirectories with 'images/' folder found.")
        sys.exit(1)
    
    print(f"Found {len(subdirs_to_process)} subdirectories to process: {subdirs_to_process}")
    
    # Process each subdirectory
    for subdir_idx, subdir in enumerate(subdirs_to_process, 1):
        print(f"\n{'='*80}")
        print(f"Processing subdirectory [{subdir_idx}/{len(subdirs_to_process)}]: {subdir}")
        print(f"{'='*80}")
        
        if args.text_only:
            images_dir = None  # No images directory needed
            output_path = os.path.join(base_output, "text_only_mc_eval.json")
        else:
            images_dir = os.path.join(images_parent_dir, subdir, 'images')
            output_path = os.path.join(base_output, f"{subdir}_mc_eval.json")
        
        # Check if already processed
        if os.path.exists(output_path):
            print(f"⏭  Results already exist at {output_path}, skipping...")
            continue
        
        # Build task queue for this subdirectory
        print(f"\nBuilding task queue for {subdir}...")
        all_tasks = []
        total_questions = 0
        
        for instance in mc_data:
            instance_id = instance['_id']
            question = instance.get('question', '')
            
            # Get choices
            choices = {
                'A': instance.get('choice_A', ''),
                'B': instance.get('choice_B', ''),
                'C': instance.get('choice_C', ''),
                'D': instance.get('choice_D', '')
            }
            
            answer = instance.get('answer', '')
            context = instance.get('context', '')
            
            # Get image paths for this instance (skip check in text-only mode)
            if args.text_only:
                image_paths = []  # No images needed in text-only mode
            else:
                image_pattern = f"{instance_id}_*.png"
                import glob
                image_paths = sorted([
                    os.path.basename(p) 
                    for p in glob.glob(os.path.join(images_dir, image_pattern))
                ])
                
                if not image_paths:
                    print(f"Warning: No images found for instance {instance_id}")
                    continue
            
            # Add task for this instance
            all_tasks.append((instance_id, question, choices, answer, image_paths, context))
            total_questions += 1
        
        print(f"Total questions to process: {total_questions}")
        print(f"Starting concurrent processing (max_workers={args.max_workers})...\n")
        
        # Process all questions concurrently
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    process_question,
                    client,
                    instance_id,
                    question,
                    choices,
                    answer,
                    image_paths,
                    images_dir,
                    model_config,
                    presence_penalty,
                    context,
                    args.text_only
                ): instance_id
                for instance_id, question, choices, answer, image_paths, context in all_tasks
            }
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_tasks), desc=f"Processing {subdir}"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    instance_id = futures[future]
                    results.append({
                        'instance_id': instance_id,
                        'status': 'exception',
                        'error': str(exc),
                        'correct': False
                    })
        
        # Calculate statistics
        total = len(results)
        correct = sum(1 for r in results if r.get('correct', False))
        errors = sum(1 for r in results if r.get('status') not in ['success'])
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"Results Summary for {subdir}:")
        print(f"{'='*80}")
        print(f"Total questions:     {total}")
        print(f"Correct answers:     {correct}")
        print(f"Incorrect answers:   {total - correct - errors}")
        print(f"Errors:              {errors}")
        print(f"Accuracy:            {accuracy:.2f}%")
        print(f"{'='*80}")
        
        # Perform detailed analysis by difficulty and length
        print(f"\nPerforming detailed analysis by difficulty and length...")
        detailed_analysis = analyze_by_difficulty_and_length(results, mc_data)
        
        # Perform analysis by context length
        print(f"\nPerforming analysis by context length...")
        tokenizer = tiktoken.get_encoding('cl100k_base')
        context_analysis = analyze_by_context_length(results, mc_data, tokenizer)
        
        # Print analysis summary
        print(f"\n{'='*80}")
        print(f"Analysis by Context Length:")
        print(f"{'='*80}")
        for length_bin, stats in context_analysis['by_context_length'].items():
            if stats['total'] > 0:
                print(f"{length_bin:12} | Total: {stats['total']:4} | Correct: {stats['correct']:4} | Accuracy: {stats['accuracy']:6.2f}%")
        
        print(f"\n{'='*80}")
        print(f"Analysis by Difficulty:")
        print(f"{'='*80}")
        for diff, stats in detailed_analysis['by_difficulty'].items():
            if stats['total'] > 0:
                print(f"{diff:12} | Total: {stats['total']:4} | Correct: {stats['correct']:4} | Accuracy: {stats['accuracy']:6.2f}%")
        
        print(f"\n{'='*80}")
        print(f"Analysis by Length:")
        print(f"{'='*80}")
        for length, stats in detailed_analysis['by_length'].items():
            if stats['total'] > 0:
                print(f"{length:12} | Total: {stats['total']:4} | Correct: {stats['correct']:4} | Accuracy: {stats['accuracy']:6.2f}%")
        
        print(f"\n{'='*80}")
        print(f"Analysis by Difficulty x Length:")
        print(f"{'='*80}")
        print(f"{'Difficulty':<12} | {'Length':<12} | {'Total':>5} | {'Correct':>7} | {'Accuracy':>8}")
        print(f"{'-'*80}")
        for diff, length_dict in detailed_analysis['by_difficulty_and_length'].items():
            for length, stats in length_dict.items():
                if stats['total'] > 0:
                    print(f"{diff:<12} | {length:<12} | {stats['total']:5} | {stats['correct']:7} | {stats['accuracy']:7.2f}%")
        
        # Save results for this subdirectory
        output_data = {
            'subdirectory': subdir,
            'model': model_name,
            'config': args.config,
            'text_only_mode': args.text_only,
            'data_path': data_json_path,
            'images_dir': images_dir,
            'total_questions': total,
            'correct': correct,
            'accuracy': accuracy,
            'errors': errors,
            'context_length_analysis': context_analysis,
            'detailed_analysis': detailed_analysis,
            'results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {output_path}")
        
        # Print some examples
        print(f"\nSample results:")
        print(f"{'-'*80}")
        for i, result in enumerate(results[:3]):
            status_icon = "✓" if result.get('correct') else "✗"
            print(f"{status_icon} Q{i+1}: {result['question'][:60]}...")
            print(f"   GT: {result['ground_truth']}")
            if result.get('extracted_answer'):
                print(f"   Model extracted: {result['extracted_answer']}")
            if result.get('model_output'):
                print(f"   Model output: {result['model_output'][:80]}...")
            print()
    
    print(f"\n{'='*80}")
    print(f"All subdirectories processed!")
    print(f"Results saved in: {base_output}")
    print(f"{'='*80}")
