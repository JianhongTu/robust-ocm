#!/usr/bin/env python3
"""
Multi-model NIAH (Needle in a Haystack) VQA inference script with concurrent processing.
Evaluates vision-language models on the NIAH task by asking questions about secrets
hidden in long documents.

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

Then run this script:
micromamba run -n test python scripts/vqa/niah_inference.py \
    --input data/niah \
    --output results/qwen3 \
    --config config/vqa/qwen3-vl.json \
    --base_url http://localhost:8000/v1 \
    --max_workers 128

The script will automatically:
- Load data.json from the parent directory (data/niah/data.json)
- Scan all subdirectories for images/ folders
- Process each subdirectory (different perturbations of the same data)
- Save results to results/qwen3/{subdir}_niah_eval.json
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


def load_niah_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load NIAH dataset from JSON file.
    
    Args:
        data_path: Path to the NIAH data.json file
    
    Returns:
        List of NIAH instances
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def format_question_prompt(question: str, prompt_template: str = None) -> str:
    """
    Format the question prompt using a template if provided.
    
    Args:
        question: The question to ask
        prompt_template: Optional template string with {question} placeholder
    
    Returns:
        Formatted prompt string
    """
    if prompt_template:
        return prompt_template.format(question=question)
    else:
        # Default prompt format
        return question


def check_answer_correct(model_output: str, ground_truth: str, case_sensitive: bool = False) -> bool:
    """
    Check if the model output contains the ground truth answer.
    
    Args:
        model_output: The model's output string
        ground_truth: The expected answer
        case_sensitive: Whether to perform case-sensitive matching
    
    Returns:
        True if the answer is correct, False otherwise
    """
    if not case_sensitive:
        model_output = model_output.lower()
        ground_truth = ground_truth.lower()
    
    return ground_truth in model_output


def process_question(
    client: OpenAI,
    instance_id: str,
    question_idx: int,
    question: str,
    answer: str,
    image_paths: List[str],
    images_dir: str,
    model_config: dict,
    presence_penalty: float = 0.0,
    case_sensitive: bool = False
) -> Dict[str, Any]:
    """
    Process a single question and evaluate the answer.
    
    Args:
        client: OpenAI client instance
        instance_id: ID of the instance
        question_idx: Index of the question in the instance
        question: The question to ask
        answer: The ground truth answer
        image_paths: List of image file names for this instance
        images_dir: Directory containing the images
        model_config: Model configuration dict
        presence_penalty: Presence penalty parameter
        case_sensitive: Whether to use case-sensitive answer matching
    
    Returns:
        Dictionary with question results
    """
    try:
        # Format the prompt
        prompt = format_question_prompt(question, model_config.get('prompt_template'))
        
        # Prepare image data URLs
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
        
        # Check if answer is correct
        is_correct = check_answer_correct(model_output, answer, case_sensitive)
        
        return {
            'instance_id': instance_id,
            'question_idx': question_idx,
            'question': question,
            'ground_truth': answer,
            'model_output': model_output,
            'correct': is_correct,
            'num_images': len(image_paths),
            'status': 'success'
        }
    
    except APIConnectionError as e:
        return {
            'instance_id': instance_id,
            'question_idx': question_idx,
            'question': question,
            'ground_truth': answer,
            'model_output': None,
            'correct': False,
            'num_images': len(image_paths),
            'status': 'connection_error',
            'error': str(e)
        }
    except Exception as e:
        return {
            'instance_id': instance_id,
            'question_idx': question_idx,
            'question': question,
            'ground_truth': answer,
            'model_output': None,
            'correct': False,
            'num_images': len(image_paths),
            'status': 'error',
            'error': str(e)
        }


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Multi-model NIAH VQA inference with concurrent processing')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Parent directory with data.json and subdirectories containing images/')
    
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
                       default=256,
                       help='Number of concurrent workers')
    
    parser.add_argument('--presence_penalty', type=float,
                       default=0.0,
                       help='Presence penalty for repetition control (0.0 to 2.0)')
    
    parser.add_argument('--case_sensitive', action='store_true',
                       help='Use case-sensitive answer matching')
    
    parser.add_argument('--limit', type=int,
                       default=None,
                       help='Limit processing to N instances per subdirectory (for testing)')
    
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


def get_needle_depth(instance: Dict[str, Any], question_idx: int, question: str) -> str:
    """
    Determine which depth range (0-20%, 20-40%, etc.) a question belongs to.
    
    First tries to use the question_depths field if available (recommended).
    Falls back to inferring from secret position for backward compatibility.
    
    Args:
        instance: NIAH instance with question_depths and secrets
        question_idx: Index of the question in the instance
        question: The question text (used for fallback inference)
    
    Returns:
        Depth range string (e.g., '0-20%', '20-40%', etc.)
    """
    # Primary method: Use question_depths field if available
    question_depths = instance.get('question_depths', [])
    if question_depths and question_idx < len(question_depths):
        return question_depths[question_idx]
    
    # Fallback method: Infer from secret position (for backward compatibility)
    # This is less accurate because questions may not be in insertion order
    secrets = instance.get('secrets', {})
    
    for key in secrets.keys():
        if key in question.lower():
            secret_keys = list(secrets.keys())
            position_idx = secret_keys.index(key)
            
            depth_ranges = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
            if position_idx < len(depth_ranges):
                return depth_ranges[position_idx]
    
    return 'unknown'


def analyze_by_context_and_depth(
    results: List[Dict[str, Any]], 
    niah_data: List[Dict[str, Any]],
    tokenizer
) -> Dict[str, Any]:
    """
    Analyze accuracy by context length and needle depth.
    
    Args:
        results: List of result dictionaries from inference
        niah_data: Original NIAH dataset
        tokenizer: Tiktoken tokenizer for counting tokens
    
    Returns:
        Dictionary with analysis results
    """
    # Create instance lookup
    instance_lookup = {inst['_id']: inst for inst in niah_data}
    
    # Initialize stats structure
    length_bins = ['<8k', '8k-16k', '16k-24k', '24k-32k', '32k-48k', '48k-64k', '>=64k']
    depth_bins = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    
    # Stats by length
    length_stats = {length: {'total': 0, 'correct': 0} for length in length_bins}
    
    # Stats by depth
    depth_stats = {depth: {'total': 0, 'correct': 0} for depth in depth_bins}
    
    # Stats by length x depth
    combined_stats = {
        length: {depth: {'total': 0, 'correct': 0} for depth in depth_bins}
        for length in length_bins
    }
    
    # Tokenize contexts (cache to avoid re-tokenizing)
    context_token_counts = {}
    for inst in niah_data:
        inst_id = inst['_id']
        context = inst.get('context', '')
        if isinstance(context, str):
            token_count = len(tokenizer.encode(context, disallowed_special=()))
            context_token_counts[inst_id] = token_count
    
    # Process each result
    for result in results:
        inst_id = result['instance_id']
        question_idx = result['question_idx']
        question = result['question']
        is_correct = result.get('correct', False)
        
        # Get instance and token count
        instance = instance_lookup.get(inst_id)
        if not instance:
            continue
        
        token_count = context_token_counts.get(inst_id, 0)
        length_bin = categorize_context_length(token_count)
        depth_bin = get_needle_depth(instance, question_idx, question)
        
        if depth_bin == 'unknown':
            continue
        
        # Update stats
        length_stats[length_bin]['total'] += 1
        if is_correct:
            length_stats[length_bin]['correct'] += 1
        
        depth_stats[depth_bin]['total'] += 1
        if is_correct:
            depth_stats[depth_bin]['correct'] += 1
        
        combined_stats[length_bin][depth_bin]['total'] += 1
        if is_correct:
            combined_stats[length_bin][depth_bin]['correct'] += 1
    
    # Calculate accuracies
    for length in length_bins:
        total = length_stats[length]['total']
        if total > 0:
            length_stats[length]['accuracy'] = length_stats[length]['correct'] / total * 100
        else:
            length_stats[length]['accuracy'] = None
    
    for depth in depth_bins:
        total = depth_stats[depth]['total']
        if total > 0:
            depth_stats[depth]['accuracy'] = depth_stats[depth]['correct'] / total * 100
        else:
            depth_stats[depth]['accuracy'] = None
    
    for length in length_bins:
        for depth in depth_bins:
            total = combined_stats[length][depth]['total']
            if total > 0:
                combined_stats[length][depth]['accuracy'] = combined_stats[length][depth]['correct'] / total * 100
            else:
                combined_stats[length][depth]['accuracy'] = None
    
    return {
        'by_context_length': length_stats,
        'by_needle_depth': depth_stats,
        'by_length_and_depth': combined_stats
    }



if __name__ == "__main__":
    args = parse_args()
    
    parent_input = args.input
    base_output = args.output
    
    # Create output directory
    os.makedirs(base_output, exist_ok=True)

    # Load model configuration
    try:
        model_config = load_model_config(args.config)
        model_name = model_config['model_name']
        print(f"Loaded configuration from: {args.config}")
        print(f"Model: {model_name}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Load NIAH data from parent directory (shared across all subdirectories)
    data_json_path = os.path.join(parent_input, 'data.json')
    if not os.path.exists(data_json_path):
        print(f"Error: data.json not found at {data_json_path}")
        sys.exit(1)
    
    try:
        niah_data = load_niah_data(data_json_path)
        print(f"Loaded {len(niah_data)} instances from: {data_json_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Apply limit if specified
    if args.limit is not None:
        niah_data = niah_data[:args.limit]
        print(f"Limited to {len(niah_data)} instances for testing")
    
    # Create OpenAI client
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key if args.api_key else "dummy",
    )
    
    # Find subdirectories with images/
    print(f"\nScanning for subdirectories with images/ in {parent_input}...")
    subdirs_to_process = []
    
    for item in os.listdir(parent_input):
        item_path = os.path.join(parent_input, item)
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
        
        images_dir = os.path.join(parent_input, subdir, 'images')
        output_path = os.path.join(base_output, f"{subdir}_niah_eval.json")
        
        # Check if already processed
        if os.path.exists(output_path):
            print(f"⏭  Results already exist at {output_path}, skipping...")
            continue
        
        # Build task queue for this subdirectory
        print(f"\nBuilding task queue for {subdir}...")
        all_tasks = []
        total_questions = 0
        
        for instance in niah_data:
            instance_id = instance['_id']
            questions = instance.get('questions', [])
            answers = instance.get('answers', [])
            
            # Get image paths for this instance
            image_pattern = f"{instance_id}_*.png"
            import glob
            image_paths = sorted([
                os.path.basename(p) 
                for p in glob.glob(os.path.join(images_dir, image_pattern))
            ])
            
            if not image_paths:
                print(f"Warning: No images found for instance {instance_id}")
                continue
            
            # Add all questions for this instance
            for q_idx, (question, answer) in enumerate(zip(questions, answers)):
                all_tasks.append((instance_id, q_idx, question, answer, image_paths))
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
                    q_idx,
                    question,
                    answer,
                    image_paths,
                    images_dir,
                    model_config,
                    args.presence_penalty,
                    args.case_sensitive
                ): (instance_id, q_idx)
                for instance_id, q_idx, question, answer, image_paths in all_tasks
            }
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_tasks), desc=f"Processing {subdir}"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    instance_id, q_idx = futures[future]
                    results.append({
                        'instance_id': instance_id,
                        'question_idx': q_idx,
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
        
        # Group results by instance
        instance_results = {}
        for result in results:
            inst_id = result['instance_id']
            if inst_id not in instance_results:
                instance_results[inst_id] = []
            instance_results[inst_id].append(result)
        
        # Calculate per-instance accuracy
        instance_accuracies = {}
        for inst_id, inst_results in instance_results.items():
            inst_total = len(inst_results)
            inst_correct = sum(1 for r in inst_results if r.get('correct', False))
            instance_accuracies[inst_id] = {
                'total': inst_total,
                'correct': inst_correct,
                'accuracy': (inst_correct / inst_total * 100) if inst_total > 0 else 0
            }
        
        # Perform detailed analysis by context length and needle depth
        print(f"\nPerforming detailed analysis by context length and needle depth...")
        tokenizer = tiktoken.get_encoding('cl100k_base')
        detailed_analysis = analyze_by_context_and_depth(results, niah_data, tokenizer)
        
        # Print analysis summary
        print(f"\n{'='*80}")
        print(f"Analysis by Context Length:")
        print(f"{'='*80}")
        for length_bin, stats in detailed_analysis['by_context_length'].items():
            if stats['total'] > 0:
                print(f"{length_bin:12} | Total: {stats['total']:4} | Correct: {stats['correct']:4} | Accuracy: {stats['accuracy']:6.2f}%")
        
        print(f"\n{'='*80}")
        print(f"Analysis by Needle Depth:")
        print(f"{'='*80}")
        for depth_bin, stats in detailed_analysis['by_needle_depth'].items():
            if stats['total'] > 0:
                print(f"{depth_bin:12} | Total: {stats['total']:4} | Correct: {stats['correct']:4} | Accuracy: {stats['accuracy']:6.2f}%")
        
        print(f"\n{'='*80}")
        print(f"Analysis by Context Length x Needle Depth:")
        print(f"{'='*80}")
        print(f"{'Length':<12} | {'Depth':<12} | {'Total':>5} | {'Correct':>7} | {'Accuracy':>8}")
        print(f"{'-'*80}")
        for length_bin, depth_dict in detailed_analysis['by_length_and_depth'].items():
            for depth_bin, stats in depth_dict.items():
                if stats['total'] > 0:
                    print(f"{length_bin:<12} | {depth_bin:<12} | {stats['total']:5} | {stats['correct']:7} | {stats['accuracy']:7.2f}%")
        
        # Save results for this subdirectory
        output_data = {
            'subdirectory': subdir,
            'model': model_name,
            'config': args.config,
            'data_path': data_json_path,
            'images_dir': images_dir,
            'total_questions': total,
            'correct': correct,
            'accuracy': accuracy,
            'errors': errors,
            'case_sensitive': args.case_sensitive,
            'instance_accuracies': instance_accuracies,
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
            if result.get('model_output'):
                print(f"   Model: {result['model_output'][:80]}...")
            print()
    
    print(f"\n{'='*80}")
    print(f"All subdirectories processed!")
    print(f"Results saved in: {base_output}")
    print(f"{'='*80}")
