# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_inputs_for_vllm(image_path: str, prompt: str, processor) -> dict:
    """Prepare input for vLLM inference"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def get_image_paths(input_path: str) -> List[str]:
    """Get all image paths from directory"""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_path, ext)))
        image_paths.extend(glob.glob(os.path.join(input_path, ext.upper())))
    return sorted(set(image_paths))


def infer_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    image_paths: List[str],
    prompt: str,
    processor,
) -> List[Tuple[str, str]]:
    """Run inference on a batch of images"""
    inputs = []
    valid_paths = []

    for path in image_paths:
        try:
            Image.open(path).convert("RGB")  # Validate image
            input_data = prepare_inputs_for_vllm(path, prompt, processor)
            inputs.append(input_data)
            valid_paths.append(path)
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")

    if not inputs:
        return []

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    results = [(path, output.outputs[0].text) for output, path in zip(outputs, valid_paths)]
    return results


def save_results(results: List[Tuple[str, str]], output_dir: str) -> int:
    """Save OCR results to markdown files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for image_path, content in results:
        mmd_path = output_dir / (Path(image_path).stem + ".md")
        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(content.strip())
        saved_count += 1

    logger.info(f"Saved {saved_count} results to {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL OCR inference with vLLM")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for OCR results")
    parser.add_argument("--batch-size", "-b", type=int, default=2048, help="Batch size (default: 2048)")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Maximum output tokens (default: 32768)")
    parser.add_argument(
        "--prompt", type=str, default="Convert this document to markdown.", help="Prompt for OCR task"
    )
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8", help="Model path")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed (default: 3407)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p sampling threshold (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling (default: 20)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty (default: 1.0)")
    parser.add_argument("--presence-penalty", type=float, default=1.5, help="Presence penalty (default: 1.5)")

    args = parser.parse_args()

    # Get image paths
    image_paths = get_image_paths(args.input)
    if not image_paths:
        logger.error(f"No images found in {args.input}")
        return

    logger.info(f"Found {len(image_paths)} images to process")

    # Initialize model and processor
    logger.info(f"Loading model: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)

    llm = LLM(
        model=args.model_path,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=True,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=args.seed,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        stop_token_ids=[],
    )

    # Run inference
    all_results = []
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Processing batches"):
        batch_paths = image_paths[i : i + args.batch_size]
        batch_results = infer_batch(llm, sampling_params, batch_paths, args.prompt, processor)
        all_results.extend(batch_results)

    # Save results
    save_results(all_results, args.output)
    logger.info(f"Completed: {len(all_results)} images processed")


if __name__ == "__main__":
    main()