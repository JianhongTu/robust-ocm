# -*- coding: utf-8 -*-
"""
Command-line interface for OCR evaluation.

This module provides a CLI for evaluating OCR predictions against ground truth.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

from .metrics import calculate_all_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_ground_truth(gt_file: str) -> Dict[str, str]:
    """
    Load ground truth from JSON file.

    Args:
        gt_file: Path to ground truth JSON file

    Returns:
        Dictionary mapping page_id to ground truth text
    """
    logger.info(f"Loading ground truth from: {gt_file}")

    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ground_truths = {}
    for item in data:
        image_path = item.get("page_info", {}).get("image_path", "")
        if not image_path:
            continue

        # Extract page_id from image path (e.g., 66ebb0e55a08c7b9b35ddd6a_page_001.png)
        page_id = Path(image_path).stem
        text = item.get("layout_dets", [{}])[0].get("text", "")

        if text:
            ground_truths[page_id] = text

    logger.info(f"Loaded {len(ground_truths)} ground truth entries")
    return ground_truths


def load_predictions(pred_dir: str) -> Dict[str, str]:
    """
    Load predictions from directory.

    Args:
        pred_dir: Path to directory containing .md prediction files

    Returns:
        Dictionary mapping page_id to predicted text
    """
    logger.info(f"Loading predictions from: {pred_dir}")

    predictions = {}
    pred_path = Path(pred_dir)

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    for md_file in pred_path.glob("*.md"):
        page_id = md_file.stem
        with open(md_file, "r", encoding="utf-8") as f:
            predictions[page_id] = f.read()

    logger.info(f"Loaded {len(predictions)} prediction files")
    return predictions


def print_results(results: Dict):
    """
    Print evaluation results in a formatted way.

    Args:
        results: Results dictionary from calculate_all_metrics
    """
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)

    metrics_order = ["cer", "bleu"]

    for metric in metrics_order:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        min_key = f"{metric}_min"
        max_key = f"{metric}_max"
        count_key = f"{metric}_count"

        if mean_key in results["summary"]:
            print(f"\n{metric.upper()}:")
            print(f"  Mean:   {results['summary'][mean_key]:.4f}")
            print(f"  Std:    {results['summary'][std_key]:.4f}")
            print(f"  Min:    {results['summary'][min_key]:.4f}")
            print(f"  Max:    {results['summary'][max_key]:.4f}")
            print(f"  Count:  {results['summary'][count_key]}")

    print("\n" + "=" * 80)


def save_results(results: Dict, output_file: str):
    """
    Save evaluation results to JSON file with summary first, excluding duplicate metrics.

    Args:
        results: Results dictionary from calculate_all_metrics
        output_file: Path to output JSON file
    """
    logger.info(f"Saving results to: {output_file}")

    # Reorder results to put summary first, exclude duplicate metrics
    ordered_results = {}
    if "summary" in results:
        ordered_results["summary"] = results["summary"]
    if "per_page" in results:
        ordered_results["per_page"] = results["per_page"]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ordered_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved successfully")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate OCR predictions against ground truth using OmniDocBench metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation
    python -m robust_ocm.eval --gt data/longbenchv2_img/OmniDocBench_concatenated.json \\
                              --pred data/pred/dpsk

    # Evaluation with output file
    python -m robust_ocm.eval --gt data/longbenchv2_img/OmniDocBench_concatenated.json \\
                              --pred data/pred/dpsk \\
                              --output results/dpsk_evaluation.json

    # Evaluation without normalization
    python -m robust_ocm.eval --gt data/longbenchv2_img/OmniDocBench_concatenated.json \\
                              --pred data/pred/dpsk \\
                              --no-normalize

Metrics:
    - CER: Character Error Rate (lower is better)
    - TEDS: Tree Edit Distance Similarity (higher is better)
    - BLEU: BLEU score (higher is better)
    - METEOR: METEOR score (higher is better)
        """,
    )

    parser.add_argument(
        "--gt",
        "-g",
        type=str,
        required=True,
        help="Path to ground truth JSON file (e.g., data/longbenchv2_img/OmniDocBench_concatenated.json)",
    )
    parser.add_argument(
        "--pred",
        "-p",
        type=str,
        required=True,
        help="Path to prediction directory containing .md files (e.g., data/pred/dpsk)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output JSON file for saving results",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable text normalization before metric calculation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load data
    try:
        ground_truths = load_ground_truth(args.gt)
        predictions = load_predictions(args.pred)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    # Check for missing predictions
    missing_gt = set(ground_truths.keys()) - set(predictions.keys())
    missing_pred = set(predictions.keys()) - set(ground_truths.keys())

    if missing_gt:
        logger.warning(f"Missing predictions for {len(missing_gt)} ground truth pages")
    if missing_pred:
        logger.warning(f"Extra predictions not in ground truth: {len(missing_pred)}")

    # Calculate metrics
    logger.info("Calculating metrics...")
    normalize = not args.no_normalize
    results = calculate_all_metrics(
        predictions=predictions,
        ground_truths=ground_truths,
        normalize=normalize,
    )

    # Print results
    print_results(results)

    # Save results if output file specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_results(results, args.output)

    logger.info("Evaluation complete!")
    return 0


if __name__ == "__main__":
    exit(main())