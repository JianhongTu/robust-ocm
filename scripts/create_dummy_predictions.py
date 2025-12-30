#!/usr/bin/env python3
"""
Create dummy predictions from ground truth for testing evaluation scripts

This script extracts text from ground truth files and saves them as .md files
to simulate OCR predictions. Useful for testing evaluation pipelines.

Usage:
    python create_dummy_predictions.py --ground-truth data/longbenchv2_img/OmniDocBench.json --output data/pred/dummy
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_text_from_omnidoc(omnidoc_doc: Dict[str, Any]) -> str:
    """
    Extract concatenated text from OmniDocBench format document
    
    Args:
        omnidoc_doc: Dictionary in OmniDocBench format
        
    Returns:
        Concatenated text string (single line)
    """
    layout_dets = omnidoc_doc.get("layout_dets", [])
    
    # Extract text from all layout elements in reading order
    texts = []
    for element in sorted(layout_dets, key=lambda x: x.get("order", 0)):
        text = element.get("text", "")
        if text:
            texts.append(text)
    
    # Join with a single space to create one continuous line
    return " ".join(texts)


def extract_text_from_line_bbox(line_data: Dict[str, Any]) -> List[str]:
    """
    Extract text from line_bbox.jsonl format
    
    Args:
        line_data: Dictionary with unique_id, image_paths, bboxes
        
    Returns:
        List of text strings (one per page, each as a single line)
    """
    bboxes = line_data.get("bboxes", [])
    texts_per_page = []
    
    for page_bboxes in bboxes:
        # Extract text from all bboxes on this page
        page_texts = []
        for bbox_item in page_bboxes:
            if len(bbox_item) > 4:
                text = bbox_item[4]
                if text:
                    page_texts.append(text)
        
        # Join with a single space to create one continuous line
        texts_per_page.append(" ".join(page_texts))
    
    return texts_per_page


def create_dummy_predictions_from_omnidoc(
    input_path: str,
    output_dir: str,
    limit: int = None
) -> int:
    """
    Create dummy predictions from OmniDocBench.json format
    
    Args:
        input_path: Path to OmniDocBench.json file
        output_dir: Directory to save .md files
        limit: Maximum number of documents to process
        
    Returns:
        Number of files created
    """
    logger.info(f"Reading OmniDocBench format from {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        omnidoc_docs = json.load(f)
    
    if limit:
        omnidoc_docs = omnidoc_docs[:limit]
        logger.info(f"Limited to {len(omnidoc_docs)} documents")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    created_count = 0
    
    for idx, omnidoc_doc in enumerate(omnidoc_docs):
        try:
            # Extract text
            text = extract_text_from_omnidoc(omnidoc_doc)
            
            # Get image path for naming
            page_info = omnidoc_doc.get("page_info", {})
            image_path = page_info.get("image_path", f"page_{idx:06d}")
            
            # Create filename (remove extension, add .md)
            filename = Path(image_path).stem + ".md"
            output_file = output_path / filename
            
            # Save as .md file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            created_count += 1
            
            if created_count % 100 == 0:
                logger.info(f"Created {created_count} dummy predictions...")
                
        except Exception as e:
            logger.warning(f"Failed to process document {idx}: {e}")
            continue
    
    logger.info(f"Created {created_count} dummy predictions from OmniDocBench format")
    return created_count


def create_dummy_predictions_from_line_bbox(
    input_path: str,
    output_dir: str,
    limit: int = None
) -> int:
    """
    Create dummy predictions from line_bbox.jsonl format
    
    Args:
        input_path: Path to line_bbox.jsonl file
        output_dir: Directory to save .md files
        limit: Maximum number of documents to process
        
    Returns:
        Number of files created
    """
    logger.info(f"Reading line_bbox.jsonl format from {input_path}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    created_count = 0
    doc_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if limit and doc_count >= limit:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                line_data = json.loads(line)
                unique_id = line_data.get("unique_id", f"doc_{doc_count}")
                image_paths = line_data.get("image_paths", [])
                texts_per_page = extract_text_from_line_bbox(line_data)
                
                # Create .md file for each page
                for page_idx, text in enumerate(texts_per_page):
                    # Use image path for naming if available
                    if page_idx < len(image_paths):
                        image_path = image_paths[page_idx]
                        filename = Path(image_path).stem + ".md"
                    else:
                        filename = f"{unique_id}_page_{page_idx:03d}.md"
                    
                    output_file = output_path / filename
                    
                    # Save as .md file
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    created_count += 1
                
                doc_count += 1
                
                if doc_count % 50 == 0:
                    logger.info(f"Processed {doc_count} documents ({created_count} pages)...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse document {doc_count}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Failed to process document {doc_count}: {e}")
                continue
    
    logger.info(f"Created {created_count} dummy predictions from {doc_count} documents")
    return created_count


def detect_format(input_path: str) -> str:
    """
    Detect the format of the input file
    
    Args:
        input_path: Path to input file
        
    Returns:
        'omnidoc' or 'line_bbox' or 'unknown'
    """
    # Check file extension
    if input_path.endswith('.json'):
        # Try to read and check structure
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if it's a list (OmniDocBench format)
            if isinstance(data, list):
                if len(data) > 0 and "layout_dets" in data[0]:
                    return "omnidoc"
            
            # Check if it's a dict with specific keys
            if isinstance(data, dict):
                if "layout_dets" in data:
                    return "omnidoc"
                
        except Exception:
            pass
    
    elif input_path.endswith('.jsonl'):
        # JSONL format is typically line_bbox
        return "line_bbox"
    
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description='Create dummy predictions from ground truth for testing evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create dummy predictions from OmniDocBench format
    python create_dummy_predictions.py --ground-truth data/longbenchv2_img/OmniDocBench.json --output data/pred/dummy
    
    # Create dummy predictions from line_bbox.jsonl format
    python create_dummy_predictions.py --ground-truth data/longbenchv2_img/line_bbox.jsonl --output data/pred/dummy
    
    # Limit to 100 documents for quick testing
    python create_dummy_predictions.py --ground-truth data/longbenchv2_img/OmniDocBench.json --output data/pred/dummy --limit 100
    
    # Auto-detect format
    python create_dummy_predictions.py --ground-truth data/longbenchv2_img/OmniDocBench.json --output data/pred/dummy
        """
    )
    
    parser.add_argument(
        '--ground-truth', '-g',
        required=True,
        help='Path to ground truth file (OmniDocBench.json or line_bbox.jsonl)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for dummy prediction .md files'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Maximum number of documents to process (default: all)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['omnidoc', 'line_bbox', 'auto'],
        default='auto',
        help='Input format (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.ground_truth):
        logger.error(f"Ground truth file does not exist: {args.ground_truth}")
        return 1
    
    # Detect format if auto
    format_type = args.format
    if format_type == 'auto':
        format_type = detect_format(args.ground_truth)
        logger.info(f"Auto-detected format: {format_type}")
    
    if format_type == 'unknown':
        logger.error(f"Could not detect format from {args.ground_truth}. Please specify --format explicitly.")
        return 1
    
    # Create dummy predictions
    try:
        if format_type == 'omnidoc':
            created_count = create_dummy_predictions_from_omnidoc(
                args.ground_truth,
                args.output,
                args.limit
            )
        elif format_type == 'line_bbox':
            created_count = create_dummy_predictions_from_line_bbox(
                args.ground_truth,
                args.output,
                args.limit
            )
        else:
            logger.error(f"Unsupported format: {format_type}")
            return 1
        
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Format: {format_type}")
        logger.info(f"Files created: {created_count}")
        logger.info(f"Output directory: {args.output}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error creating dummy predictions: {e}")
        return 1


if __name__ == "__main__":
    exit(main())