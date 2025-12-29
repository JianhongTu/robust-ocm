"""
Command line interface to convert line_bbox.jsonl to OmniDocBench format
"""

import os
import json
import argparse
from typing import List, Dict, Any
from pathlib import Path


def bbox_to_polygon(bbox: List[int]) -> List[float]:
    """
    Convert bbox format [x1, y1, x2, y2] to polygon format [x1, y1, x2, y1, x2, y2, x1, y2]
    
    Args:
        bbox: List of [x1, y1, x2, y2]
        
    Returns:
        List of 8 polygon coordinates
    """
    x1, y1, x2, y2 = bbox[:4]
    return [float(x1), float(y1), float(x2), float(y1), float(x2), float(y2), float(x1), float(y2)]


def create_layout_element(bbox_item: List[int], order: int, category_type: str = "text_block") -> Dict[str, Any]:
    """
    Create a layout element in OmniDocBench format from a bbox item
    
    Args:
        bbox_item: [x1, y1, x2, y2, text] format
        order: Reading order
        category_type: Type of the element (default: text_block)
        
    Returns:
        Dictionary in OmniDocBench layout_dets format
    """
    x1, y1, x2, y2, text = bbox_item
    poly = bbox_to_polygon([x1, y1, x2, y2])
    
    # Create text span
    text_span = {
        "category_type": "text_span",
        "poly": poly,
        "text": text
    }
    
    # Create layout element
    layout_element = {
        "category_type": category_type,
        "poly": poly,
        "ignore": False,
        "order": order,
        "anno_id": order,  # Use order as annotation ID
        "text": text,
        "line_with_spans": [text_span],
        "attribute": {
            "text_language": "text_english",  # English text
            "text_background": "white",
            "text_rotate": "normal"
        }
    }
    
    return layout_element


def create_giant_bbox(page_bboxes: List[List[int]]) -> List[int]:
    """
    Create a single giant bounding box that encompasses all bboxes
    
    Args:
        page_bboxes: List of [x1, y1, x2, y2, text] format
        
    Returns:
        Single bbox [x1, y1, x2, y2, concatenated_text] with outermost coordinates
    """
    if not page_bboxes:
        return [0, 0, 0, 0, ""]
    
    # Initialize with first bbox
    x1_min, y1_min, x2_max, y2_max = page_bboxes[0][:4]
    
    # Find outermost coordinates
    for bbox_item in page_bboxes:
        x1, y1, x2, y2 = bbox_item[:4]
        x1_min = min(x1_min, x1)
        y1_min = min(y1_min, y1)
        x2_max = max(x2_max, x2)
        y2_max = max(y2_max, y2)
    
    # Concatenate all text
    concatenated_text = " ".join(bbox_item[4] if len(bbox_item) > 4 else "" for bbox_item in page_bboxes)
    
    return [x1_min, y1_min, x2_max, y2_max, concatenated_text]


def convert_line_to_omnidoc(line_data: Dict[str, Any], page_idx: int = 0, concatenate: bool = False) -> Dict[str, Any]:
    """
    Convert a single line from line_bbox.jsonl to OmniDocBench format
    
    Args:
        line_data: Dictionary with unique_id, image_paths, bboxes
        page_idx: Which page to convert (0-based)
        concatenate: If True, create single giant bbox for all text
        
    Returns:
        Dictionary in OmniDocBench format
    """
    unique_id = line_data["unique_id"]
    image_paths = line_data["image_paths"]
    bboxes = line_data["bboxes"]
    
    if page_idx >= len(bboxes):
        raise ValueError(f"Page index {page_idx} out of range for document {unique_id}")
    
    page_bboxes = bboxes[page_idx]
    image_path = image_paths[page_idx] if page_idx < len(image_paths) else ""
    
    # Extract image dimensions from path or use defaults
    # For now, we'll use default dimensions
    width, height = 1700, 2200  # Default dimensions
    
    # Create layout elements
    layout_dets = []
    
    if concatenate:
        # Create single giant bbox encompassing all text
        giant_bbox = create_giant_bbox(page_bboxes)
        layout_element = create_layout_element(giant_bbox, order=1, category_type="text_block")
        layout_dets.append(layout_element)
    else:
        # Create individual layout elements for each bbox
        for order, bbox_item in enumerate(page_bboxes, 1):
            # Determine category type based on text content or position
            text = bbox_item[4] if len(bbox_item) > 4 else ""
            
            # Simple heuristic for categorization
            if text.isupper() and len(text) < 50:
                category_type = "title"
            elif text.startswith("Figure") or text.startswith("Table"):
                category_type = "figure_caption"
            else:
                category_type = "text_block"
            
            layout_element = create_layout_element(bbox_item, order, category_type)
            layout_dets.append(layout_element)
    
    # Create page info
    page_info = {
        "page_attribute": {
            "data_source": "academic_literature",  # LongBench-v2 is academic literature
            "language": "english",  # English text
            "layout": "single_column",  # Most academic papers use single column
            "special_issue": []  # No special issues for generated images
        },
        "page_no": page_idx + 1,
        "height": height,
        "width": width,
        "image_path": os.path.basename(image_path)
    }
    
    # Create extra info (empty relations for now)
    extra = {
        "relation": []
    }
    
    # Create OmniDocBench format document
    omnidoc_doc = {
        "layout_dets": layout_dets,
        "extra": extra,
        "page_info": page_info
    }
    
    return omnidoc_doc


def convert_jsonl_to_omnidoc(input_path: str, output_path: str, limit: int = None, concatenate: bool = False):
    """
    Convert line_bbox.jsonl to OmniDocBench.json format
    
    Args:
        input_path: Path to input line_bbox.jsonl file
        output_path: Path to output OmniDocBench.json file
        limit: Maximum number of documents to convert (None for all)
        concatenate: If True, create single giant bbox for all text on each page
    """
    mode_str = "concatenated (single giant bbox)" if concatenate else "individual bboxes"
    print(f"Converting {input_path} to {output_path} (mode: {mode_str})...")
    
    omnidoc_documents = []
    total_pages_processed = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        doc_count = 0
        for line in f:
            if limit and doc_count >= limit:
                break
                
            line = line.strip()
            if not line:
                continue
                
            try:
                line_data = json.loads(line)
                unique_id = line_data["unique_id"]
                bboxes = line_data["bboxes"]
                
                # Process each page of the document
                for page_idx in range(len(bboxes)):
                    omnidoc_doc = convert_line_to_omnidoc(line_data, page_idx, concatenate)
                    omnidoc_documents.append(omnidoc_doc)
                    total_pages_processed += 1
                
                doc_count += 1
                
                if doc_count % 50 == 0:
                    print(f"Processed {doc_count} documents ({total_pages_processed} pages total)...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing document {doc_count}: {e}")
                continue
            except Exception as e:
                print(f"Error processing document {doc_count}: {e}")
                continue
    
    print(f"Converted {len(omnidoc_documents)} pages from {doc_count} documents")
    
    # Save to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(omnidoc_documents, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Convert line_bbox.jsonl to OmniDocBench.json format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert all documents with individual bboxes (default)
    python convert_to_omnidoc.py --input data/longbenchv2_img/line_bbox.jsonl --output data/longbenchv2_img/omnidoc_format.json
    
    # Convert with single giant bbox per page (all text concatenated)
    python convert_to_omnidoc.py --input data/longbenchv2_img/line_bbox.jsonl --output data/longbenchv2_img/omnidoc_concatenated.json --concatenate
    
    # Convert first 100 documents with giant bbox
    python convert_to_omnidoc.py --input data/longbenchv2_img/line_bbox.jsonl --output data/longbenchv2_img/omnidoc_concatenated.json --concatenate --limit 100
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Path to input line_bbox.jsonl file'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output OmniDocBench.json file'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of documents to convert (default: all)'
    )
    
    parser.add_argument(
        '--concatenate',
        action='store_true',
        help='Concatenate all bboxes into a single giant bbox per page based on outermost coordinates'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert the file
    try:
        convert_jsonl_to_omnidoc(args.input, args.output, args.limit, args.concatenate)
        print("Conversion completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


if __name__ == '__main__':
    exit(main())