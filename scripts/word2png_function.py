#!/usr/bin/env python3
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import io
import os
import json
import numpy as np
import gc
from pdf2image import pdfinfo_from_bytes, convert_from_bytes
import re
from multiprocessing import Pool
from tqdm import tqdm
from xml.sax.saxutils import escape
import shutil
import warnings
import sys
import argparse

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors

# Alignment mapping
ALIGN_MAP = {
    "LEFT": TA_LEFT,
    "CENTER": TA_CENTER,
    "RIGHT": TA_RIGHT,
    "JUSTIFY": TA_JUSTIFY,
}

# Global variables for multiprocessing
GLOBAL_CONFIG = None
OUTPUT_DIR = None
recover = False


def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Convert colors
    if 'page-bg-color' in config and isinstance(config['page-bg-color'], str):
        config['page-bg-color'] = colors.HexColor(config['page-bg-color'])
    if 'font-color' in config and isinstance(config['font-color'], str):
        config['font-color'] = colors.HexColor(config['font-color'])
    if 'para-bg-color' in config and isinstance(config['para-bg-color'], str):
        config['para-bg-color'] = colors.HexColor(config['para-bg-color'])
    if 'para-border-color' in config and isinstance(config['para-border-color'], str):
        config['para-border-color'] = colors.HexColor(config['para-border-color'])
    
    # Convert alignment
    if 'alignment' in config and isinstance(config['alignment'], str):
        config['alignment'] = ALIGN_MAP.get(config['alignment'], TA_JUSTIFY)
    
    # Convert page size
    if 'page-size' in config and isinstance(config['page-size'], str):
        config['page-size'] = tuple(map(float, config['page-size'].split(',')))
    
    return config


def text_to_images(text, output_dir, config_path=None, config_dict=None, unique_id=None):
    """
    Convert text to images - Inference interface
    
    Args:
        text: Input text content
        output_dir: Image output directory
        config_path: Configuration file path (optional)
        config_dict: Configuration dictionary (optional, higher priority than config_path)
        unique_id: Unique identifier (optional, auto-generated if not provided)
        
    Returns:
        list: List of generated image paths
        
    Example:
        >>> images = text_to_images(
        ...     text="Hello World",
        ...     output_dir="./output",
        ...     config_path="config.json"
        ... )
        >>> print(images)  # ['./output/xxx/page_001.png', ...]
    """
    # Load configuration
    if config_dict is None:
        if config_path is None:
            raise ValueError("Must provide either config_path or config_dict")
        config = load_config(config_path)
    else:
        config = config_dict.copy()
        # Convert special fields in config
        if 'page-bg-color' in config and isinstance(config['page-bg-color'], str):
            config['page-bg-color'] = colors.HexColor(config['page-bg-color'])
        if 'font-color' in config and isinstance(config['font-color'], str):
            config['font-color'] = colors.HexColor(config['font-color'])
        if 'para-bg-color' in config and isinstance(config['para-bg-color'], str):
            config['para-bg-color'] = colors.HexColor(config['para-bg-color'])
        if 'para-border-color' in config and isinstance(config['para-border-color'], str):
            config['para-border-color'] = colors.HexColor(config['para-border-color'])
        if 'alignment' in config and isinstance(config['alignment'], str):
            config['alignment'] = ALIGN_MAP.get(config['alignment'], TA_JUSTIFY)
        if 'page-size' in config and isinstance(config['page-size'], str):
            config['page-size'] = tuple(map(float, config['page-size'].split(',')))
    
    # Generate unique ID
    if unique_id is None:
        import hashlib
        unique_id = hashlib.md5(text.encode()).hexdigest()[:16]
    
    # Extract configuration parameters
    page_size = config.get('page-size', A4)
    margin_x = config.get('margin-x', 20)
    margin_y = config.get('margin-y', 20)
    font_path = config.get('font-path')
    assert font_path, "Must provide font-path"
    
    font_name = os.path.basename(font_path).split('.')[0]
    font_size = config.get('font-size', 9)
    line_height = config.get('line-height') or (font_size + 1)
    
    page_bg_color = config.get('page-bg-color', colors.HexColor('#FFFFFF'))
    font_color = config.get('font-color', colors.HexColor('#000000'))
    para_bg_color = config.get('para-bg-color', colors.HexColor('#FFFFFF'))
    para_border_color = config.get('para-border-color', colors.HexColor('#FFFFFF'))
    
    first_line_indent = config.get('first-line-indent', 0)
    left_indent = config.get('left-indent', 0)
    right_indent = config.get('right-indent', 0)
    alignment = config.get('alignment', TA_JUSTIFY)
    space_before = config.get('space-before', 0)
    space_after = config.get('space-after', 0)
    border_width = config.get('border-width', 0)
    border_padding = config.get('border-padding', 0)
    
    horizontal_scale = config.get('horizontal-scale', 1.0)
    dpi = config.get('dpi', 72)
    auto_crop_last_page = config.get('auto-crop-last-page', False)
    auto_crop_width = config.get('auto-crop-width', False)
    # newline_markup = config.get('newline-markup', '<font color="#FF0000"> \\n </font>')
    newline_markup = config.get('newline-markup', '<br/>')
    
    # Register font
    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
    except:
        pass  # Font already registered
    
    # Create PDF
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=page_size,
        leftMargin=margin_x,
        rightMargin=margin_x,
        topMargin=margin_y,
        bottomMargin=margin_y,
    )
    
    # Create paragraph style
    styles = getSampleStyleSheet()
    RE_CJK = re.compile(r'[\u4E00-\u9FFF]')
    
    custom = ParagraphStyle(
        name="Custom",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=font_size,
        leading=line_height,
        textColor=font_color,
        backColor=para_bg_color,
        borderColor=para_border_color,
        borderWidth=border_width,
        borderPadding=border_padding,
        firstLineIndent=first_line_indent,
        wordWrap="CJK" if RE_CJK.search(text) else None,
        leftIndent=left_indent,
        rightIndent=right_indent,
        alignment=alignment,
        spaceBefore=space_before,
        spaceAfter=space_after,
    )
    
    # Process text
    def replace_spaces(s):
        return re.sub(r' {2,}', lambda m: '&nbsp;'*len(m.group()), s)
    
    text = text.replace('\xad', '').replace('\u200b', '')
    processed_text = replace_spaces(escape(text))
    parts = processed_text.split('\n')
    
    # Create paragraphs in batches
    story = []
    turns = 30
    for i in range(0, len(parts), turns):
        tmp_text = newline_markup.join(parts[i:i+turns])
        story.append(Paragraph(tmp_text, custom))
    
    # Build PDF
    doc.build(
        story,
        onFirstPage=lambda c, d: (c.saveState(), c.setFillColor(page_bg_color), c.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1), c.restoreState()),
        onLaterPages=lambda c, d: (c.saveState(), c.setFillColor(page_bg_color), c.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1), c.restoreState())
    )
    
    pdf_bytes = buf.getvalue()
    buf.close()
    
    # Create output directory (flat structure)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PDF to images
    info = pdfinfo_from_bytes(pdf_bytes)
    num_pages = total = info["Pages"]
    batch = 20
    image_paths = []
    
    for start in range(1, total + 1, batch):
        end = min(start + batch - 1, total)
        images = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=start, last_page=end)
        
        for offset, img in enumerate(images, start=start):
            w, h = img.size
            
            # Horizontal scaling
            if horizontal_scale != 1.0:
                img = img.resize((int(w * horizontal_scale), h))
            
            # Adaptive cropping
            if auto_crop_width or (auto_crop_last_page and offset == num_pages):
                gray = np.array(img.convert("L"))
                bg_gray = np.median(gray[:2, :2])
                tolerance = 5
                mask = np.abs(gray - bg_gray) > tolerance
                
                if auto_crop_width:
                    cols = np.where(mask.any(axis=0))[0]
                    if cols.size:
                        rightmost_col = cols[-1] + 1
                        right = min(img.width, rightmost_col + margin_x)
                        img = img.crop((0, 0, right, img.height))
                
                if auto_crop_last_page and offset == num_pages:
                    rows = np.where(mask.any(axis=1))[0]
                    if rows.size:
                        last_row = rows[-1]
                        lower = min(img.height, last_row + margin_y)
                        img = img.crop((0, 0, img.width, lower))
            
            # Generate representative filename
            out_path = os.path.join(output_dir, f"{unique_id}_page_{offset:03d}.png")
            img.save(out_path, 'PNG')
            image_paths.append(os.path.abspath(out_path))
            img.close()
        
        images.clear()
        del images
    
    del pdf_bytes
    gc.collect()
    
    return image_paths


def process_one(item):
    """Process single item - for batch processing"""
    global GLOBAL_CONFIG, OUTPUT_DIR, recover
    
    # Handle _id to unique_id renaming
    if 'unique_id' not in item and '_id' in item:
        item['unique_id'] = item.pop('_id')
    
    _id = item.get('unique_id')
    if not _id:
        warnings.warn(f"Item missing both 'unique_id' and '_id' fields")
        return item
    
    # Generate shorter name for recovery check
    domain = item.get('domain', 'unknown')
    
    # Domain mapping for shorter codes (create if not exists)
    if not hasattr(process_one, 'domain_map'):
        # Initialize domain mapping on first call
        process_one.domain_map = {
            'Long In-context Learning': 'd0',
            'Long Doc Understanding': 'd1',
    	    'Long Context QA': 'd2',
            'Long Summarization': 'd3',
            'Long Synthetic': 'd4',
            'Long Code Completion': 'd5'
        }
    
    domain_code = process_one.domain_map.get(domain, 'dx')
    short_id = _id[:8]
    representative_id = f"{domain_code}_{short_id}"
    
    # Check recovery mode - look for any existing images with this representative ID
    if recover:
        existing_images = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(representative_id) and f.endswith('.png')]
        if existing_images:
            item['image_paths'] = [os.path.join(OUTPUT_DIR, img) for img in sorted(existing_images)]
            return item
    
    # Parse configuration
    item_config = item.get('config', {}) or {}
    config = {**GLOBAL_CONFIG, **item_config}
    
    # Process special fields in item config
    if 'page-size' in item_config and isinstance(item_config['page-size'], str):
        config['page-size'] = tuple(map(float, item_config['page-size'].split(',')))
    if 'page-bg-color' in item_config and isinstance(item_config['page-bg-color'], str):
        config['page-bg-color'] = colors.HexColor(item_config['page-bg-color'])
    if 'font-color' in item_config and isinstance(item_config['font-color'], str):
        config['font-color'] = colors.HexColor(item_config['font-color'])
    if 'para-bg-color' in item_config and isinstance(item_config['para-bg-color'], str):
        config['para-bg-color'] = colors.HexColor(item_config['para-bg-color'])
    if 'para-border-color' in item_config and isinstance(item_config['para-border-color'], str):
        config['para-border-color'] = colors.HexColor(item_config['para-border-color'])
    if 'alignment' in item_config and isinstance(item_config['alignment'], str):
        config['alignment'] = ALIGN_MAP.get(item_config['alignment'], TA_JUSTIFY)
    
    # Get text
    text = item.get('context', '')
    assert text
    
    # Generate shorter name using domain mapping
    domain = item.get('domain', 'unknown')
    
    # Domain mapping for shorter codes (create if not exists)
    if not hasattr(process_one, 'domain_map'):
        # Initialize domain mapping on first call
        process_one.domain_map = {
            'Long In-context Learning': 'd0',
            'Long Doc Understanding': 'd1',
    	    'Long Context QA': 'd2',
            'Long Summarization': 'd3',
            'Long Synthetic': 'd4',
            'Long Code Completion': 'd5'
        }
    
    domain_code = process_one.domain_map.get(domain, 'dx')
    short_id = _id[:8]
    representative_id = f"{domain_code}_{short_id}"
    
    # Call inference function with representative name
    image_paths = text_to_images(
        text=text,
        output_dir=OUTPUT_DIR,
        config_dict=config,
        unique_id=representative_id
    )
    
    item['image_paths'] = image_paths
    return item


def batch_process_to_images(json_path, output_dir, output_jsonl_path, 
                            config_path, processes=16, is_recover=False, batch_size=100, limit=None):
    """Batch process JSON data to generate images"""
    global GLOBAL_CONFIG, OUTPUT_DIR, recover
    
    # Set global variables
    GLOBAL_CONFIG = load_config(config_path)
    OUTPUT_DIR = output_dir
    recover = is_recover
    
    print(f"Loaded config from: {config_path}")
    
    # Prepare output directory
    if not recover:
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(output_jsonl_path):
            os.remove(output_jsonl_path)
    
    # Read data and handle _id to unique_id renaming
    with open(json_path, 'r', encoding='utf-8') as f:
        data_to_process = json.load(f)
    
    # Rename _id to unique_id for all items in memory
    for item in data_to_process:
        if '_id' in item and 'unique_id' not in item:
            item['unique_id'] = item.pop('_id')
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        data_to_process = data_to_process[:limit]
        print(f"Limited processing to first {len(data_to_process)} items")
    
    # Get already processed IDs
    processed_ids = set()
    if recover and os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    processed_ids.add(item.get('unique_id'))
                except:
                    continue
        print(f"Found {len(processed_ids)} already processed items")
    
    # Filter processed items
    data_to_process = [item for item in data_to_process 
                      if item.get('unique_id') not in processed_ids]
    print(f"Remaining items to process: {len(data_to_process)}")
    
    if not data_to_process:
        print("All items processed")
        return
    
    # Parallel processing
    batch_buffer = []
    
    with Pool(processes=processes) as pool:
        for result_item in tqdm(pool.imap_unordered(process_one, data_to_process, chunksize=1), 
                               total=len(data_to_process)):
            if result_item:
                batch_buffer.append(result_item)
                _id = result_item.get('unique_id', 'UNKNOWN')
                count = len(result_item.get('image_paths', []))
                tqdm.write(f"{_id}: generated {count} pages")
                
                # Batch write
                if len(batch_buffer) >= batch_size:
                    with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                        for item in batch_buffer:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    batch_buffer = []
    
    # Write remaining items
    if batch_buffer:
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            for item in batch_buffer:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("Processing complete")


def main():
    parser = argparse.ArgumentParser(
        description='Convert LongBench-v2 text contexts to PNG images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process all samples
  python word2png_function.py
  
  # Process only 10 samples
  python word2png_function.py --limit 10
  
  # Use custom paths
  python word2png_function.py --data-json /path/to/data.json --output-dir /path/to/images
  
  # Resume interrupted processing
  python word2png_function.py --recover
        '''
    )
    
    parser.add_argument('--data-json', 
                       default='../data/longbenchv2/data.json',
                       help='Path to LongBench-v2 data.json file (default: ../data/longbenchv2/data.json)')
    
    parser.add_argument('--config',
                       default='../config/config.json',
                       help='Path to configuration file (default: ../config/config.json)')
    
    parser.add_argument('--output-dir',
                       default='../data/longbenchv2_img/images',
                       help='Directory to save generated images (default: ../data/longbenchv2_img/images)')
    
    parser.add_argument('--output-jsonl',
                       default='../data/longbenchv2_img/processed_output.jsonl',
                       help='Path to save processed output JSONL file (default: ../data/longbenchv2_img/processed_output.jsonl)')
    
    parser.add_argument('--limit',
                       type=int,
                       default=None,
                       help='Limit processing to N samples only (default: process all samples)')
    
    parser.add_argument('--processes',
                       type=int,
                       default=8,
                       help='Number of parallel processes to use (default: 8)')
    
    parser.add_argument('--batch-size',
                       type=int,
                       default=50,
                       help='Batch size for writing JSONL output (default: 50)')
    
    parser.add_argument('--recover',
                       action='store_true',
                       help='Resume from where processing left off')
    
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists(args.config):
        warnings.warn(
            f"Config file not found at {args.config}. "
            "Please create a config file with font-path and other settings."
        )
        sys.exit(1)
    
    # Check if data exists
    if not os.path.exists(args.data_json):
        warnings.warn(
            f"Data file not found at {args.data_json}. "
            "Please download it using: ./scripts/download_longbenchv2_raw.sh"
        )
        sys.exit(1)
    
    # Load data and apply limit if specified
    with open(args.data_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if args.limit is not None:
        print(f"Limiting processing to {args.limit} samples...")
        data = data[:args.limit]
        print(f"Processing {len(data)} samples")
    
    # Batch process to images
    print(f"Starting batch processing to images (limit: {args.limit or 'all'})...")
    batch_process_to_images(
        json_path=args.data_json,
        output_dir=args.output_dir,
        output_jsonl_path=args.output_jsonl,
        config_path=args.config,
        processes=args.processes,
        is_recover=args.recover,
        batch_size=args.batch_size,
        limit=args.limit
    )
    
    print("Processing complete. Images saved to:", args.output_dir)


if __name__ == '__main__':
    main()