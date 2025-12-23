"""
Command line interface for the robust_ocm render module
"""

import os
import json
import shutil
import warnings
import argparse
from multiprocessing import Pool
from tqdm import tqdm

try:
    from .render import TextRenderer
    from .config import Config
except ImportError:
    from robust_ocm.render.render import TextRenderer
    from robust_ocm.render.config import Config


# Domain mapping for shorter codes
DOMAIN_MAP = {
    'Long In-context Learning': 'd0',
    'Long Doc Understanding': 'd1',
    'Long Context QA': 'd2',
    'Long Summarization': 'd3',
    'Long Synthetic': 'd4',
    'Long Code Completion': 'd5'
}


def process_one_item(args):
    """Process single item - for batch processing"""
    item, output_dir, config_dict, extraction_level = args
    
    # Handle _id to unique_id renaming
    if 'unique_id' not in item and '_id' in item:
        item['unique_id'] = item.pop('_id')
    
    _id = item.get('unique_id')
    if not _id:
        warnings.warn(f"Item missing both 'unique_id' and '_id' fields")
        return item
    
    # Generate representative ID
    domain = item.get('domain', 'unknown')
    domain_code = DOMAIN_MAP.get(domain, 'dx')
    short_id = _id[:8]
    representative_id = f"{domain_code}_{short_id}"
    
    # Create renderer with merged config
    item_config = item.get('config', {}) or {}
    merged_config = Config.merge_configs(config_dict, item_config)
    
    renderer = TextRenderer(config_dict=merged_config)
    
    # Render the item
    try:
        result = renderer.render_batch_item(
            item=item,
            output_dir=output_dir,
            extraction_level=extraction_level
        )
        
        # Count bboxes for reporting
        bbox_count = 0
        if 'bboxes' in result:
            bbox_count = sum(len(page_bboxes) for page_bboxes in result['bboxes'])
        
        return result, bbox_count
    
    except Exception as e:
        warnings.warn(f"Failed to process item {_id}: {e}")
        return item, 0


def batch_process_to_images(json_path, output_dir, output_jsonl_path, 
                            config_path, processes=8, is_recover=False, 
                            batch_size=50, limit=None, extraction_level="line",
                            blacklist_path=None):
    """Batch process JSON data to generate images"""
    
    # Load configuration
    config = Config.load_config(config_path)
    
    print(f"Loaded config from: {config_path}")
    
    # Prepare output directory
    if not is_recover:
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
    if is_recover and os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    processed_ids.add(item.get('unique_id'))
                except:
                    continue
        print(f"Found {len(processed_ids)} already processed items")
    
    # Load blacklist if provided
    blacklist_ids = set()
    if blacklist_path and os.path.exists(blacklist_path):
        with open(blacklist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    blacklist_ids.add(line)
        print(f"Loaded {len(blacklist_ids)} blacklisted sample IDs")
    elif blacklist_path:
        print(f"Warning: Blacklist file {blacklist_path} not found")
    
    # Report blacklisted items that were found
    found_blacklisted = []
    for item in data_to_process:
        item_id = item.get('unique_id')
        if item_id in blacklist_ids:
            found_blacklisted.append(item_id)
    
    # Filter processed items and blacklisted items
    data_to_process = [item for item in data_to_process 
                      if item.get('unique_id') not in processed_ids 
                      and item.get('unique_id') not in blacklist_ids]
    
    if found_blacklisted:
        print(f"Skipping {len(found_blacklisted)} blacklisted items: {found_blacklisted[:5]}{'...' if len(found_blacklisted) > 5 else ''}")
    
    print(f"Remaining items to process: {len(data_to_process)}")
    
    if not data_to_process:
        print("All items processed")
        return
    
    # Prepare arguments for multiprocessing
    process_args = [(item, output_dir, config, extraction_level) for item in data_to_process]
    
    # Parallel processing
    batch_buffer = []
    
    with Pool(processes=processes) as pool:
        for result_item, bbox_count in tqdm(pool.imap_unordered(process_one_item, process_args, chunksize=1), 
                                           total=len(data_to_process)):
            if result_item:
                batch_buffer.append(result_item)
                _id = result_item.get('unique_id', 'UNKNOWN')
                count = len(result_item.get('image_paths', []))
                tqdm.write(f"{_id}: generated {count} pages, {bbox_count} bboxes")
                
                # Batch write
                if len(batch_buffer) >= batch_size:
                    with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                        for item in batch_buffer:
                            # Extract only essential fields
                            essential_data = {
                                'unique_id': item.get('unique_id'),
                                'image_paths': item.get('image_paths'),
                                'bboxes': item.get('bboxes')
                            }
                            f.write(json.dumps(essential_data, ensure_ascii=False) + '\n')
                    batch_buffer = []
    
    # Write remaining items
    if batch_buffer:
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            for item in batch_buffer:
                # Extract only essential fields
                essential_data = {
                    'unique_id': item.get('unique_id'),
                    'image_paths': item.get('image_paths'),
                    'bboxes': item.get('bboxes')
                }
                f.write(json.dumps(essential_data, ensure_ascii=False) + '\n')
    
    print("Processing complete")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Convert LongBench-v2 text contexts to PNG images with bounding box extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process all samples with line-level bbox extraction
  robust-ocm-render --data-json ../data/longbenchv2/data.json
  
  # Process only 10 samples with word-level extraction
  robust-ocm-render --limit 10 --extraction-level word
  
  # Use custom configuration
  robust-ocm-render --config ../config/config_en.json
  
  # Resume interrupted processing
  robust-ocm-render --recover
        '''
    )
    
    parser.add_argument('--data-json', 
                       default='./data/longbenchv2/data.json',
                       help='Path to LongBench-v2 data.json file')
    
    parser.add_argument('--config',
                       default='./config/config_en.json',
                       help='Path to configuration file')
    
    parser.add_argument('--output-dir',
                       default='./data/longbenchv2_img/images',
                       help='Directory to save generated images')
    
    parser.add_argument('--output-jsonl',
                       default='./data/longbenchv2_img/line_bbox.jsonl',
                       help='Path to save processed output JSONL file')
    
    parser.add_argument('--limit',
                       type=int,
                       default=None,
                       help='Limit processing to N samples only')
    
    parser.add_argument('--processes',
                       type=int,
                       default=8,
                       help='Number of parallel processes to use')
    
    parser.add_argument('--batch-size',
                       type=int,
                       default=50,
                       help='Batch size for writing JSONL output')
    
    parser.add_argument('--extraction-level',
                       default='line',
                       choices=['word', 'line'],
                       help='PDF bbox extraction level: word-level or line-level')
    
    parser.add_argument('--recover',
                       action='store_true',
                       help='Resume from where processing left off')
    
    parser.add_argument('--blacklist',
                       default=None,
                       help='Path to blacklist file with sample IDs to skip (one ID per line)')
    
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists(args.config):
        warnings.warn(
            f"Config file not found at {args.config}. "
            "Please create a config file with font-path and other settings."
        )
        return 1
    
    # Check if data exists
    if not os.path.exists(args.data_json):
        warnings.warn(
            f"Data file not found at {args.data_json}. "
            "Please download it using: ./scripts/download_longbenchv2_raw.sh"
        )
        return 1
    
    # Load data and apply limit if specified
    with open(args.data_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if args.limit is not None:
        print(f"Limiting processing to {args.limit} samples...")
        data = data[:args.limit]
        print(f"Processing {len(data)} samples")
    
    # Batch process to images
    print(f"Starting batch processing to images (limit: {args.limit or 'all'}, extraction: {args.extraction_level})...")
    batch_process_to_images(
        json_path=args.data_json,
        output_dir=args.output_dir,
        output_jsonl_path=args.output_jsonl,
        config_path=args.config,
        processes=args.processes,
        is_recover=args.recover,
        batch_size=args.batch_size,
        limit=args.limit,
        extraction_level=args.extraction_level,
        blacklist_path=args.blacklist
    )
    
    print("Processing complete. Images saved to:", args.output_dir)
    return 0


if __name__ == '__main__':
    exit(main())