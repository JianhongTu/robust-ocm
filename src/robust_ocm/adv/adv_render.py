"""
Adversarial rendering wrapper that applies text perturbations and outputs to dedicated folders
"""

import os
import json
import shutil
import warnings
import argparse
from multiprocessing import Pool
from tqdm import tqdm

# Import once at module level
from robust_ocm.render.cli import process_one_item, DOMAIN_MAP
from robust_ocm.render.config import Config
from robust_ocm.adv import apply_perturbation


def adv_process_one_item(args):
    """Process single item with adversarial perturbations - optimized version"""
    item, output_dir, config_dict, extraction_level, perturbation_type, perturbation_params = args
    
    # Apply perturbation directly without extra function calls
    if perturbation_type:
        if perturbation_type in ['kerning_collisions', 'line_height_compression']:
            # These modify the config - apply once per item
            config_dict = apply_perturbation(config_dict, perturbation_type, **perturbation_params)
        elif perturbation_type in ['font_weight', 'homoglyph_substitution']:
            # These modify the text
            if 'context' in item:
                item['context'] = apply_perturbation(item['context'], perturbation_type, **perturbation_params)
    
    # Call process_one_item with correct parameters
    return process_one_item((item, output_dir, config_dict, extraction_level))


def adv_batch_process_to_images(json_path, output_dir, output_jsonl_path, 
                               config_path, processes=8, is_recover=False, 
                               batch_size=50, limit=None, extraction_level="line",
                               blacklist_path=None, perturbation_type=None, perturbation_params=None):
    """Batch process JSON data to generate adversarial images"""
    
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
        if 'unique_id' not in item and '_id' in item:
            item['unique_id'] = item.pop('_id')
    
    # Apply blacklist if provided
    if blacklist_path and os.path.exists(blacklist_path):
        with open(blacklist_path, 'r', encoding='utf-8') as f:
            blacklist = set(line.strip() for line in f if line.strip())
        original_count = len(data_to_process)
        data_to_process = [item for item in data_to_process if item.get('unique_id') not in blacklist]
        print(f"Filtered {original_count - len(data_to_process)} items using blacklist")
    
    # Apply limit if specified
    if limit is not None:
        print(f"Limiting processing to {limit} samples...")
        data_to_process = data_to_process[:limit]
    
    if not data_to_process:
        print("All items processed")
        return
    
    # Prepare arguments for multiprocessing - simplified
    process_args = [(item, output_dir, config, extraction_level, perturbation_type, perturbation_params or {}) for item in data_to_process]
    
    # Parallel processing with optimized chunk size
    batch_buffer = []
    
    # Use larger chunk size for better performance
    chunk_size = max(1, len(process_args) // (processes * 4))
    
    with Pool(processes=processes) as pool:
        for result_item, bbox_count in tqdm(pool.imap_unordered(adv_process_one_item, process_args, chunksize=chunk_size), 
                                           total=len(data_to_process), desc="Processing adversarial samples"):
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
    
    print("Adversarial processing complete")


def main():
    """Main CLI entry point for adversarial rendering"""
    parser = argparse.ArgumentParser(
        description='Generate adversarial text images with perturbations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate bold text adversarial samples
  adv-render --perturbation-type font_weight --weight bold --limit 100
  
  # Generate homoglyph substitution samples
  adv-render --perturbation-type homoglyph_substitution --substitution-rate 0.2
  
  # Generate line height compressed samples
  adv-render --perturbation-type line_height_compression --compression-factor 0.8
        '''
    )
    
    parser.add_argument('--data-json', 
                       default='./data/longbenchv2/data.json',
                       help='Path to LongBench-v2 data.json file')
    
    parser.add_argument('--config',
                       default='./config/config_en.json',
                       help='Path to configuration file')
    
    parser.add_argument('--output-dir',
                       help='Output directory (auto-generated if not specified)')
    
    parser.add_argument('--output-jsonl',
                       help='Path to save processed output JSONL file (auto-generated if not specified)')
    
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
    
    # Perturbation arguments
    parser.add_argument('--perturbation-type',
                       required=True,
                       choices=['font_weight', 'kerning_collisions', 'homoglyph_substitution', 'line_height_compression'],
                       help='Type of text perturbation to apply')
    
    # Perturbation parameters
    parser.add_argument('--weight', default='bold', help='Font weight for font_weight perturbation')
    parser.add_argument('--collision-factor', type=float, default=0.1, help='Collision factor for kerning_collisions')
    parser.add_argument('--substitution-rate', type=float, default=0.1, help='Substitution rate for homoglyph_substitution')
    parser.add_argument('--compression-factor', type=float, default=0.8, help='Compression factor for line_height_compression')
    
    # Markdown mode
    parser.add_argument('--markdown-mode', action='store_true', help='Enable markdown parsing for rich text formatting')
    
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
    
    # Generate automatic output paths if not specified
    if not args.output_dir:
        perturbation_suffix = args.perturbation_type
        if args.perturbation_type == 'font_weight':
            perturbation_suffix = f"font_weight_{args.weight}"
        elif args.perturbation_type == 'homoglyph_substitution':
            perturbation_suffix = f"homoglyph_{args.substitution_rate}"
        elif args.perturbation_type == 'kerning_collisions':
            perturbation_suffix = f"kerning_{args.collision_factor}"
        elif args.perturbation_type == 'line_height_compression':
            perturbation_suffix = f"line_height_{args.compression_factor}"
        
        args.output_dir = f'./data/adv_{perturbation_suffix}/images'
        args.output_jsonl = f'./data/adv_{perturbation_suffix}/line_bbox.jsonl'
    
    # Load and modify config for markdown mode if needed - do this once
    config = Config.load_config(args.config)
    if args.markdown_mode:
        config['markdown-mode'] = True
        print("Markdown mode enabled")
    
    # Save config to temporary file to avoid reloading
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_config:
        json.dump(config, temp_config)
        temp_config_path = temp_config.name
    
    try:
        # Collect perturbation parameters
        perturbation_params = {}
        if args.perturbation_type == 'font_weight':
            perturbation_params['weight'] = args.weight
        elif args.perturbation_type == 'kerning_collisions':
            perturbation_params['collision_factor'] = args.collision_factor
        elif args.perturbation_type == 'homoglyph_substitution':
            perturbation_params['substitution_rate'] = args.substitution_rate
        elif args.perturbation_type == 'line_height_compression':
            perturbation_params['compression_factor'] = args.compression_factor
        
        # Save metadata
        metadata = {
            'perturbation_type': args.perturbation_type,
            'params': perturbation_params,
            'input_dir': args.data_json,
            'output_dir': args.output_dir,
            'output_jsonl': args.output_jsonl,
            'config': args.config,
            'markdown_mode': args.markdown_mode
        }
        metadata_path = os.path.join(os.path.dirname(args.output_dir), 'metadata.json')
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Starting adversarial rendering with {args.perturbation_type}")
        print(f"Output directory: {args.output_dir}")
        print(f"Output JSONL: {args.output_jsonl}")
        
        # Batch process to images with optimized config
        adv_batch_process_to_images(
            json_path=args.data_json,
            output_dir=args.output_dir,
            output_jsonl_path=args.output_jsonl,
            config_path=temp_config_path,  # Use temp config
            processes=args.processes,
            is_recover=args.recover,
            batch_size=args.batch_size,
            limit=args.limit,
            extraction_level=args.extraction_level,
            blacklist_path=args.blacklist,
            perturbation_type=args.perturbation_type,
            perturbation_params=perturbation_params
        )
        
        print(f"Adversarial processing complete. Images saved to: {args.output_dir}")
        print(f"Metadata saved to: {metadata_path}")
        return 0
    
    finally:
        # Clean up temporary config file
        try:
            os.unlink(temp_config_path)
        except:
            pass


if __name__ == '__main__':
    exit(main())