"""
Adversarial rendering wrapper that applies config perturbations and outputs to dedicated folders
"""

import os
import json
import warnings
import argparse
import tempfile

# Import once at module level
from robust_ocm.render.cli import batch_process_to_images


def apply_perturbation_to_config(base_config, perturbation_type, perturbation_params):
    """
    Apply perturbation to the base config dict.
    
    Args:
        base_config: Base configuration dictionary
        perturbation_type: Type of perturbation to apply
        perturbation_params: Parameters for the perturbation
    
    Returns:
        Modified configuration dictionary
    """
    config = base_config.copy()
    
    if perturbation_type == 'dense_text':
        font_size = perturbation_params.get('font_size', 8)
        config['font-size'] = font_size
        config['line-height'] = font_size - 1
    elif perturbation_type == 'dpi_downscale':
        dpi = perturbation_params.get('dpi', 72)
        config['dpi'] = dpi
    elif perturbation_type == 'tofu':
        config['font-path'] = base_config.get('font-path')
    
    return config


def main():
    """Main CLI entry point for adversarial rendering"""
    parser = argparse.ArgumentParser(
        description='Generate adversarial text images with perturbations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate dense text samples (small font + tight line spacing)
  adv-render --perturbation-type dense_text --font-size 8
  
  # Generate DPI downscaled samples
  adv-render --perturbation-type dpi_downscale --dpi 72
  
  # Generate tofu samples (missing characters)
  adv-render --perturbation-type tofu
  
  # Generate samples in a task subfolder
  adv-render --perturbation-type dense_text --font-size 8 --task ocr
        '''
    )
    
    parser.add_argument('--data-json', 
                       default='./data/longbenchv2/data.json',
                       help='Path to LongBench-v2 data.json file')
    
    parser.add_argument('--config',
                       default='./config/config_general.json',
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
    
    # Task argument
    parser.add_argument('--task',
                       default=None,
                       help='Task subfolder in data/ (e.g., ocr, vqa, etc.)')
    
    # Perturbation arguments
    parser.add_argument('--perturbation-type',
                       required=True,
                       choices=['dense_text', 'tofu', 'dpi_downscale'],
                       help='Type of text perturbation to apply')
    
    # Perturbation parameters
    parser.add_argument('--font-size', type=int, default=8, help='Target font size for dense_text perturbation (default 8 points)')
    parser.add_argument('--dpi', type=int, default=72, help='Target DPI value for dpi_downscale perturbation (default 72)')
    
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
        if args.perturbation_type == 'dense_text':
            perturbation_suffix = f"fontsize_{args.font_size}"
        elif args.perturbation_type == 'dpi_downscale':
            perturbation_suffix = f"dpi_{args.dpi}"
        
        # Include task subfolder if specified
        task_prefix = f'{args.task}/' if args.task else ''
        args.output_dir = f'./data/{task_prefix}adv_{perturbation_suffix}/images'
        args.output_jsonl = f'./data/{task_prefix}adv_{perturbation_suffix}/line_bbox.jsonl'
    
    # Load base config directly from JSON to ensure it's JSON-serializable
    with open(args.config, 'r', encoding='utf-8') as f:
        base_config = json.load(f)
    
    # Apply markdown mode if needed
    if args.markdown_mode:
        base_config['markdown-mode'] = True
        print("Markdown mode enabled")
    
    # Collect perturbation parameters
    perturbation_params = {}
    if args.perturbation_type == 'dense_text':
        perturbation_params['font_size'] = args.font_size
    elif args.perturbation_type == 'dpi_downscale':
        perturbation_params['dpi'] = args.dpi
    
    # Apply perturbation to config
    config = apply_perturbation_to_config(base_config, args.perturbation_type, perturbation_params)
    
    # Save metadata
    metadata = {
        'perturbation_type': args.perturbation_type,
        'params': perturbation_params,
        'input_dir': args.data_json,
        'output_dir': args.output_dir,
        'output_jsonl': args.output_jsonl,
        'config': args.config,
        'markdown_mode': args.markdown_mode,
        'task': args.task
    }
    metadata_path = os.path.join(os.path.dirname(args.output_dir), 'metadata.json')
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Starting adversarial rendering with {args.perturbation_type}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output JSONL: {args.output_jsonl}")
    
    # Save modified config to temporary file for workers
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_config:
        json.dump(config, temp_config)
        temp_config_path = temp_config.name
    
    try:
        # Call the standard render batch_process_to_images function
        batch_process_to_images(
            json_path=args.data_json,
            output_dir=args.output_dir,
            output_jsonl_path=args.output_jsonl,
            config_path=temp_config_path,
            processes=args.processes,
            is_recover=args.recover,
            batch_size=args.batch_size,
            limit=args.limit,
            extraction_level=args.extraction_level,
            blacklist_path=args.blacklist
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