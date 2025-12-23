"""
Analyze the output of robust_ocm.render with bbox extraction
"""

import os
import json
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm


def main():
    """CLI for analyzing render output"""
    parser = argparse.ArgumentParser(
        description='Analyze output from robust_ocm.render with bbox extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Analyze all samples
  analyze --input ./data/longbenchv2_img/line_bbox.jsonl
  
  # Analyze with detailed statistics
  analyze --input ./data/longbenchv2_img/line_bbox.jsonl --detailed
  
  # Analyze specific sample
  analyze --input ./data/longbenchv2_img/line_bbox.jsonl --sample-id 66fcffd9bb02136c067c94c5
        '''
    )
    
    parser.add_argument('--input',
                       required=True,
                       help='Path to processed_output.jsonl file')
    
    parser.add_argument('--sample-id',
                       help='Analyze only this specific sample ID')
    
    parser.add_argument('--detailed',
                       action='store_true',
                       help='Show detailed statistics per sample')
    
    parser.add_argument('--export-stats',
                       help='Export statistics to JSON file')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Statistics
    stats = {
        'total_samples': 0,
        'total_pages': 0,
        'total_bboxes': 0,
        'avg_pages_per_sample': 0,
        'avg_bboxes_per_page': 0,
        'extraction_level': None,
        'domains': defaultdict(int),
        'bbox_lengths': [],
        'samples': []
    }
    
    # Process samples
    with open(args.input, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Analyzing samples")):
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line {line_num + 1}")
                continue
            
            sample_id = item.get('unique_id')
            if not sample_id:
                continue
            
            # Filter by sample ID if specified
            if args.sample_id and sample_id != args.sample_id:
                continue
            
            # Get data
            domain = item.get('domain', 'unknown')
            image_paths = item.get('image_paths', [])
            all_bboxes = item.get('bboxes', [])
            
            # Update stats
            stats['total_samples'] += 1
            stats['total_pages'] += len(image_paths)
            stats['domains'][domain] += 1
            
            # Count bboxes
            sample_bbox_count = 0
            for page_bboxes in all_bboxes:
                stats['total_bboxes'] += len(page_bboxes)
                sample_bbox_count += len(page_bboxes)
                
                # Collect bbox text lengths
                for bbox in page_bboxes:
                    if len(bbox) >= 5:
                        text = bbox[4]
                        stats['bbox_lengths'].append(len(text))
            
            # Store sample info for detailed output
            sample_info = {
                'sample_id': sample_id,
                'domain': domain,
                'pages': len(image_paths),
                'bboxes': sample_bbox_count,
                'avg_bboxes_per_page': sample_bbox_count / len(image_paths) if image_paths else 0
            }
            stats['samples'].append(sample_info)
            
            # Show detailed info if requested
            if args.detailed:
                print(f"\nSample {sample_id} ({domain}):")
                print(f"  Pages: {len(image_paths)}")
                print(f"  Total bboxes: {sample_bbox_count}")
                if image_paths:
                    print(f"  Avg bboxes per page: {sample_bbox_count / len(image_paths):.2f}")
    
    # Calculate averages
    if stats['total_samples'] > 0:
        stats['avg_pages_per_sample'] = stats['total_pages'] / stats['total_samples']
    if stats['total_pages'] > 0:
        stats['avg_bboxes_per_page'] = stats['total_bboxes'] / stats['total_pages']
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Total pages: {stats['total_pages']}")
    print(f"Total bounding boxes: {stats['total_bboxes']}")
    print(f"Average pages per sample: {stats['avg_pages_per_sample']:.2f}")
    print(f"Average bboxes per page: {stats['avg_bboxes_per_page']:.2f}")
    
    # Domain distribution
    print("\nDomain distribution:")
    for domain, count in sorted(stats['domains'].items()):
        percentage = (count / stats['total_samples']) * 100
        print(f"  {domain}: {count} samples ({percentage:.1f}%)")
    
    # Bbox text length statistics
    if stats['bbox_lengths']:
        print("\nBounding box text length statistics:")
        print(f"  Min: {min(stats['bbox_lengths'])}")
        print(f"  Max: {max(stats['bbox_lengths'])}")
        print(f"  Mean: {np.mean(stats['bbox_lengths']):.2f}")
        print(f"  Median: {np.median(stats['bbox_lengths']):.2f}")
        print(f"  95th percentile: {np.percentile(stats['bbox_lengths'], 95):.2f}")
    
    # Detailed per-sample info
    if args.detailed and stats['samples']:
        print("\nPer-sample details:")
        print("-" * 60)
        for sample in sorted(stats['samples'], key=lambda x: x['bboxes'], reverse=True)[:10]:
            print(f"{sample['sample_id'][:16]}... ({sample['domain'][:20]}): "
                  f"{sample['pages']} pages, {sample['bboxes']} bboxes "
                  f"({sample['avg_bboxes_per_page']:.1f}/page)")
    
    # Export statistics if requested
    if args.export_stats:
        # Convert defaultdict to regular dict for JSON serialization
        export_data = {
            'total_samples': stats['total_samples'],
            'total_pages': stats['total_pages'],
            'total_bboxes': stats['total_bboxes'],
            'avg_pages_per_sample': stats['avg_pages_per_sample'],
            'avg_bboxes_per_page': stats['avg_bboxes_per_page'],
            'domains': dict(stats['domains']),
            'bbox_length_stats': {
                'min': min(stats['bbox_lengths']) if stats['bbox_lengths'] else 0,
                'max': max(stats['bbox_lengths']) if stats['bbox_lengths'] else 0,
                'mean': float(np.mean(stats['bbox_lengths'])) if stats['bbox_lengths'] else 0,
                'median': float(np.median(stats['bbox_lengths'])) if stats['bbox_lengths'] else 0,
                'p95': float(np.percentile(stats['bbox_lengths'], 95)) if stats['bbox_lengths'] else 0
            },
            'samples': stats['samples']
        }
        
        with open(args.export_stats, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics exported to: {args.export_stats}")
    
    return 0


if __name__ == '__main__':
    exit(main())