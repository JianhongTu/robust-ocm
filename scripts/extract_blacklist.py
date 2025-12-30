#!/usr/bin/env python3
"""
Extract IDs of top N% longest samples and add to blacklist file
"""

import json
import os
from pathlib import Path
import numpy as np
import argparse

def count_words(text):
    """Count words in text, handling various delimiters"""
    if not text:
        return 0
    
    # Split on whitespace and common punctuation
    words = []
    for word in text.split():
        # Remove common punctuation from word boundaries
        clean_word = word.strip('.,;:!?()[]{}"\'`')
        if clean_word:
            words.append(clean_word)
    
    return len(words)

def extract_longest_samples(data_file, output_file='blacklist.txt', threshold_percent=25.0):
    """Extract IDs of top N% longest samples"""
    print(f"Loading data from {data_file}...")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Collect word counts and IDs
    entries_with_word_counts = []
    
    print("Processing entries...")
    for i, entry in enumerate(data):
        # Check for text field or use context field
        text = None
        if 'text' in entry:
            text = entry['text']
        elif 'context' in entry:
            text = entry['context']
        elif 'question' in entry:
            text = entry['question']
        
        if text:
            word_count = count_words(text)
            entries_with_word_counts.append({
                'id': entry.get('_id', f'entry_{i}'),
                'word_count': word_count,
                'index': i
            })
    
    if not entries_with_word_counts:
        print("No text content found in the data!")
        return
    
    # Sort by word count (descending)
    entries_sorted = sorted(entries_with_word_counts, key=lambda x: x['word_count'], reverse=True)
    
    # Calculate threshold for top N%
    total_entries = len(entries_sorted)
    top_n_count = int(total_entries * (threshold_percent / 100.0))
    
    print(f"\nTotal entries with text: {total_entries}")
    print(f"Top {threshold_percent}% threshold: {top_n_count} entries")
    
    # Get top N% entries
    top_n_entries = entries_sorted[:top_n_count]
    
    # Calculate threshold word count
    if top_n_entries:
        threshold_word_count = top_n_entries[-1]['word_count']
        print(f"Word count threshold for top {threshold_percent}%: {threshold_word_count} words")
    else:
        print(f"No entries found for top {threshold_percent}%")
        return []
    
    # Extract IDs
    top_ids = [entry['id'] for entry in top_n_entries]
    
    # Read existing blacklist if it exists
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(line)
        print(f"Found {len(existing_ids)} existing IDs in {output_file}")
    
    # Combine existing and new IDs
    all_ids = existing_ids.union(set(top_ids))
    
    # Write to blacklist file
    with open(output_file, 'w') as f:
        for id_str in sorted(all_ids):
            f.write(f"{id_str}\n")
    
    print(f"\nAdded {len(top_ids)} new IDs to {output_file}")
    print(f"Total IDs in blacklist: {len(all_ids)}")
    
    # Show some statistics
    print(f"\n=== Top 10 Longest Entries ===")
    for i, entry in enumerate(top_n_entries[:10]):
        print(f"{i+1}. ID: {entry['id']}, Words: {entry['word_count']:,}")
    
    print(f"\n=== Sample of Newly Blacklisted IDs ===")
    for i, id_str in enumerate(list(top_ids)[:10]):
        print(f"{i+1}. {id_str}")
    
    return top_ids


def extract_domain_samples(data_file, domain, output_file='blacklist.txt', threshold_percent=None):
    """Extract IDs to blacklist
    
    Args:
        data_file: Path to data.json file
        domain: Domain name to blacklist (ALL samples from this domain will be blacklisted)
        output_file: Output blacklist file path
        threshold_percent: If specified, also blacklist top N% longest from ALL domains
    """
    print(f"Loading data from {data_file}...")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Collect all entries with word counts
    all_entries = []
    domain_entries = []
    
    print(f"Processing entries...")
    
    for i, entry in enumerate(data):
        entry_domain = entry.get('domain', 'unknown')
        
        # Get text for word counting
        text = None
        if 'text' in entry:
            text = entry['text']
        elif 'context' in entry:
            text = entry['context']
        elif 'question' in entry:
            text = entry['question']
        
        word_count = count_words(text) if text else 0
        
        entry_data = {
            'id': entry.get('_id', f'entry_{i}'),
            'domain': entry_domain,
            'sub_domain': entry.get('sub_domain', 'unknown'),
            'index': i,
            'word_count': word_count
        }
        
        all_entries.append(entry_data)
        
        if entry_domain == domain:
            domain_entries.append(entry_data)
    
    if not domain_entries:
        print(f"No entries found with domain '{domain}'!")
        return []
    
    print(f"\nFound {len(domain_entries)} entries with domain '{domain}'")
    print(f"Total entries in dataset: {len(all_entries)}")
    
    # Start with all domain entries (will be blacklisted)
    domain_ids = set(entry['id'] for entry in domain_entries)
    entries_to_blacklist = domain_entries.copy()
    
    # If threshold is specified, also blacklist top N% longest from ALL domains
    if threshold_percent is not None:
        print(f"\nApplying threshold: top {threshold_percent}% longest samples from ALL domains...")
        
        # Sort all entries by word count (descending)
        all_sorted = sorted(all_entries, key=lambda x: x['word_count'], reverse=True)
        
        # Calculate threshold for top N%
        total_entries = len(all_sorted)
        top_n_count = int(total_entries * (threshold_percent / 100.0))
        
        print(f"Top {threshold_percent}% threshold: {top_n_count} entries out of {total_entries}")
        
        # Get top N% entries from all domains
        top_n_entries = all_sorted[:top_n_count]
        
        if top_n_entries:
            threshold_word_count = top_n_entries[-1]['word_count']
            print(f"Word count threshold for top {threshold_percent}%: {threshold_word_count} words")
        
        print(f"\nAdding {len(top_n_entries)} entries from top {threshold_percent}% longest to blacklist")
        
        # Show top entries
        print(f"\n=== Top 10 Longest Entries ===")
        for i, entry in enumerate(top_n_entries[:10]):
            print(f"{i+1}. ID: {entry['id']}, Words: {entry['word_count']:,}, Domain: {entry['domain']}")
        
        # Combine domain entries and top N% entries (use set to avoid duplicates)
        combined_ids = set(entry['id'] for entry in domain_entries + top_n_entries)
        
        # Rebuild entries_to_blacklist from combined IDs
        id_to_entry = {entry['id']: entry for entry in all_entries}
        entries_to_blacklist = [id_to_entry[id] for id in combined_ids if id in id_to_entry]
    else:
        print(f"\nWill blacklist all {len(domain_entries)} entries in domain '{domain}'")
        
        # Show distribution by sub-domain
        sub_domain_counts = {}
        for entry in domain_entries:
            sub = entry['sub_domain']
            sub_domain_counts[sub] = sub_domain_counts.get(sub, 0) + 1
        
        print("\nDistribution by sub-domain:")
        for sub, count in sorted(sub_domain_counts.items()):
            print(f"  {sub}: {count} samples")
    
    # Extract IDs
    ids = [entry['id'] for entry in entries_to_blacklist]
    
    # Read existing blacklist if it exists
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(line)
        print(f"\nFound {len(existing_ids)} existing IDs in {output_file}")
    
    # Combine existing and new IDs
    all_ids = existing_ids.union(set(ids))
    
    # Write to blacklist file
    with open(output_file, 'w') as f:
        for id_str in sorted(all_ids):
            f.write(f"{id_str}\n")
    
    print(f"\nAdded {len(ids)} new IDs to {output_file}")
    print(f"Total IDs in blacklist: {len(all_ids)}")
    
    # Show some statistics
    print(f"\n=== Sample of Newly Blacklisted IDs ===")
    for i, id_str in enumerate(list(ids)[:10]):
        entry = entries_to_blacklist[i]
        print(f"{i+1}. ID: {id_str}, Words: {entry['word_count']:,}, Sub-domain: {entry['sub_domain']}")
    
    return ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract IDs of samples to add to blacklist file. Can filter by domain or by longest samples.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Blacklist all samples under a specific domain
  %(prog)s --domain "Long Structured Data Understanding"
  
  # Blacklist all samples from domain X AND top 50%% longest from ALL domains
  %(prog)s --domain "Long Structured Data Understanding" --threshold 50
  
  # Blacklist top 25%% longest samples from entire dataset
  %(prog)s --threshold 25
  
  # Use custom output file
  %(prog)s --domain "Long Structured Data Understanding" --output domain_blacklist.txt
        '''
    )
    parser.add_argument('--threshold', '-t', type=float, default=25.0,
                       help='Percentage of longest samples to blacklist (default: 25.0). If --domain is specified, applies threshold to ALL domains.')
    parser.add_argument('--domain', '-D', type=str, default=None,
                       help='Domain to blacklist (ALL samples from this domain). If --threshold is also specified, blacklists this domain AND top N%% longest from ALL domains.')
    parser.add_argument('--output', '-o', type=str, default='blacklist.txt',
                       help='Output blacklist file name (default: blacklist.txt)')
    parser.add_argument('--data-file', '-d', type=str,
                       help='Path to data.json file (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Find data.json file if not specified
    data_file = args.data_file
    if not data_file:
        possible_paths = [
            "data/longbenchv2/data.json",
            "data.json",
            "data/data.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_file = path
                break
    
    if not data_file:
        print("Error: Could not find data.json file")
        print("Please specify with --data-file or ensure data.json is in one of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        exit(1)
    
    # Run extraction with specified parameters
    if args.domain:
        extract_domain_samples(data_file, args.domain, args.output, args.threshold)
    else:
        extract_longest_samples(data_file, args.output, args.threshold)