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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract IDs of top N% longest samples and add to blacklist file')
    parser.add_argument('--threshold', '-t', type=float, default=25.0,
                       help='Percentage of longest samples to blacklist (default: 25.0)')
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
    extract_longest_samples(data_file, args.output, args.threshold)