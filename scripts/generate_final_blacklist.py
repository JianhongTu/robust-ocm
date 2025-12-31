#!/usr/bin/env python3
"""Generate final_blacklist.txt with instances to exclude (>64K or Long Structured Data Understanding).

Usage:
    python scripts/generate_final_blacklist.py
"""

import json
import tiktoken
import os
from pathlib import Path

tokenizer = tiktoken.get_encoding('cl100k_base')

# Get paths relative to workspace root
workspace_root = Path(__file__).parent.parent
data_file = workspace_root / 'data' / 'longbenchv2' / 'data.json'
output_file = workspace_root / 'final_blacklist.txt'

def main():
    """Generate final_blacklist.txt with instances to exclude."""
    # Read the data
    with open(data_file, 'r') as f:
        data = json.load(f)

    # Find instances to exclude
    # Exclude if: >64K tokens OR from "Long Structured Data Understanding"
    excluded_ids = []
    kept_ids = []

    for item in data:
        if 'context' in item:
            context = item['context']
            if isinstance(context, str):
                token_count = len(tokenizer.encode(context, disallowed_special=()))
                task = item.get('domain', 'unknown')
                instance_id = item.get('_id', item.get('id', 'unknown'))

                # Exclude if >64K or from "Long Structured Data Understanding"
                if token_count > 64000 or task == 'Long Structured Data Understanding':
                    excluded_ids.append(instance_id)
                else:
                    kept_ids.append(instance_id)

    # Write to blacklist file
    with open(output_file, 'w') as f:
        for instance_id in sorted(excluded_ids):
            f.write(instance_id + '\n')

    print(f'Total instances: {len(data)}')
    print(f'Kept (<=64K, not Long Structured Data Understanding): {len(kept_ids)}')
    print(f'Excluded (>64K or Long Structured Data Understanding): {len(excluded_ids)}')
    print(f'\nBlacklist written to: {output_file}')

    # Show distribution by exclusion reason
    excluded_by_reason = {'token_count': 0, 'task': 0}

    for item in data:
        if 'context' in item:
            context = item['context']
            if isinstance(context, str):
                token_count = len(tokenizer.encode(context, disallowed_special=()))
                task = item.get('domain', 'unknown')

                if token_count > 64000 and task == 'Long Structured Data Understanding':
                    excluded_by_reason['token_count'] += 1
                    excluded_by_reason['task'] += 1
                elif token_count > 64000:
                    excluded_by_reason['token_count'] += 1
                elif task == 'Long Structured Data Understanding':
                    excluded_by_reason['task'] += 1

    print('\nExclusion breakdown:')
    print(f'  Excluded due to token count (>64K): {excluded_by_reason["token_count"]}')
    print(f'  Excluded due to task (Long Structured Data Understanding): {excluded_by_reason["task"]}')
    print(f'  Note: Some instances may be counted in both categories')

if __name__ == '__main__':
    main()