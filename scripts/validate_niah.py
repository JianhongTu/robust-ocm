#!/usr/bin/env python3
"""
Validate NIAH Dataset - Check Needle Distribution

This script validates that each NIAH instance has exactly one needle (secret)
in each of the 5 regions: 0-20%, 20-40%, 40-60%, 60-80%, and 80-100%.
"""

import json
import argparse
import re
from typing import List, Tuple


def find_secret_positions(context: str) -> List[Tuple[str, str, int]]:
    """
    Find all secrets in the context and their positions.
    
    Args:
        context: The context text to search
    
    Returns:
        List of tuples (key, value, position) where position is character index
    """
    # Pattern to match: "The secret {key} is {value}."
    pattern = r'The secret (\w+) is ([\w\s]+)\.'
    matches = re.finditer(pattern, context)
    
    secrets = []
    for match in matches:
        key = match.group(1)
        value = match.group(2).strip()
        position = match.start()
        secrets.append((key, value, position))
    
    return secrets


def calculate_region(position: int, context_length: int) -> int:
    """
    Calculate which region (0-4) a position falls into.
    
    Args:
        position: Character position in context
        context_length: Total length of context
    
    Returns:
        Region index (0-4) for 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
    """
    percentage = position / context_length
    
    if percentage < 0.2:
        return 0
    elif percentage < 0.4:
        return 1
    elif percentage < 0.6:
        return 2
    elif percentage < 0.8:
        return 3
    else:
        return 4


def validate_instance(instance: dict) -> Tuple[bool, str]:
    """
    Validate that an instance has exactly one needle in each region.
    
    Args:
        instance: NIAH instance dictionary
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    context = instance.get('context', '')
    if not context:
        return False, "No context found"
    
    # Find all secrets
    secrets = find_secret_positions(context)
    
    if len(secrets) == 0:
        return False, "No secrets found in context"
    
    # Calculate regions for each secret
    context_length = len(context)
    region_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    region_secrets = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    for key, value, position in secrets:
        region = calculate_region(position, context_length)
        region_counts[region] += 1
        region_secrets[region].append(f"{key}={value} at pos {position}")
    
    # Check if each region has exactly one secret
    errors = []
    for region in range(5):
        count = region_counts[region]
        region_range = f"{region*20}-{(region+1)*20}%"
        
        if count == 0:
            errors.append(f"Region {region_range}: NO secrets found")
        elif count > 1:
            errors.append(f"Region {region_range}: {count} secrets found (expected 1)")
            for secret in region_secrets[region]:
                errors.append(f"  - {secret}")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, "OK"


def main():
    parser = argparse.ArgumentParser(
        description='Validate NIAH dataset needle distribution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Path to NIAH dataset JSON file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show details for all instances, not just errors'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading NIAH dataset from {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} instances\n")
    print("=" * 80)
    print("Validating needle distribution across 5 regions...")
    print("=" * 80)
    
    # Validate each instance
    valid_count = 0
    invalid_count = 0
    
    for i, instance in enumerate(data):
        instance_id = instance.get('_id', f'instance_{i}')
        is_valid, message = validate_instance(instance)
        
        if is_valid:
            valid_count += 1
            if args.verbose:
                print(f"✓ Instance {i+1}/{len(data)} ({instance_id}): {message}")
        else:
            invalid_count += 1
            print(f"✗ Instance {i+1}/{len(data)} ({instance_id}): {message}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total instances: {len(data)}")
    print(f"Valid instances: {valid_count} ({valid_count/len(data)*100:.1f}%)")
    print(f"Invalid instances: {invalid_count} ({invalid_count/len(data)*100:.1f}%)")
    
    if invalid_count == 0:
        print("\n✓ All instances have exactly one needle in each region!")
        return 0
    else:
        print(f"\n✗ {invalid_count} instances have incorrect needle distribution")
        return 1


if __name__ == '__main__':
    exit(main())
