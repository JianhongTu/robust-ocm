#!/usr/bin/env python3
"""
Verify NIAH Dataset - Check that keys and values don't appear in original context

This script verifies that the secret keys and values inserted into NIAH instances
do not already appear in the original context before insertion.
"""

import json
import argparse
import re
from typing import Dict, Tuple


def check_instance(instance: Dict) -> Tuple[bool, list]:
    """
    Check if keys and values appear in the original context.
    
    Args:
        instance: NIAH instance dictionary
    
    Returns:
        Tuple of (is_clean, issues_list)
    """
    # Get the modified context and remove inserted secrets to get original
    context = instance.get('context', '')
    original_context = re.sub(r'\n\nThe secret \w+ is [\w\s]+\.\n\n', '', context)
    original_context_lower = original_context.lower()
    
    secrets = instance.get('secrets', {})
    issues = []
    
    for key, value in secrets.items():
        # Check if key appears as a complete word
        key_pattern = r'\b' + re.escape(key.lower()) + r's?\b'  # Allow plural
        key_in_context = bool(re.search(key_pattern, original_context_lower))
        
        # Check if value appears as a complete word
        value_pattern = r'\b' + re.escape(value.lower()) + r's?\b'  # Allow plural
        value_in_context = bool(re.search(value_pattern, original_context_lower))
        
        if key_in_context:
            issues.append(f"key '{key}' found in context")
        if value_in_context:
            issues.append(f"value '{value}' found in context")
    
    return len(issues) == 0, issues


def main():
    parser = argparse.ArgumentParser(
        description='Verify NIAH dataset keys and values are not in original context',
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
        help='Show details for all instances, not just issues'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Only check first N instances'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading NIAH dataset from {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    if args.limit:
        data = data[:args.limit]
        print(f"Checking first {len(data)} instances")
    else:
        print(f"Checking all {len(data)} instances")
    
    print("\n" + "=" * 80)
    print("Verifying keys and values don't appear in original context...")
    print("=" * 80 + "\n")
    
    # Check each instance
    clean_count = 0
    issue_count = 0
    all_issues = []
    
    for i, instance in enumerate(data):
        instance_id = instance.get('_id', f'instance_{i}')
        is_clean, issues = check_instance(instance)
        
        if is_clean:
            clean_count += 1
            if args.verbose:
                print(f"✓ Instance {i+1}/{len(data)} ({instance_id[:12]}...): Clean")
        else:
            issue_count += 1
            print(f"✗ Instance {i+1}/{len(data)} ({instance_id[:12]}...):")
            for issue in issues:
                print(f"    - {issue}")
            all_issues.extend(issues)
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total instances checked: {len(data)}")
    print(f"Clean instances: {clean_count} ({clean_count/len(data)*100:.1f}%)")
    print(f"Instances with issues: {issue_count} ({issue_count/len(data)*100:.1f}%)")
    
    if all_issues:
        print(f"\nTotal issues found: {len(all_issues)}")
        
        # Count issue types
        key_issues = sum(1 for issue in all_issues if 'key' in issue)
        value_issues = sum(1 for issue in all_issues if 'value' in issue)
        print(f"  - Key collisions: {key_issues}")
        print(f"  - Value collisions: {value_issues}")
    
    if issue_count == 0:
        print("\n✓ All instances have clean keys and values!")
        return 0
    else:
        print(f"\n⚠️  {issue_count} instances have keys/values in original context")
        return 1


if __name__ == '__main__':
    exit(main())
