#!/usr/bin/env python3
"""
Needle in a Haystack (NIAH) CLI Tool

This CLI generates synthetic NIAH tasks by inserting key-value pairs into documents
from a ground truth dataset and creating questions to retrieve those values.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple


# Define key-value categories with associated values
KEY_VALUE_CATEGORIES = {
    "number": ["42", "137", "256", "777", "1024", "314159", "9876", "2048", "365", "1337"],
    "animal": ["elephant", "dolphin", "penguin", "tiger", "falcon", "octopus", "koala", "lynx", "panda", "phoenix"],
    "flower": ["rose", "tulip", "orchid", "sunflower", "lily", "lotus", "jasmine", "dahlia", "iris", "peony"],
    "color": ["crimson", "azure", "amber", "violet", "coral", "indigo", "turquoise", "gold", "burgundy", "teal"],
    "city": ["Paris", "Tokyo", "Sydney", "Mumbai", "Cairo", "Seattle", "Barcelona", "Vienna", "Prague", "Kyoto"],
    "fruit": ["mango", "papaya", "lychee", "persimmon", "starfruit", "dragonfruit", "pomegranate", "kiwi", "fig", "guava"],
    "element": ["titanium", "platinum", "mercury", "cobalt", "silver", "copper", "iron", "zinc", "neon", "argon"],
    "planet": ["Mars", "Venus", "Jupiter", "Saturn", "Neptune", "Uranus", "Mercury", "Pluto", "Ceres", "Titan"],
    "instrument": ["violin", "saxophone", "piano", "flute", "harp", "cello", "trumpet", "guitar", "drums", "oboe"],
    "gemstone": ["diamond", "ruby", "sapphire", "emerald", "topaz", "amethyst", "jade", "opal", "pearl", "garnet"],
}


def load_blacklist(blacklist_path: str) -> set:
    """Load a blacklist file containing instance IDs to exclude."""
    if not blacklist_path or not Path(blacklist_path).exists():
        return set()
    
    with open(blacklist_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def find_sentence_boundaries(context: str) -> List[int]:
    """
    Find all sentence boundaries (positions after '.', '!', '?', or '\n\n').
    
    Args:
        context: The text to analyze
    
    Returns:
        List of character positions that mark sentence boundaries
    """
    import re
    # Match sentence endings: period, exclamation, question mark followed by space/newline
    # Also match paragraph breaks (double newlines)
    pattern = r'[.!?]+[\s]+|[\n]{2,}'
    
    boundaries = []
    for match in re.finditer(pattern, context):
        # Use the end position of the match (after the punctuation and whitespace)
        boundaries.append(match.end())
    
    return boundaries


def calculate_insertion_positions(context: str, num_regions: int = 5) -> List[int]:
    """
    Calculate insertion positions for key-value pairs across the context.
    Finds the nearest sentence boundary to the target region midpoint.
    
    Args:
        context: The text context to insert into
        num_regions: Number of regions to divide the context into (default: 5)
    
    Returns:
        List of character positions for insertions (at sentence boundaries)
    """
    context_length = len(context)
    boundaries = find_sentence_boundaries(context)
    
    if not boundaries:
        # Fallback: if no sentence boundaries found, use simple positions
        return [int((i + 0.5) * context_length / num_regions) for i in range(num_regions)]
    
    positions = []
    
    for i in range(num_regions):
        # Calculate the target position (midpoint of region)
        target_position = int((i + 0.5) * context_length / num_regions)
        
        # Find the nearest sentence boundary to the target position
        nearest_boundary = min(boundaries, key=lambda x: abs(x - target_position))
        positions.append(nearest_boundary)
    
    return positions


def insert_secrets(context: str, secrets: List[Tuple[str, str]], positions: List[int]) -> str:
    """
    Insert secret key-value pairs into the context at specified positions.
    Secrets are inserted between sentences to avoid breaking existing text.
    
    Args:
        context: Original context text
        secrets: List of (key, value) tuples to insert
        positions: Character positions where secrets should be inserted (should be at sentence boundaries)
    
    Returns:
        Modified context with secrets inserted
    """
    # Sort positions in descending order to insert from end to start
    # This prevents position shifts from affecting later insertions
    sorted_insertions = sorted(zip(positions, secrets), key=lambda x: x[0], reverse=True)
    
    modified_context = context
    for position, (key, value) in sorted_insertions:
        # Insert with spacing to separate from surrounding sentences
        secret_text = f"The secret {key} is {value}. "
        modified_context = modified_context[:position] + secret_text + modified_context[position:]
    
    return modified_context


def generate_niah_instance(
    instance: Dict[str, Any],
    multiplier: int = 1,
    num_secrets: int = 5,
    seed: int = None
) -> Dict[str, Any]:
    """
    Generate a NIAH instance from an original instance.
    
    Args:
        instance: Original data instance
        multiplier: Number of questions to generate per instance
        num_secrets: Number of secrets to insert (default: 5)
        seed: Random seed for reproducibility
    
    Returns:
        Modified instance with NIAH task
    """
    if seed is not None:
        random.seed(seed)
    
    context = instance.get('context', '')
    if not context:
        raise ValueError(f"Instance {instance.get('_id', 'unknown')} has no context")
    
    # Check multiplier constraint
    if multiplier > num_secrets:
        raise ValueError(
            f"Multiplier ({multiplier}) cannot exceed num_secrets ({num_secrets}). "
            f"Cannot generate {multiplier} unique questions from only {num_secrets} secrets."
        )
    
    # Use random.sample to select unique categories without replacement
    all_categories = list(KEY_VALUE_CATEGORIES.keys())
    categories = random.sample(all_categories, min(num_secrets, len(all_categories)))
    
    # Generate secrets by selecting a random value for each sampled category
    secrets = []
    for category in categories:
        value = random.choice(KEY_VALUE_CATEGORIES[category])
        secrets.append((category, value))
    
    # Calculate insertion positions
    positions = calculate_insertion_positions(context, len(secrets))
    
    # Calculate depth percentages for each secret based on insertion position
    context_length = len(context)
    secret_depths = []
    for pos in positions:
        depth_percent = (pos / context_length) * 100 if context_length > 0 else 0
        
        # Categorize into depth bins
        if depth_percent < 20:
            depth_bin = "0-20%"
        elif depth_percent < 40:
            depth_bin = "20-40%"
        elif depth_percent < 60:
            depth_bin = "40-60%"
        elif depth_percent < 80:
            depth_bin = "60-80%"
        else:
            depth_bin = "80-100%"
        
        secret_depths.append(depth_bin)
    
    # Insert secrets into context
    modified_context = insert_secrets(context, secrets, positions)
    
    # Generate questions and answers with corresponding depths
    questions = []
    answers = []
    question_depths = []
    
    # Sample secrets without replacement for questions
    sampled_secrets = random.sample(secrets, multiplier)
    for key, value in sampled_secrets:
        question = f"What is the secret {key} in the above context?"
        questions.append(question)
        answers.append(value)
        
        # Find the depth for this secret
        secret_idx = next(i for i, (k, v) in enumerate(secrets) if k == key and v == value)
        question_depths.append(secret_depths[secret_idx])
    
    # Create new instance
    niah_instance = {
        '_id': instance['_id'],
        'domain': 'Needle in a Haystack',
        'sub_domain': 'Secret Retrieval',
        'difficulty': instance.get('difficulty', 'medium'),
        'length': instance.get('length', 'long'),
        'questions': questions,  # Always a list
        'answers': answers,      # Always a list
        'question_depths': question_depths,  # Depth bin for each question
        'context': modified_context,
        'secrets': {key: value for key, value in secrets},  # Store all secrets for reference
        'original_question': instance.get('question', ''),
        'original_answer': instance.get('answer', '')
    }
    
    return niah_instance


def main():
    parser = argparse.ArgumentParser(
        description='Generate Needle in a Haystack (NIAH) synthetic dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/longbenchv2/data.json',
        help='Path to input ground truth JSON file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/niah/data.json',
        help='Path to output NIAH dataset JSON file'
    )
    
    parser.add_argument(
        '--blacklist',
        type=str,
        default=None,
        help='Path to blacklist file containing instance IDs to exclude'
    )
    
    parser.add_argument(
        '--multiplier',
        type=int,
        default=1,
        help='Number of questions to generate per instance'
    )
    
    parser.add_argument(
        '--num-secrets',
        type=int,
        default=5,
        help='Number of secret key-value pairs to insert per instance'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Validate multiplier against num_secrets
    if args.multiplier > args.num_secrets:
        print(f"\n⚠️  WARNING: multiplier ({args.multiplier}) is greater than num_secrets ({args.num_secrets})")
        print(f"    This will result in duplicate questions for the same secrets.")
        print(f"    Consider using --multiplier <= {args.num_secrets} to avoid duplicates.\n")
    
    # Set random seed
    random.seed(args.seed)
    
    # Load input data
    print(f"Loading input data from {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} instances")
    
    # Load blacklist
    blacklist = load_blacklist(args.blacklist)
    if blacklist:
        print(f"Loaded blacklist with {len(blacklist)} instances to exclude")
    
    # Filter out blacklisted instances
    filtered_data = [inst for inst in data if inst.get('_id') not in blacklist]
    print(f"After filtering: {len(filtered_data)} instances remaining")
    
    # Generate NIAH instances
    print(f"Generating NIAH instances with multiplier={args.multiplier}...")
    niah_data = []
    
    for i, instance in enumerate(filtered_data):
        try:
            # Use instance index + seed for reproducibility
            instance_seed = args.seed + i if args.seed is not None else None
            niah_instance = generate_niah_instance(
                instance,
                multiplier=args.multiplier,
                num_secrets=args.num_secrets,
                seed=instance_seed
            )
            niah_data.append(niah_instance)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(filtered_data)} instances")
        except Exception as e:
            print(f"Error processing instance {instance.get('_id', 'unknown')}: {e}")
            continue
    
    print(f"Generated {len(niah_data)} NIAH instances")
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save output
    print(f"Saving output to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(niah_data, f, indent=2)
    
    print("Done!")
    print(f"\nSummary:")
    print(f"  Input instances: {len(data)}")
    print(f"  Blacklisted: {len(blacklist)}")
    print(f"  Filtered instances: {len(filtered_data)}")
    print(f"  Output instances: {len(niah_data)}")
    print(f"  Questions per instance: {args.multiplier}")
    print(f"  Total questions: {len(niah_data) * args.multiplier}")


if __name__ == '__main__':
    main()
