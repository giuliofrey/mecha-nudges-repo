#!/usr/bin/env python3
"""
Balance labeled datasets by undersampling the majority class.

This script reads labeled JSONL files and creates balanced versions where
both classes have equal representation. Useful for fair PVI comparisons.

Usage:
    python balance_labels.py --input out/y_labels/before_title_description_small.json \
                            --output out/y_labels/before_title_description_small_balanced.json \
                            --label-col y_title_item_details
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def balance_dataset(
    input_path: str,
    output_path: str,
    label_col: str,
    seed: int = 42
) -> str:
    """Balance a dataset by undersampling the majority class.

    Args:
        input_path: Path to input JSONL file with labels
        output_path: Path to output balanced JSONL file
        label_col: Name of the label column
        seed: Random seed for reproducibility

    Returns:
        Path to the balanced output file
    """
    print(f"Loading dataset from: {input_path}")
    df = pd.read_json(input_path, lines=True)
    
    # Check label column exists
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset. Available: {df.columns.tolist()}")
    
    # Get label counts
    label_counts = df[label_col].value_counts()
    print(f"\nOriginal label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Find minority class size
    min_count = label_counts.min()
    print(f"\nBalancing to {min_count:,} samples per class...")
    
    # Undersample majority class(es) to match minority
    np.random.seed(seed)
    balanced_dfs = []
    
    for label in label_counts.index:
        label_df = df[df[label_col] == label]
        if len(label_df) > min_count:
            # Undersample
            label_df = label_df.sample(n=min_count, random_state=seed)
        balanced_dfs.append(label_df)
    
    # Combine and shuffle
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Report final distribution
    final_counts = balanced_df[label_col].value_counts()
    print(f"\nBalanced label distribution:")
    for label, count in final_counts.items():
        print(f"  {label}: {count:,} ({count/len(balanced_df)*100:.1f}%)")
    
    # Report data loss
    samples_lost = len(df) - len(balanced_df)
    pct_lost = (samples_lost / len(df)) * 100
    print(f"\nTotal samples: {len(df):,} -> {len(balanced_df):,}")
    print(f"Samples removed: {samples_lost:,} ({pct_lost:.1f}% of original data)")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_json(output_path, orient='records', lines=True)
    print(f"Saved balanced dataset to: {output_path}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Balance labeled datasets by undersampling the majority class.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Balance YES/NO dataset
  python balance_labels.py \\
      --input out/y_labels/before_title_description_small.json \\
      --output out/y_labels/before_title_description_small_balanced.json \\
      --label-col y_title_item_details
  
  # Balance SELECT/PASS dataset
  python balance_labels.py \\
      --input out/y_labels/before_title_description_small_pass_select.json \\
      --output out/y_labels/before_title_description_small_pass_select_balanced.json \\
      --label-col y_title_item_details
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input JSONL file with labels'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output balanced JSONL file'
    )
    parser.add_argument(
        '--label-col',
        type=str,
        required=True,
        help='Name of the label column to balance'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    balance_dataset(
        input_path=args.input,
        output_path=args.output,
        label_col=args.label_col,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
