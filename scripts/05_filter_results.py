#!/usr/bin/env python3
"""
Step 5: Filter Classification Results

This script filters the GPT classification results to keep only
high-confidence predictions (score >= threshold).

Usage:
    python scripts/05_filter_results.py --min-score 8
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Filter classification results")
    parser.add_argument("--input-dir", type=str,
                        default="data/output/batch_csvs",
                        help="Directory containing batch CSV files")
    parser.add_argument("--output", type=str,
                        default="data/output/naics_training_data_filtered.csv",
                        help="Output filtered CSV file")
    parser.add_argument("--min-score", type=int, default=8,
                        help="Minimum classification score to keep")
    parser.add_argument("--figures-dir", type=str,
                        default="figures",
                        help="Directory to save figures")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("filter_results")

    logger.info(f"Loading batch CSVs from {args.input_dir}")
    # TODO: Load and concatenate batch CSVs

    logger.info(f"Filtering results with score >= {args.min_score}")
    # TODO: Filter results

    logger.info("Generating distribution figures...")
    # TODO: Create visualizations

    logger.info(f"Saving filtered data to {args.output}")
    # TODO: Save filtered results

    logger.info("Done!")


if __name__ == "__main__":
    main()
