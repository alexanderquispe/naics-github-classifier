#!/usr/bin/env python3
"""
Step 4: Classify Repositories with GPT

This script uses GPT-4 via the GitHub Copilot API to classify repositories
against their assigned NAICS sectors.

Usage:
    python scripts/04_classify_repos.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import load_repos
from classification.gpt_classifier import GPTClassifier
from classification.batch_processor import BatchProcessor
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Classify repositories with GPT")
    parser.add_argument("--input", type=str,
                        default="data/processed/retrieved_repos_by_naics.parquet",
                        help="Input file with retrieved repos")
    parser.add_argument("--output-dir", type=str,
                        default="data/output/batch_csvs",
                        help="Output directory for batch CSVs")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Batch size for saving results")
    parser.add_argument("--max-tokens", type=int, default=3000,
                        help="Max tokens for README content")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("classify_repos")

    logger.info(f"Loading retrieved repos from {args.input}")
    # TODO: Load repos

    logger.info("Initializing GPT classifier...")
    # TODO: Initialize classifier

    logger.info("Starting classification...")
    # TODO: Run batch classification

    logger.info("Done!")


if __name__ == "__main__":
    main()
