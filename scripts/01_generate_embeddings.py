#!/usr/bin/env python3
"""
Step 1: Generate BGE Embeddings for Repository Dataset

This script loads the repository dataset and generates semantic embeddings
using the BAAI/bge-large-en model.

Usage:
    python scripts/01_generate_embeddings.py --input data/raw/repos_eu_531k.parquet
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import load_repos
from data.preprocessor import clean_readme, create_input_text
from embeddings.bge_encoder import BGEEncoder
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BGE embeddings for repositories")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file path")
    parser.add_argument("--output", type=str, default="data/processed/embeddings/repos_with_embeddings.parquet",
                        help="Output parquet file path")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for embedding")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("generate_embeddings")

    logger.info(f"Loading repositories from {args.input}")
    # TODO: Implement full pipeline

    logger.info("Generating embeddings...")
    # TODO: Implement embedding generation

    logger.info(f"Saving to {args.output}")
    # TODO: Save embeddings

    logger.info("Done!")


if __name__ == "__main__":
    main()
