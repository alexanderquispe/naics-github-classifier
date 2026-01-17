#!/usr/bin/env python3
"""
Step 2: Build FAISS Index for Semantic Search

This script creates a FAISS index from the pre-computed embeddings
for efficient similarity search.

Usage:
    python scripts/02_build_faiss_index.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings.faiss_indexer import FAISSIndexer
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings")
    parser.add_argument("--embeddings", type=str,
                        default="data/processed/embeddings/repos_with_embeddings.parquet",
                        help="Input embeddings parquet file")
    parser.add_argument("--output-index", type=str,
                        default="data/processed/faiss_indices/faiss_index.index",
                        help="Output FAISS index file")
    parser.add_argument("--output-metadata", type=str,
                        default="data/processed/faiss_indices/repo_metadata.parquet",
                        help="Output metadata parquet file")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for index creation")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("build_faiss_index")

    logger.info(f"Loading embeddings from {args.embeddings}")
    # TODO: Load embeddings

    logger.info("Building FAISS index...")
    # TODO: Build index

    logger.info(f"Saving index to {args.output_index}")
    # TODO: Save index and metadata

    logger.info("Done!")


if __name__ == "__main__":
    main()
