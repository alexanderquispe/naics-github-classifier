#!/usr/bin/env python3
"""
Step 2: Build FAISS Index for Semantic Search

This script creates a FAISS index from the pre-computed embeddings
for efficient similarity search.

Usage:
    python scripts/02_build_faiss_index.py
    python scripts/02_build_faiss_index.py --embeddings data/processed/embeddings/repos_with_embeddings.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings.faiss_indexer import FAISSIndexer
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings")
    parser.add_argument("--embeddings", type=str,
                        default="data/processed/embeddings/repos_with_embeddings.parquet",
                        help="Input embeddings parquet file")
    parser.add_argument("--vectors", type=str, default=None,
                        help="Input numpy vectors file (optional, will extract from parquet if not provided)")
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

    # Create output directories
    Path(args.output_index).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_metadata).parent.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    if args.vectors and Path(args.vectors).exists():
        logger.info(f"Loading embeddings from numpy file: {args.vectors}")
        embeddings = np.load(args.vectors)

        logger.info(f"Loading metadata from: {args.embeddings}")
        df = pd.read_parquet(args.embeddings)
        if 'embedding' in df.columns:
            df = df.drop(columns=['embedding'])
    else:
        logger.info(f"Loading embeddings from parquet: {args.embeddings}")
        df = pd.read_parquet(args.embeddings)

        if 'embedding' not in df.columns:
            # Try to load from numpy file
            vectors_path = args.embeddings.replace('.parquet', '_vectors.npy')
            if Path(vectors_path).exists():
                logger.info(f"Loading vectors from: {vectors_path}")
                embeddings = np.load(vectors_path)
            else:
                raise ValueError("No embeddings found in parquet file and no vectors file available")
        else:
            logger.info("Extracting embeddings from parquet...")
            embeddings = np.vstack(df['embedding'].values)
            df = df.drop(columns=['embedding'])

    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Metadata rows: {len(df):,}")

    # Ensure embeddings are float32
    embeddings = embeddings.astype(np.float32)

    # Build FAISS index
    logger.info(f"Building FAISS index (GPU: {args.use_gpu})...")
    indexer = FAISSIndexer(use_gpu=args.use_gpu)
    indexer.build_index(embeddings, normalize=True)
    indexer.set_metadata(df)

    logger.info(f"Index contains {indexer.index.ntotal:,} vectors")

    # Save index and metadata
    logger.info(f"Saving index to: {args.output_index}")
    logger.info(f"Saving metadata to: {args.output_metadata}")
    indexer.save(args.output_index, args.output_metadata)

    # Verify by loading
    logger.info("Verifying saved index...")
    test_indexer = FAISSIndexer(use_gpu=False)
    test_indexer.load(args.output_index, args.output_metadata)
    logger.info(f"Verified: Index has {test_indexer.index.ntotal:,} vectors")

    logger.info("Done!")


if __name__ == "__main__":
    main()
