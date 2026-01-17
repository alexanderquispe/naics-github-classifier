#!/usr/bin/env python3
"""
Step 1: Generate BGE Embeddings for Repository Dataset

This script loads the repository dataset and generates semantic embeddings
using the BAAI/bge-large-en model.

Usage:
    python scripts/01_generate_embeddings.py --input data/raw/repos_eu_531k.parquet
    python scripts/01_generate_embeddings.py --input data/raw/repos_eu_531k.parquet --batch-size 512
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import load_repos
from data.preprocessor import clean_readme, truncate_text, create_input_text
from embeddings.bge_encoder import BGEEncoder
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BGE embeddings for repositories")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output parquet file path (default: data/processed/embeddings/repos_with_embeddings.parquet)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for embedding")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--max-readme-chars", type=int, default=1000, help="Max README characters")
    parser.add_argument("--sample", type=int, default=None, help="Sample N rows for testing")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("generate_embeddings")

    # Set default output path
    if args.output is None:
        output_dir = Path("data/processed/embeddings")
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / "repos_with_embeddings.parquet")

    # Load data
    logger.info(f"Loading repositories from {args.input}")
    df = load_repos(args.input)
    logger.info(f"Loaded {len(df):,} repositories")

    # Sample if requested
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        logger.info(f"Sampled {len(df):,} repositories for testing")

    # Drop rows with empty README
    readme_col = None
    for col in ['readme_content', 'readme', 'README']:
        if col in df.columns:
            readme_col = col
            break

    if readme_col:
        initial_count = len(df)
        df = df[df[readme_col].notna() & (df[readme_col] != '')]
        logger.info(f"Dropped {initial_count - len(df):,} rows with empty README")

    # Create input text
    logger.info("Creating input text from description, topics, and README...")

    desc_col = 'description' if 'description' in df.columns else None
    topics_col = 'topics' if 'topics' in df.columns else None

    input_texts = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        desc = row.get(desc_col, '') if desc_col else ''
        topics = row.get(topics_col, '') if topics_col else ''
        readme = row.get(readme_col, '') if readme_col else ''

        input_text = create_input_text(desc, topics, readme, args.max_readme_chars)
        input_texts.append(input_text)

    df = df.copy()
    df['input_text'] = input_texts

    # Drop rows with empty input text
    df = df[df['input_text'].str.len() > 0]
    logger.info(f"Final dataset: {len(df):,} repositories")

    # Initialize encoder
    logger.info("Initializing BGE encoder...")
    encoder = BGEEncoder(use_half_precision=True)

    # Generate embeddings
    logger.info(f"Generating embeddings with batch size {args.batch_size}...")
    embeddings = encoder.encode(
        df['input_text'].tolist(),
        batch_size=args.batch_size,
        show_progress=True,
        normalize=True
    )

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Add embeddings to dataframe
    df['embedding'] = list(embeddings)

    # Save
    logger.info(f"Saving to {args.output}")
    df.to_parquet(args.output, index=False)

    # Also save metadata separately (without embeddings) for FAISS
    metadata_cols = [c for c in df.columns if c not in ['embedding', 'input_text', readme_col]]
    if 'nwo' in df.columns or 'repo_url' in df.columns:
        metadata_path = args.output.replace('.parquet', '_metadata.parquet')
        df[metadata_cols + ['input_text']].to_parquet(metadata_path, index=False)
        logger.info(f"Metadata saved to {metadata_path}")

    # Save embeddings as numpy array for FAISS
    embeddings_path = args.output.replace('.parquet', '_vectors.npy')
    np.save(embeddings_path, embeddings)
    logger.info(f"Embeddings array saved to {embeddings_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
