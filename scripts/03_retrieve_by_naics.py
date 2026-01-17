#!/usr/bin/env python3
"""
Step 3: Retrieve Repositories by NAICS Sector

This script uses semantic search to find repositories relevant to each
NAICS sector based on their subindustry descriptions.

Usage:
    python scripts/03_retrieve_by_naics.py
    python scripts/03_retrieve_by_naics.py --base-k 400 --min-k 20
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import load_naics_definitions
from embeddings.faiss_indexer import FAISSIndexer
from embeddings.bge_encoder import BGEEncoder
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieve repositories by NAICS sector")
    parser.add_argument("--naics-file", type=str,
                        default="data/raw/naics_titles_by_group_6digit_clean.json",
                        help="NAICS definitions JSON file")
    parser.add_argument("--index", type=str,
                        default="data/processed/faiss_indices/faiss_index.index",
                        help="FAISS index file")
    parser.add_argument("--metadata", type=str,
                        default="data/processed/faiss_indices/repo_metadata.parquet",
                        help="Repository metadata file")
    parser.add_argument("--output", type=str,
                        default="data/processed/retrieved_repos_by_naics.parquet",
                        help="Output file with retrieved repos")
    parser.add_argument("--base-k", type=int, default=400,
                        help="Target repos per NAICS group")
    parser.add_argument("--min-k", type=int, default=20,
                        help="Minimum repos per subindustry query")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("retrieve_by_naics")

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load NAICS definitions
    logger.info(f"Loading NAICS definitions from {args.naics_file}")
    naics_data = load_naics_definitions(args.naics_file)
    logger.info(f"Loaded {len(naics_data)} NAICS sectors")

    # Load FAISS index
    logger.info(f"Loading FAISS index from {args.index}")
    indexer = FAISSIndexer(use_gpu=False)
    indexer.load(args.index, args.metadata)
    logger.info(f"Index contains {indexer.index.ntotal:,} vectors")

    # Initialize BGE encoder for queries
    logger.info("Initializing BGE encoder for queries...")
    encoder = BGEEncoder(use_half_precision=False)  # CPU mode for queries

    # Retrieve repositories for each NAICS sector
    all_results = []

    for naics_code, definition in tqdm(naics_data.items(), desc="Processing NAICS sectors"):
        # Split definition into subindustries
        subindustries = [s.strip() for s in definition.split(';') if s.strip()]

        if not subindustries:
            logger.warning(f"No subindustries for NAICS {naics_code}")
            continue

        # Calculate k per subindustry
        k_per_sub = max(args.min_k, math.ceil(args.base_k / len(subindustries)))

        sector_results = []
        seen_repos = set()

        for subindustry in subindustries:
            # Create query
            query = f"{subindustry} software tools applications"

            # Encode query
            query_embedding = encoder.encode_query(query)

            # Search
            results = indexer.search_with_metadata(query_embedding, k=k_per_sub)

            # Add NAICS info and deduplicate
            for _, row in results.iterrows():
                repo_id = row.get('nwo', row.get('repo_url', str(row.name)))

                if repo_id not in seen_repos:
                    seen_repos.add(repo_id)
                    result_dict = row.to_dict()
                    result_dict['naics_code'] = naics_code
                    result_dict['naics_definition'] = definition
                    result_dict['matched_subindustry'] = subindustry
                    sector_results.append(result_dict)

        all_results.extend(sector_results)
        logger.debug(f"NAICS {naics_code}: Retrieved {len(sector_results)} unique repos")

    # Create output dataframe
    logger.info(f"Total retrieved: {len(all_results):,} repo-NAICS pairs")
    results_df = pd.DataFrame(all_results)

    # Remove duplicates (same repo, same NAICS)
    if 'nwo' in results_df.columns:
        results_df = results_df.drop_duplicates(subset=['nwo', 'naics_code'])
    elif 'repo_url' in results_df.columns:
        results_df = results_df.drop_duplicates(subset=['repo_url', 'naics_code'])

    logger.info(f"After deduplication: {len(results_df):,} repo-NAICS pairs")

    # Save results
    logger.info(f"Saving to {args.output}")
    results_df.to_parquet(args.output, index=False)

    # Summary statistics
    logger.info("\nSummary by NAICS sector:")
    sector_counts = results_df['naics_code'].value_counts()
    for code, count in sector_counts.head(10).items():
        logger.info(f"  {code}: {count:,} repos")

    if len(sector_counts) > 10:
        logger.info(f"  ... and {len(sector_counts) - 10} more sectors")

    logger.info("Done!")


if __name__ == "__main__":
    main()
