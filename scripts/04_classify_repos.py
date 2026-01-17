#!/usr/bin/env python3
"""
Step 4: Classify Repositories with GPT

This script uses GPT-4 via the GitHub Copilot API to classify repositories
against their assigned NAICS sectors.

Usage:
    python scripts/04_classify_repos.py
    python scripts/04_classify_repos.py --batch-size 500 --start-idx 0
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import load_repos
from data.preprocessor import clean_readme, truncate_to_tokens
from classification.gpt_classifier import GPTClassifier
from classification.prompt_builder import build_classification_prompt
from utils.logger import setup_logger

load_dotenv()


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
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Starting index (for resuming)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of repos to classify (for testing)")
    parser.add_argument("--log-file", type=str,
                        default="data/output/classification_log.txt",
                        help="Log file for classification results")
    parser.add_argument("--backend", type=str, choices=["openai", "github"],
                        default=None,
                        help="API backend: 'openai' (OpenAI API) or 'github' (GitHub Copilot). Auto-detects if not specified.")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name to use (default: gpt-4-turbo for OpenAI, gpt-4.1 for GitHub)")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("classify_repos")

    # Check for API token (either OpenAI or GitHub)
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GITHUB_TOKEN"):
        logger.error("No API key found. Set OPENAI_API_KEY or GITHUB_TOKEN in .env file.")
        sys.exit(1)

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    # Load retrieved repos
    logger.info(f"Loading retrieved repos from {args.input}")
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} repo-NAICS pairs")

    # Apply limit if specified
    if args.limit:
        df = df.iloc[:args.limit]
        logger.info(f"Limited to {len(df):,} repos for testing")

    # Find README column
    readme_col = None
    for col in ['readme_content', 'readme', 'README', 'input_text']:
        if col in df.columns:
            readme_col = col
            break

    if readme_col is None:
        logger.error("No README column found in input data")
        sys.exit(1)

    logger.info(f"Using '{readme_col}' as README column")

    # Initialize classifier
    logger.info("Initializing GPT classifier...")
    classifier = GPTClassifier(backend=args.backend, model=args.model)

    # Process repos
    results = []
    batch_num = args.start_idx // args.batch_size

    # Open log file
    log_file = open(args.log_file, 'a', encoding='utf-8')

    try:
        for idx in tqdm(range(args.start_idx, len(df)), desc="Classifying repos"):
            row = df.iloc[idx]

            # Get README content
            readme = row.get(readme_col, '')
            if readme:
                readme = clean_readme(str(readme))
                readme = truncate_to_tokens(readme, args.max_tokens)

            # Get repo identifier
            repo_url = row.get('repo_url', row.get('nwo', f"repo_{idx}"))
            naics_code = str(row.get('naics_code', ''))
            naics_definition = str(row.get('naics_definition', ''))

            # Skip if no README
            if not readme or len(readme.strip()) < 50:
                result = {
                    'repo_url': repo_url,
                    'naics_code': naics_code,
                    'match': False,
                    'score': 0,
                    'rationale': 'README too short or empty'
                }
                results.append(result)
                continue

            # Classify
            success, response = classifier.classify(
                readme_content=readme,
                naics_code=naics_code,
                naics_definition=naics_definition
            )

            result = {
                'repo_url': repo_url,
                'naics_code': naics_code,
                'match': response.get('match', False) if success else False,
                'score': response.get('score', 0) if success else 0,
                'rationale': response.get('rationale', response.get('error', 'Unknown error'))
            }
            results.append(result)

            # Log result
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"Repo: {repo_url}\n")
            log_file.write(f"NAICS: {naics_code}\n")
            log_file.write(f"Match: {result['match']}, Score: {result['score']}\n")
            log_file.write(f"Rationale: {result['rationale']}\n")
            log_file.flush()

            # Save batch
            if len(results) >= args.batch_size:
                batch_df = pd.DataFrame(results)
                batch_path = output_dir / f"batch_{batch_num:04d}.csv"
                batch_df.to_csv(batch_path, index=False)
                logger.info(f"Saved batch {batch_num} ({len(results)} results) to {batch_path}")
                results = []
                batch_num += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        log_file.close()

        # Save remaining results
        if results:
            batch_df = pd.DataFrame(results)
            batch_path = output_dir / f"batch_{batch_num:04d}.csv"
            batch_df.to_csv(batch_path, index=False)
            logger.info(f"Saved final batch {batch_num} ({len(results)} results)")

    # Summary
    logger.info("\nClassification complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Log saved to: {args.log_file}")

    # Load all batches and show summary
    all_files = sorted(output_dir.glob("batch_*.csv"))
    if all_files:
        all_results = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
        logger.info(f"\nTotal classified: {len(all_results):,}")
        logger.info(f"Matches: {all_results['match'].sum():,}")
        logger.info(f"Average score: {all_results['score'].mean():.2f}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
