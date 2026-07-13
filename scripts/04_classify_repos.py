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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    parser.add_argument("--backend", type=str, choices=["openai", "github", "deepseek"],
                        default=None,
                        help="API backend: 'openai' (OpenAI API) or 'github' (GitHub Copilot). Auto-detects if not specified.")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name to use (default: gpt-4-turbo for OpenAI, gpt-4.1 for GitHub)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel API workers (default: 1)")
    parser.add_argument("--source-pool", type=str,
                        default="data/processed/source_pool_merged.parquet",
                        help="Source pool parquet for joining readme_content. "
                             "Pass empty string to skip the join (will fall back "
                             "to input_text from the retrieved parquet, which is "
                             "truncated to 1000 chars at embedding time).")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("classify_repos")

    # Check for API token (either OpenAI or GitHub)
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("DEEPSEEK_API_KEY") and not os.getenv("GITHUB_TOKEN"):
        logger.error("No API key found. Set OPENAI_API_KEY, DEEPSEEK_API_KEY, or GITHUB_TOKEN in .env file.")
        sys.exit(1)

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    # Load retrieved repos
    logger.info(f"Loading retrieved repos from {args.input}")
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} repo-NAICS pairs")

    # Join with the source pool to attach the FULL readme_content. The
    # retrieved parquet only carries the embedding-time `input_text` field,
    # whose README was truncated to 1000 chars to fit BGE's seq_len budget.
    # The notebook (cell 49) re-joined readme_content from the source frame at
    # GPT classification time so the model sees the full README (~3000 tokens
    # after preprocess_readme); we replicate that here.
    if args.source_pool and Path(args.source_pool).exists():
        logger.info(f"Joining with source pool from {args.source_pool}")
        source = pd.read_parquet(args.source_pool, columns=['nwo', 'readme_content'])
        before = len(df)
        df = df.merge(source, on='nwo', how='left')
        missing = df['readme_content'].isna().sum()
        logger.info(
            f"Joined: {len(df):,} rows ({missing:,} have no matching "
            f"readme_content in source pool)"
        )
        assert len(df) == before, "Join should be many-to-one and not change row count"
    else:
        logger.warning(
            f"Source pool not found or disabled. Falling back to input_text "
            f"(truncated to 1000 chars at embedding time)."
        )

    # Apply limit if specified
    if args.limit:
        df = df.iloc[:args.limit]
        logger.info(f"Limited to {len(df):,} repos for testing")

    # Find README column — prefer readme_content (just joined from source);
    # fall back to input_text if the source-pool join was skipped/failed.
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
    results_lock = threading.Lock()

    # Open log file
    log_file = open(args.log_file, 'a', encoding='utf-8')
    log_lock = threading.Lock()

    def classify_row(idx):
        row = df.iloc[idx]

        # Repo identifier and NAICS target
        nwo = row.get('nwo', '')
        repo_url = row.get('repo_url', f"https://github.com/{nwo}" if nwo else f"repo_{idx}")
        naics_code = str(row.get('naics_code', ''))
        naics_definition = str(row.get('naics_definition', ''))

        # Pull the raw fields the notebook combined in cell 50:
        # `description: {d}, topics: {t}, readme: {r}`. We use readme_content
        # if the source-pool join succeeded; otherwise fall back to whatever
        # column the readme_col search picked.
        description = row.get('description', '')
        topics = row.get('topics', '')
        if readme_col == 'readme_content':
            raw_readme = row.get('readme_content', '')
        else:
            # Fallback path: input_text is already a combined string, treat it
            # as the readme body to avoid double-prefixing description/topics.
            raw_readme = row.get(readme_col, '')

        # Skip only if the readme is genuinely empty (notebook cell 70:
        # `if not processed_content: empty_count += 1; continue`).
        if not raw_readme or not isinstance(raw_readme, str) or not raw_readme.strip():
            return {
                'repo_url': repo_url,
                'naics_code': naics_code,
                'match': '',
                'score': '',
                'rationale': 'README empty'
            }

        # Normalise topics to a comma-joined string (matches notebook cell 50
        # which used `', '.join(x) if isinstance(x, list) else str(x)`).
        if isinstance(topics, list):
            topics_str = ', '.join(str(t) for t in topics if t)
        else:
            topics_str = str(topics) if topics else ''
        description_str = description if isinstance(description, str) else ''

        # Build combined input text exactly once — only when readme_col is the
        # raw readme_content. If we're falling back to input_text, it already
        # has description + topics baked in, so skip the prefix to avoid the
        # double-concatenation bug the previous version had.
        if readme_col == 'readme_content':
            combined = (
                f"description: {description_str}, "
                f"topics: {topics_str}, "
                f"readme: {raw_readme}"
            )
        else:
            combined = str(raw_readme)

        # Clean and truncate (notebook preprocess_readme in cell 67).
        cleaned = clean_readme(combined)
        truncated = truncate_to_tokens(cleaned, args.max_tokens)

        if not truncated.strip():
            return {
                'repo_url': repo_url,
                'naics_code': naics_code,
                'match': '',
                'score': '',
                'rationale': 'README empty after cleaning'
            }

        # Classify
        success, response = classifier.classify(
            readme_content=truncated,
            naics_code=naics_code,
            naics_definition=naics_definition
        )

        return {
            'repo_url': repo_url,
            'naics_code': naics_code,
            'match': response.get('match', '') if success else '',
            'score': response.get('score', '') if success else '',
            'rationale': response.get('rationale', response.get('error', 'Unknown error'))
        }

    num_workers = args.workers
    logger.info(f"Using {num_workers} parallel workers")

    try:
        pbar = tqdm(total=len(df) - args.start_idx, desc="Classifying repos")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(classify_row, idx): idx
                for idx in range(args.start_idx, len(df))
            }
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)

                with log_lock:
                    log_file.write(f"\n{'='*60}\n")
                    log_file.write(f"Repo: {result['repo_url']}\n")
                    log_file.write(f"NAICS: {result['naics_code']}\n")
                    log_file.write(f"Match: {result['match']}, Score: {result['score']}\n")
                    log_file.write(f"Rationale: {result['rationale']}\n")
                    log_file.flush()

                with results_lock:
                    results.append(result)
                    if len(results) >= args.batch_size:
                        batch_df = pd.DataFrame(results)
                        batch_path = output_dir / f"batch_{batch_num:04d}.csv"
                        batch_df.to_csv(batch_path, index=False)
                        logger.info(f"Saved batch {batch_num} ({len(results)} results) to {batch_path}")
                        results = []
                        batch_num += 1

        pbar.close()

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
