#!/usr/bin/env python3
"""
Step 5: Filter Classification Results

This script filters the GPT classification results to keep only
high-confidence predictions (score >= threshold) and generates visualizations.

Usage:
    python scripts/05_filter_results.py --min-score 8
    python scripts/05_filter_results.py --min-score 7 --output data/output/training_data.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
    parser.add_argument("--include-rationale", action="store_true",
                        help="Include rationale column in output")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("filter_results")

    # Create output directories
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load all batch CSVs
    input_dir = Path(args.input_dir)
    batch_files = sorted(input_dir.glob("batch_*.csv"))

    if not batch_files:
        logger.error(f"No batch files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Loading {len(batch_files)} batch files from {input_dir}")
    dfs = [pd.read_csv(f) for f in batch_files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total records: {len(df):,}")

    # Summary statistics before filtering
    logger.info("\n" + "="*60)
    logger.info("Classification Summary (Before Filtering)")
    logger.info("="*60)
    logger.info(f"Total records: {len(df):,}")
    logger.info(f"Matches (match=True): {df['match'].sum():,} ({df['match'].mean()*100:.1f}%)")
    logger.info(f"Score distribution:")
    logger.info(f"  Mean: {df['score'].mean():.2f}")
    logger.info(f"  Median: {df['score'].median():.0f}")
    logger.info(f"  Min: {df['score'].min():.0f}")
    logger.info(f"  Max: {df['score'].max():.0f}")

    # Score distribution
    score_counts = df['score'].value_counts().sort_index()
    logger.info("\nScore distribution:")
    for score, count in score_counts.items():
        pct = count / len(df) * 100
        logger.info(f"  Score {score}: {count:,} ({pct:.1f}%)")

    # Filter by minimum score
    logger.info(f"\nFiltering with min_score >= {args.min_score}")
    df_filtered = df[df['score'] >= args.min_score].copy()
    logger.info(f"Records after filtering: {len(df_filtered):,} ({len(df_filtered)/len(df)*100:.1f}%)")

    # Remove duplicates (keep highest score for each repo-NAICS pair)
    if 'repo_url' in df_filtered.columns:
        initial_count = len(df_filtered)
        df_filtered = df_filtered.sort_values('score', ascending=False)
        df_filtered = df_filtered.drop_duplicates(subset=['repo_url', 'naics_code'], keep='first')
        logger.info(f"After removing duplicates: {len(df_filtered):,} (removed {initial_count - len(df_filtered):,})")

    # Summary by NAICS code
    logger.info("\n" + "="*60)
    logger.info("Distribution by NAICS Code (Filtered)")
    logger.info("="*60)
    naics_counts = df_filtered['naics_code'].value_counts()
    for code, count in naics_counts.head(15).items():
        logger.info(f"  {code}: {count:,}")
    if len(naics_counts) > 15:
        logger.info(f"  ... and {len(naics_counts) - 15} more NAICS codes")

    # Generate visualizations
    logger.info("\nGenerating visualizations...")

    # 1. Score distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    df['score'].hist(bins=10, edgecolor='black', ax=ax)
    ax.axvline(x=args.min_score, color='red', linestyle='--', label=f'Threshold ({args.min_score})')
    ax.set_xlabel('Classification Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Classification Scores', fontsize=14)
    ax.legend()
    plt.tight_layout()
    hist_path = figures_dir / 'score_distribution.png'
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {hist_path}")
    plt.close()

    # 2. NAICS code distribution (filtered)
    fig, ax = plt.subplots(figsize=(14, 8))
    top_naics = naics_counts.head(20)
    bars = ax.bar(range(len(top_naics)), top_naics.values, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(top_naics)))
    ax.set_xticklabels(top_naics.index, rotation=45, ha='right')
    ax.set_xlabel('NAICS Code', fontsize=12)
    ax.set_ylabel('Number of Repositories', fontsize=12)
    ax.set_title(f'Repository Count by NAICS Code (Score >= {args.min_score})', fontsize=14)

    # Add count labels on bars
    for bar, count in zip(bars, top_naics.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    naics_path = figures_dir / 'naics_distribution.png'
    plt.savefig(naics_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {naics_path}")
    plt.close()

    # 3. Match rate by NAICS (before filtering)
    if len(df) > 0:
        match_rates = df.groupby('naics_code')['match'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(14, 8))
        top_match = match_rates.head(20)
        ax.bar(range(len(top_match)), top_match.values * 100, color='green', edgecolor='black')
        ax.set_xticks(range(len(top_match)))
        ax.set_xticklabels(top_match.index, rotation=45, ha='right')
        ax.set_xlabel('NAICS Code', fontsize=12)
        ax.set_ylabel('Match Rate (%)', fontsize=12)
        ax.set_title('Match Rate by NAICS Code (All Scores)', fontsize=14)
        plt.tight_layout()
        match_path = figures_dir / 'match_rate_by_naics.png'
        plt.savefig(match_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {match_path}")
        plt.close()

    # Save filtered data
    output_cols = ['repo_url', 'naics_code', 'match', 'score']
    if args.include_rationale and 'rationale' in df_filtered.columns:
        output_cols.append('rationale')

    # Keep only available columns
    output_cols = [c for c in output_cols if c in df_filtered.columns]

    logger.info(f"\nSaving filtered data to {args.output}")
    df_filtered[output_cols].to_csv(args.output, index=False)

    # Also save as parquet
    parquet_path = args.output.replace('.csv', '.parquet')
    df_filtered[output_cols].to_parquet(parquet_path, index=False)
    logger.info(f"Also saved to {parquet_path}")

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"Input records: {len(df):,}")
    logger.info(f"Filtered records (score >= {args.min_score}): {len(df_filtered):,}")
    logger.info(f"Unique NAICS codes: {df_filtered['naics_code'].nunique()}")
    if 'repo_url' in df_filtered.columns:
        logger.info(f"Unique repositories: {df_filtered['repo_url'].nunique()}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
