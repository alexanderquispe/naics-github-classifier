#!/usr/bin/env python3
"""
Step 5: Filter GPT Results into the Final Labeled Corpus

Consumes the per-batch CSVs written by scripts/04_classify_repos.py and
produces the final training-ready corpus. Mirrors notebook cells 78-80
(score >= 8 filter) plus the Step 6 released-dataset filter
(min-samples-per-class) documented in PIPELINE.md.

Pipeline:
  1. Read every batch_*.csv in --gpt-output-dir and concatenate.
  2. Normalise types: score -> numeric, match -> lowercase string. (The
     patched step-4 classifier emits score/match as STRINGS, matching the
     notebook's nested-JSON output, so we cast here.)
  3. Filter to score >= --min-score (default 8). Optionally also require
     match == "yes" (--require-match-yes; off by default to mirror the
     notebook, which filtered on score alone).
  4. Deduplicate on (nwo, naics_code), keeping the highest score.
  5. Join back to the retrieval candidates (--candidates) on (nwo,
     naics_code) to recover description, topics, naics_definition, and —
     for the Tier 1 regional run — the `region` column for cross-country
     analysis.
  6. Optionally join the source pool (--source-pool) on nwo to attach the
     full readme_content.
  7. Optionally drop sectors with fewer than --min-samples-per-class rows
     (Step 6: 80 drops sector 55).
  8. Write the final corpus parquet.

The GPT batch CSVs carry `repo_url` (https://github.com/<nwo>); we derive
`nwo` by stripping the host prefix, then join on (nwo, naics_code).

Cross-country figures (score distribution by region, sector mix by region)
are produced by a dedicated analysis script, not here — this step only
builds the corpus.

Usage (Tier 2, merged/country-blind):
    python scripts/05_filter_results.py \\
        --gpt-output-dir data/output/batch_csvs \\
        --candidates     data/processed/retrieval_candidates.parquet \\
        --source-pool    data/processed/source_pool_merged.parquet \\
        --output         data/processed/train_corpus_merged.parquet

Usage (Tier 1, regional/with-country):
    python scripts/05_filter_results.py \\
        --gpt-output-dir data/output/regional_batch_csvs \\
        --candidates     data/processed/retrieval_candidates_regional.parquet \\
        --source-pool    data/processed/source_pool_merged.parquet \\
        --output         data/processed/train_corpus_regional.parquet \\
        --min-samples-per-class 80
"""

import argparse
import sys
from pathlib import Path

import polars as pl


GITHUB_PREFIX = "https://github.com/"


def parse_args():
    p = argparse.ArgumentParser(description="Filter GPT results into the final corpus")
    p.add_argument("--gpt-output-dir", required=True,
                   help="Directory containing batch_*.csv from step 4")
    p.add_argument("--candidates", required=True,
                   help="Retrieval candidates parquet from step 3 (for metadata + region)")
    p.add_argument("--source-pool", default=None,
                   help="Source parquet to join readme_content on nwo (optional)")
    p.add_argument("--output", required=True,
                   help="Output path for the final corpus parquet")
    p.add_argument("--min-score", type=int, default=8,
                   help="Keep rows with GPT score >= this (default: 8)")
    p.add_argument("--require-match-yes", action="store_true",
                   help="Additionally require match == 'yes' (default: off, "
                        "mirrors the notebook which filtered on score alone)")
    p.add_argument("--min-samples-per-class", type=int, default=0,
                   help="Drop NAICS sectors with fewer than this many rows "
                        "(0 = keep all; 80 replicates the released-dataset filter)")
    return p.parse_args()


def load_batches(gpt_output_dir: Path) -> pl.DataFrame:
    csvs = sorted(gpt_output_dir.glob("batch_*.csv"))
    if not csvs:
        sys.exit(f"No batch_*.csv files found in {gpt_output_dir}")
    # Force every column to Utf8 on read (infer_schema_length=0) so
    # score/match parse consistently across batches — a batch of all-"No"
    # could otherwise infer score as int in one file and str in another,
    # breaking the vertical concat.
    frames = [pl.read_csv(c, infer_schema_length=0) for c in csvs]
    df = pl.concat(frames, how="vertical_relaxed")
    print(f"  Read {len(csvs)} batch files -> {df.height:,} rows")
    return df


def main():
    args = parse_args()

    gpt_dir = Path(args.gpt_output_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load + concat GPT batch CSVs ----------------------------------
    print(f"Loading GPT batches from {gpt_dir}")
    results = load_batches(gpt_dir)

    required_cols = {"repo_url", "naics_code", "match", "score"}
    missing = required_cols - set(results.columns)
    if missing:
        sys.exit(f"Batch CSVs missing required columns: {sorted(missing)}")

    # ---- 2. Normalise types -----------------------------------------------
    results = results.with_columns(
        pl.col("repo_url").str.strip_prefix(GITHUB_PREFIX).alias("nwo"),
        pl.col("score").cast(pl.Float64, strict=False).alias("score_num"),
        pl.col("match").cast(pl.Utf8).str.to_lowercase().str.strip_chars().alias("match_norm"),
        pl.col("naics_code").cast(pl.Utf8).str.strip_chars().alias("naics_code"),
    )

    n_total = results.height
    n_parse_fail = results.filter(pl.col("score_num").is_null()).height
    print(f"  Parsed {n_total:,} results ({n_parse_fail:,} with unparseable score)")

    # Score distribution (useful sanity check before filtering)
    dist = (
        results.filter(pl.col("score_num").is_not_null())
        .group_by("score_num").len().rename({"len": "n"}).sort("score_num")
    )
    print("  Score distribution:")
    for score, n in dist.rows():
        print(f"    {score:g}: {n:,}")

    # ---- 3. Filter --------------------------------------------------------
    kept = results.filter(pl.col("score_num") >= args.min_score)
    print(f"  score >= {args.min_score}: {kept.height:,} rows "
          f"({kept.height / n_total * 100:.1f}% of {n_total:,})")

    if args.require_match_yes:
        before = kept.height
        kept = kept.filter(pl.col("match_norm") == "yes")
        print(f"  match == 'yes': {kept.height:,} rows (dropped {before - kept.height:,})")

    if kept.height == 0:
        sys.exit("No rows passed the filter. Check --min-score / batch contents.")

    # ---- 4. Dedup on (nwo, naics_code), keep highest score ---------------
    before = kept.height
    kept = (
        kept.sort("score_num", descending=True)
        .unique(subset=["nwo", "naics_code"], keep="first")
    )
    if before != kept.height:
        print(f"  Deduplicated (nwo, naics_code): {kept.height:,} rows "
              f"(removed {before - kept.height:,})")

    # ---- 5. Join retrieval candidates for metadata + region --------------
    print(f"Joining candidates from {args.candidates}")
    cand = pl.read_parquet(args.candidates)
    cand = cand.with_columns(pl.col("naics_code").cast(pl.Utf8).str.strip_chars())

    cand_cols = ["nwo", "naics_code"]
    for opt in ["description", "topics", "naics_definition",
                "matched_subindustry", "region", "country", "country_code",
                "similarity_score"]:
        if opt in cand.columns and opt not in cand_cols:
            cand_cols.append(opt)
    cand = cand.select(cand_cols).unique(subset=["nwo", "naics_code"])

    corpus = kept.join(cand, on=["nwo", "naics_code"], how="left")
    if "description" in corpus.columns:
        n_unmatched = corpus.filter(pl.col("description").is_null()).height
        if n_unmatched:
            print(f"  WARNING: {n_unmatched:,} kept rows had no candidate match "
                  f"(joined as nulls)")

    # ---- 6. Optional source-pool join for readme_content -----------------
    if args.source_pool and Path(args.source_pool).exists():
        print(f"Joining readme_content from {args.source_pool}")
        src = (
            pl.scan_parquet(args.source_pool)
            .select(["nwo", "readme_content"])
            .unique(subset=["nwo"])
            .collect()
        )
        corpus = corpus.join(src, on="nwo", how="left")
        n_no_readme = corpus.filter(pl.col("readme_content").is_null()).height
        print(f"  Joined readme_content ({n_no_readme:,} rows without a match)")
    elif args.source_pool:
        print(f"  WARNING: --source-pool {args.source_pool} not found; "
              f"skipping readme_content join")

    # ---- 7. Standardise output columns ------------------------------------
    # `code` is the canonical label name used downstream (training corpus,
    # stratification appendix). Keep naics_code too for traceability.
    corpus = corpus.with_columns(pl.col("naics_code").alias("code"))

    # Drop the raw string score/match columns so the normalised numeric
    # `score_num`/`match_norm` can take their canonical names without a
    # duplicate-column collision.
    corpus = corpus.drop(["score", "match"])
    corpus = corpus.rename({"score_num": "score", "match_norm": "match"})

    # ---- 8. Optional min-samples-per-class --------------------------------
    if args.min_samples_per_class > 0:
        counts = corpus.group_by("code").len().rename({"len": "n"})
        keep_codes = counts.filter(pl.col("n") >= args.min_samples_per_class)["code"]
        dropped = counts.filter(pl.col("n") < args.min_samples_per_class)
        before = corpus.height
        corpus = corpus.filter(pl.col("code").is_in(keep_codes))
        if dropped.height:
            print(f"  min-samples-per-class={args.min_samples_per_class}: "
                  f"dropped {before - corpus.height:,} rows across "
                  f"{dropped.height} under-populated sectors: "
                  f"{dropped.sort('n').rows()}")

    # ---- Save -------------------------------------------------------------
    corpus.write_parquet(out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nWrote {out_path} ({size_mb:.1f} MB, {corpus.height:,} rows, "
          f"{corpus.width} cols)")

    # ---- Summary ----------------------------------------------------------
    print("\nRows per NAICS sector:")
    by_code = corpus.group_by("code").len().rename({"len": "n"}).sort("code")
    for code, n in by_code.rows():
        print(f"  {code}: {n:,}")

    if "region" in corpus.columns:
        print("\nRows per region (Tier 1 cross-jurisdiction):")
        by_region = corpus.group_by("region").len().rename({"len": "n"}).sort("region")
        for region, n in by_region.rows():
            print(f"  {region}: {n:,}")

    print("\nFinal schema:")
    for col, dtype in zip(corpus.columns, corpus.dtypes):
        print(f"  {col}: {dtype}")


if __name__ == "__main__":
    main()
