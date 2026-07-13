"""
Merge the three regional source/intermediate parquets into a single
country-blind file before publication.

Per the released-data policy, no per-row jurisdictional attribute may
be published. This script:
  1. Reads each regional parquet lazily (Polars).
  2. Drops the `country_code` column (and any other per-row jurisdictional
     attribute such as `country_account`) if present.
  3. Harmonises column names across the three regions
     (`description_clean` -> `description`, `topics_clean` -> `topics`).
  4. Subsets every file to the canonical pipeline columns so any
     EU/AU-only metadata is dropped before publication.
  5. Concatenates the (now schema-aligned) frames.
  6. Shuffles deterministically (seed=42) so the regional order of the
     source files cannot be inferred from row position.
  7. Writes a single merged parquet.

Polars is used for speed: lazy reads + a single materialised concat
keeps peak memory roughly bounded by one file at a time.

Usage:
    python scripts/merge_and_strip_country.py \\
        --inputs data/raw/repos_usa_usa_511k.parquet \\
                 data/raw/repos_eu_531k.parquet \\
                 data/raw/repos_au_331k.parquet \\
        --output data/processed/source_pool_merged.parquet

The script refuses to write the output if any disallowed column
survives the merge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


# Columns we never want in the published artifact, regardless of
# whatever else is in the source files.
DISALLOWED_COLUMNS = {
    "country_code",
    "country_account",
    "owner_country",
    "country",
    "jurisdiction",
    "region",
}

# The four columns the downstream pipeline (BGE embedding,
# FAISS retrieval, GPT classification) actually consumes.
# `nwo` ("owner/repo") is the unique GitHub identifier; the
# repo name on its own is derivable from it.
CANONICAL_COLUMNS = [
    "nwo",
    "description",
    "topics",
    "readme_content",
]

# Per-region column-name aliases. The USA source parquet was
# pre-cleaned and renames description -> description_clean and
# topics -> topics_clean; the EU and AU parquets keep the raw
# column names. We harmonise both into the canonical names above.
COLUMN_ALIASES = {
    "description_clean": "description",
    "topics_clean": "topics",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs", nargs="+", required=True,
        help="Regional parquet files to merge (any number, order-insensitive).",
    )
    p.add_argument(
        "--output", required=True,
        help="Output path for the merged, country-blind parquet.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the row shuffle (default: 42).",
    )
    return p.parse_args()


def normalise_one(path: Path) -> pl.LazyFrame:
    if not path.exists():
        sys.exit(f"Input file not found: {path}")

    lf = pl.scan_parquet(path)
    cols = lf.collect_schema().names()
    n_source_cols = len(cols)

    # Drop disallowed columns at the per-file level so we never
    # even materialise them.
    per_file_disallowed = sorted(set(cols) & DISALLOWED_COLUMNS)
    if per_file_disallowed:
        lf = lf.drop(per_file_disallowed)
        cols = [c for c in cols if c not in per_file_disallowed]

    # Harmonise column names (description_clean -> description, etc.)
    renames = {k: v for k, v in COLUMN_ALIASES.items() if k in cols}
    if renames:
        lf = lf.rename(renames)
        cols = [renames.get(c, c) for c in cols]

    # Refuse if anything required is missing — that's a real data
    # problem, not something to paper over with nulls.
    missing = [c for c in CANONICAL_COLUMNS if c not in cols]
    if missing:
        sys.exit(
            f"{path.name} is missing required pipeline columns: {missing}"
        )

    lf = lf.select(CANONICAL_COLUMNS)

    # Harmonise dtypes across regions. The AU parquet stores topics
    # as the raw GitHub List[String]; USA and EU store it pre-joined
    # as String. A subsequent vertical concat refuses to mix the two,
    # so we normalise here: list -> comma-joined string, anything else
    # -> cast to string.
    schema = lf.collect_schema()
    if isinstance(schema["topics"], pl.List):
        lf = lf.with_columns(pl.col("topics").list.join(", "))
    elif schema["topics"] != pl.String:
        lf = lf.with_columns(pl.col("topics").cast(pl.String))
    for col in ("nwo", "description", "readme_content"):
        if schema[col] != pl.String:
            lf = lf.with_columns(pl.col(col).cast(pl.String))

    # Cheap row count via collect of a select(len)
    n_rows = lf.select(pl.len()).collect().item()
    print(
        f"  Loaded {path.name}: {n_rows:,} rows "
        f"({n_source_cols} cols in source -> {len(CANONICAL_COLUMNS)} canonical)"
    )
    return lf


def main() -> None:
    args = parse_args()

    frames = [normalise_one(Path(p)) for p in args.inputs]

    merged = pl.concat(frames, how="vertical").collect()
    print(f"Merged: {merged.height:,} rows, {merged.width} cols")

    # Defense in depth: re-check disallowed columns after concat.
    leftover = sorted(set(merged.columns) & DISALLOWED_COLUMNS)
    if leftover:
        sys.exit(
            f"Refusing to write: disallowed columns survived merge: {leftover}"
        )

    # Deterministic shuffle so the original regional order is not
    # recoverable from row position.
    merged = merged.sample(fraction=1.0, shuffle=True, seed=args.seed)
    print(f"After shuffle (seed={args.seed}): {merged.height:,} rows")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path}  ({size_mb:.1f} MB)")

    print("\nFinal schema:")
    for col, dtype in zip(merged.columns, merged.dtypes):
        print(f"  {col}: {dtype}")


if __name__ == "__main__":
    main()
