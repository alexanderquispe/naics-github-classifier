#!/usr/bin/env python3
"""
Step 1: Generate BGE Embeddings for Repository Dataset

Generates 1024-dim BGE-large-en-v1.5 embeddings for the merged source pool.
Uses a CHUNKED, RESUMABLE design so that a crash midway only loses the chunk
in progress (~10 min of GPU time), not the entire run.

The script writes three artifacts:
  1. <base>_metadata.parquet  : small (~50 MB), written BEFORE encoding starts
                                so it survives an encoding crash.
  2. <base>_chunks/chunk_*.npy: per-chunk FP32 (rows, 1024) arrays; atomic
                                writes (tmp -> rename) so a crash never leaves
                                a corrupted .npy.
  3. <base>_vectors.npy       : final consolidated FP32 (N, 1024) array,
                                streamed in via memmap so we never hold the
                                full 5+ GB in RAM during concatenation.

Resume semantics:
  - On start, the script counts existing chunk_*.npy files in <base>_chunks/.
  - It skips those chunks and continues from where it left off.
  - To start fresh, delete the <base>_chunks/ directory.

Usage:
    python scripts/01_generate_embeddings.py \\
        --input  data/processed/source_pool_merged.parquet \\
        --output data/processed/embeddings/source_pool_embeddings.parquet \\
        --batch-size 512 --max-length 512

The `--output` value names the *base*; no combined parquet is actually written.
"""

import argparse
import gc
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import load_repos
from data.preprocessor import clean_readme, truncate_text, create_input_text  # noqa: F401
from embeddings.bge_encoder import BGEEncoder
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BGE embeddings for repositories")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file path")
    parser.add_argument(
        "--output", type=str,
        default="data/processed/embeddings/repos_with_embeddings.parquet",
        help="Naming root for outputs. The script writes <base>_metadata.parquet, "
             "<base>_chunks/, and <base>_vectors.npy alongside it (no combined "
             "parquet is created)."
    )
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="GPU batch size passed to the BGE encoder")
    parser.add_argument("--chunk-size", type=int, default=50_000,
                        help="Rows per resumable chunk (default: 50,000 ~= 10 min of GPU)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length for the encoder")
    parser.add_argument("--max-readme-chars", type=int, default=1000,
                        help="README char truncation before encoding")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N rows for testing (deterministic via random_state=42)")
    parser.add_argument("--keep-chunks", action="store_true",
                        help="Don't delete per-chunk .npy files after consolidation "
                             "(default: delete to save disk)")
    return parser.parse_args()


def derive_paths(output_arg: str, sample: int | None):
    """Derive the three output paths from the user's --output value."""
    output_path = Path(output_arg)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    base = output_path.stem
    # If --sample is passed, the chunks dir is sample-scoped so a 1k-sample
    # run can't accidentally reuse chunks from a different-N sample run.
    chunks_suffix = f"_chunks_sample{sample}" if sample else "_chunks"

    return {
        "output_path": output_path,
        "output_dir": output_dir,
        "metadata_path": output_dir / f"{base}_metadata.parquet",
        "vectors_path": output_dir / f"{base}_vectors.npy",
        "chunks_dir": output_dir / f"{base}{chunks_suffix}",
    }


def main():
    args = parse_args()
    logger = setup_logger("generate_embeddings")

    paths = derive_paths(args.output, args.sample)
    paths["chunks_dir"].mkdir(parents=True, exist_ok=True)

    # Clean stale .tmp files from a previous crash (the atomic-write pattern
    # leaves a .tmp behind if the process died between np.save and rename).
    for stale in paths["chunks_dir"].glob("chunk_*.npy.tmp"):
        logger.info(f"Removing stale .tmp from prior crash: {stale.name}")
        stale.unlink()

    # ---- Load and preprocess ----------------------------------------------
    logger.info(f"Loading repositories from {args.input}")
    df = load_repos(args.input)
    logger.info(f"Loaded {len(df):,} repositories")

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {len(df):,} repositories for testing")

    # Find README column
    readme_col = None
    for col in ["readme_content", "readme", "README"]:
        if col in df.columns:
            readme_col = col
            break

    if readme_col:
        initial_count = len(df)
        df = df[df[readme_col].notna() & (df[readme_col] != "")].reset_index(drop=True)
        logger.info(f"Dropped {initial_count - len(df):,} rows with empty README")

    # Build input_text once. create_input_text handles cleaning + truncation.
    logger.info("Creating input text from description, topics, and README...")
    desc_col = "description" if "description" in df.columns else None
    topics_col = "topics" if "topics" in df.columns else None

    input_texts = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        desc = row.get(desc_col, "") if desc_col else ""
        topics = row.get(topics_col, "") if topics_col else ""
        readme = row.get(readme_col, "") if readme_col else ""
        input_text = create_input_text(desc, topics, readme, args.max_readme_chars)
        input_texts.append(input_text)

    df = df.copy()
    df["input_text"] = input_texts

    # Drop rows whose input_text is empty (no useful signal to embed).
    df = df[df["input_text"].str.len() > 0].reset_index(drop=True)
    logger.info(f"Final dataset: {len(df):,} repositories")

    # ---- FIX 1: drop readme_content from RAM -----------------------------
    # The downstream pipeline never reads readme_content from this artifact:
    #   - step 3 (FAISS) uses only nwo/description/topics/input_text + vectors
    #   - step 4 (GPT) re-joins the full README from source_pool_merged.parquet
    # Carrying it through the encoding phase wastes 2-4 GB of RAM and pushes
    # Windows into heavy pagefile swapping, which is what filled the disk
    # on the prior 4.5-hour run.
    if readme_col and readme_col in df.columns:
        df = df.drop(columns=[readme_col])
        gc.collect()
        logger.info(f"Dropped '{readme_col}' column from memory")

    # ---- FIX 2: save metadata FIRST, before encoding ----------------------
    # If encoding crashes for any reason, we don't have to re-run the 7-min
    # preprocessing loop just to recover the metadata table.
    if "nwo" in df.columns or "repo_url" in df.columns:
        logger.info(f"Saving metadata to {paths['metadata_path']}")
        df.to_parquet(paths["metadata_path"], index=False)
        size_mb = paths["metadata_path"].stat().st_size / 1024 / 1024
        logger.info(
            f"Metadata saved ({size_mb:.1f} MB, "
            f"{len(df):,} rows x {len(df.columns)} cols)"
        )

    # ---- Initialize the encoder ------------------------------------------
    logger.info("Initializing BGE encoder...")
    encoder = BGEEncoder(use_half_precision=True)

    # ---- FIX 3: chunked encoding with resume ------------------------------
    CHUNK_SIZE = args.chunk_size
    expected_rows = len(df)
    total_chunks = math.ceil(expected_rows / CHUNK_SIZE)

    existing_chunks = sorted(paths["chunks_dir"].glob("chunk_*.npy"))
    start_chunk = len(existing_chunks)

    if start_chunk > 0:
        # Lightweight sanity check: total row count across existing chunks
        # must not exceed the current input. If it does, the chunks dir
        # belongs to a different (larger) run.
        n_in_chunks = sum(np.load(c, mmap_mode="r").shape[0] for c in existing_chunks)
        if n_in_chunks > expected_rows:
            logger.error(
                f"Existing chunks contain {n_in_chunks:,} rows total, but "
                f"current input has only {expected_rows:,} rows. The chunks "
                f"directory {paths['chunks_dir']} appears to be from a "
                f"different run. Delete it and restart."
            )
            sys.exit(1)
        # First chunk's shape[0] must equal CHUNK_SIZE (only the last chunk
        # may legitimately be shorter, and that's not the first).
        first_size = np.load(existing_chunks[0], mmap_mode="r").shape[0]
        if first_size != CHUNK_SIZE:
            logger.warning(
                f"First existing chunk has {first_size} rows but "
                f"--chunk-size={CHUNK_SIZE}. Consider deleting {paths['chunks_dir']} "
                f"to start fresh, or passing --chunk-size {first_size}."
            )
        logger.info(
            f"Resuming: {start_chunk}/{total_chunks} chunks already on disk"
        )
    else:
        logger.info(f"Starting fresh: {total_chunks} chunks to encode")

    logger.info(
        f"Generating embeddings: batch_size={args.batch_size}, "
        f"chunk_size={CHUNK_SIZE}, total_chunks={total_chunks}"
    )

    for chunk_id in range(start_chunk, total_chunks):
        chunk_start = chunk_id * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, expected_rows)
        chunk_texts = df["input_text"].iloc[chunk_start:chunk_end].tolist()

        chunk_vectors = encoder.encode(
            chunk_texts,
            batch_size=args.batch_size,
            show_progress=True,
            normalize=True,
        )

        # FIX 3 atomic write: np.save to .tmp then rename to .npy. A crash
        # mid-write leaves a .tmp file (ignored by the resume glob), never
        # a half-written .npy file.
        #
        # Subtle: np.save(path, arr) auto-appends ".npy" to `path` if it
        # doesn't already end in ".npy" — so np.save("chunk.npy.tmp", ...)
        # writes "chunk.npy.tmp.npy" and the subsequent rename fails. The
        # workaround is to pass a file-object instead of a path string; in
        # that mode np.save respects the filename verbatim.
        tmp_path = paths["chunks_dir"] / f"chunk_{chunk_id:04d}.npy.tmp"
        final_path = paths["chunks_dir"] / f"chunk_{chunk_id:04d}.npy"
        with open(tmp_path, "wb") as f:
            np.save(f, chunk_vectors)
        # .replace() overwrites on Windows; .rename() would fail if target
        # exists. We don't expect target to exist for a fresh chunk but use
        # .replace() defensively in case of a re-run.
        tmp_path.replace(final_path)

        size_mb = final_path.stat().st_size / 1024 / 1024
        logger.info(
            f"Chunk {chunk_id + 1}/{total_chunks} saved: "
            f"{chunk_vectors.shape[0]:,} rows, {size_mb:.1f} MB"
        )

    # ---- FIX 4: consolidate via memmap (never holds full array in RAM) ---
    # If a complete vectors file already exists with the right shape, skip.
    if paths["vectors_path"].exists():
        try:
            existing_shape = np.load(paths["vectors_path"], mmap_mode="r").shape
            if existing_shape == (expected_rows, 1024):
                logger.info(
                    f"Consolidated vectors file already exists at "
                    f"{paths['vectors_path']} with correct shape {existing_shape}; "
                    f"skipping consolidation."
                )
                _maybe_cleanup_chunks(paths["chunks_dir"], args.keep_chunks, logger)
                logger.info("Done!")
                return
        except Exception as e:
            logger.warning(f"Existing vectors file unreadable ({e}); will overwrite")

    # Disk check before reserving the final ~5.4 GB
    required_bytes = expected_rows * 1024 * 4
    free_bytes = shutil.disk_usage(paths["output_dir"]).free
    if free_bytes < required_bytes * 1.1:
        logger.error(
            f"Not enough free disk to write final vectors file. "
            f"Need ~{required_bytes / 1e9:.1f} GB, "
            f"have {free_bytes / 1e9:.1f} GB free. "
            f"Chunks are intact in {paths['chunks_dir']} — "
            f"free space and re-run to consolidate."
        )
        sys.exit(1)

    logger.info(
        f"Consolidating {total_chunks} chunks into {paths['vectors_path']}"
    )
    final_vectors = np.lib.format.open_memmap(
        paths["vectors_path"], mode="w+", dtype=np.float32,
        shape=(expected_rows, 1024),
    )
    cursor = 0
    for chunk_id in range(total_chunks):
        chunk = np.load(paths["chunks_dir"] / f"chunk_{chunk_id:04d}.npy")
        n = chunk.shape[0]
        final_vectors[cursor:cursor + n] = chunk
        cursor += n
        del chunk
    final_vectors.flush()
    del final_vectors

    assert cursor == expected_rows, (
        f"Consolidation mismatch: wrote {cursor:,} rows vs {expected_rows:,} expected"
    )

    size_mb = paths["vectors_path"].stat().st_size / 1024 / 1024
    logger.info(
        f"Final vectors saved: {paths['vectors_path']} "
        f"({size_mb:.1f} MB, shape ({expected_rows:,}, 1024))"
    )

    _maybe_cleanup_chunks(paths["chunks_dir"], args.keep_chunks, logger)
    logger.info("Done!")


def _maybe_cleanup_chunks(chunks_dir: Path, keep: bool, logger) -> None:
    if keep:
        logger.info(f"--keep-chunks set; leaving {chunks_dir} intact")
        return
    logger.info(f"Cleaning up chunk files in {chunks_dir}")
    for chunk_file in chunks_dir.glob("chunk_*.npy"):
        chunk_file.unlink()
    try:
        chunks_dir.rmdir()
    except OSError:
        pass  # not empty (e.g., a manifest or .tmp), leave it


if __name__ == "__main__":
    main()
