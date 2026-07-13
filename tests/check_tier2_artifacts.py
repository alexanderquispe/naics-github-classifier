"""Verify all Tier 2 pipeline artifacts exist on disk with correct shapes."""
import glob
import os

import faiss
import numpy as np
import polars as pl

BASE = r"C:\Users\Alexander\Documents\GitHub\naics-github-classifier\data\processed"
OUT = r"C:\Users\Alexander\Documents\GitHub\naics-github-classifier\data\output"


def mb(p):
    return os.path.getsize(p) / 1024 / 1024 if os.path.exists(p) else None


print("=== TIER 2 PIPELINE ARTIFACT CHECK ===\n")

# Step 1
p = os.path.join(BASE, "source_pool_merged.parquet")
if os.path.exists(p):
    df = pl.scan_parquet(p)
    n = df.select(pl.len()).collect().item()
    print(f"[1]   Merge       OK   {mb(p):7.0f} MB  rows={n:,}  cols={df.collect_schema().names()}")
else:
    print("[1]   Merge       MISSING")

# Step 2
v = os.path.join(BASE, "embeddings", "source_pool_embeddings_vectors.npy")
m = os.path.join(BASE, "embeddings", "source_pool_embeddings_metadata.parquet")
if os.path.exists(v):
    arr = np.load(v, mmap_mode="r")
    print(f"[2]   Embeddings  OK   {mb(v):7.0f} MB  shape={arr.shape}  dtype={arr.dtype}")
else:
    print("[2]   Embeddings  MISSING")
if os.path.exists(m):
    md = pl.scan_parquet(m)
    print(f"      metadata    OK   {mb(m):7.0f} MB  rows={md.select(pl.len()).collect().item():,}  cols={md.collect_schema().names()}")
else:
    print("      metadata    MISSING")

# Step 2.5
idx = os.path.join(BASE, "faiss_indices", "source_pool.index")
fm = os.path.join(BASE, "faiss_indices", "source_pool_metadata.parquet")
if os.path.exists(idx):
    index = faiss.read_index(idx)
    print(f"[2.5] FAISS       OK   {mb(idx):7.0f} MB  ntotal={index.ntotal:,}  dim={index.d}  type={type(index).__name__}")
else:
    print("[2.5] FAISS       MISSING")
if os.path.exists(fm):
    print(f"      metadata    OK   {mb(fm):7.0f} MB")
else:
    print("      metadata    MISSING")

# Step 3
c = os.path.join(BASE, "retrieval_candidates.parquet")
if os.path.exists(c):
    cd = pl.read_parquet(c)
    print(f"[3]   Retrieve    OK   {mb(c):7.1f} MB  rows={cd.height:,}  cols={cd.columns}")
    print(f"      distinct nwo={cd['nwo'].n_unique():,}  distinct naics={cd['naics_code'].n_unique()}")
else:
    print("[3]   Retrieve    MISSING")

# Step 4 (expected NOT run)
batches = glob.glob(os.path.join(OUT, "batch_csvs", "batch_*.csv"))
status4 = f"FOUND {len(batches)} batches" if batches else "NOT RUN (expected — needs internal GPT API)"
print(f"[4]   GPT classify {status4}")

# Step 5 (expected NOT run)
fin = os.path.join(BASE, "train_corpus_merged.parquet")
status5 = "DONE" if os.path.exists(fin) else "NOT RUN (expected — depends on step 4)"
print(f"[5]   Filter       {status5}")
