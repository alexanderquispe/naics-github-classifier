"""
Upload the NAICS-GH training corpus to Hugging Face as a new public dataset.

Source:  naics-github-train/data/raw/train_data_gpt_ab8_score_with_code.parquet
Target:  https://huggingface.co/datasets/aquiro1994/naics-gh

Steps:
  1. Load the 6,588-row parquet.
  2. Apply the same stratified 70/10/20 split as the training notebook
     (random_state=42, val_size=0.1, test_size=0.2).
  3. Write data/train-00000-of-00001.parquet, validation-, test-.
  4. Write README.md (HF dataset card with YAML metadata).
  5. Create the HF repo (public) and upload all files.

Token: read from environment variable HF_TOKEN. Never write to disk.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, create_repo, upload_file


REPO_ID = "aquiro1994/naics-gh"
REPO_TYPE = "dataset"
PRIVATE = False

SOURCE = Path(
    r"C:\Users\Alexander\Documents\GitHub\naics-github-train"
    r"\data\raw\train_data_gpt_ab8_score_with_code.parquet"
)
STAGING = Path(__file__).resolve().parent / "staging"


# ---- Canonical class names (id -> NAICS sector name) ----------------------
# Order matches the integer `label` field (0..18, monotonic in NAICS code).
ID2NAME = {
    0:  "Agriculture, Forestry, Fishing and Hunting",          # code 11
    1:  "Mining, Quarrying, and Oil and Gas Extraction",       # 21
    2:  "Utilities",                                           # 22
    3:  "Construction",                                        # 23
    4:  "Manufacturing",                                       # 31-33
    5:  "Wholesale Trade",                                     # 42
    6:  "Retail Trade",                                        # 44-45
    7:  "Transportation and Warehousing",                      # 48-49
    8:  "Information",                                         # 51
    9:  "Finance and Insurance",                               # 52
    10: "Real Estate and Rental and Leasing",                  # 53
    11: "Professional, Scientific, and Technical Services",    # 54
    12: "Administrative and Support and Waste Management "
        "and Remediation Services",                            # 56
    13: "Educational Services",                                # 61
    14: "Health Care and Social Assistance",                   # 62
    15: "Arts, Entertainment, and Recreation",                 # 71
    16: "Accommodation and Food Services",                     # 72
    17: "Other Services (except Public Administration)",       # 81
    18: "Public Administration",                               # 92
}


def make_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Stratified 70/10/20 with random_state=42 — matches the training script."""
    train_val, test = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42,
    )
    val_ratio = 0.1 / (1 - 0.2)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val["label"],
        random_state=42,
    )
    return {"train": train, "validation": val, "test": test}


def write_dataset_card(path: Path, splits: dict[str, pd.DataFrame]) -> None:
    """Write README.md with YAML front-matter (HF auto-generates Croissant
    from this)."""
    n_train, n_val, n_test = (len(splits[s]) for s in ("train", "validation", "test"))
    n_total = n_train + n_val + n_test

    class_names_yaml = "\n".join(
        f"          '{i}': {name}" for i, name in ID2NAME.items()
    )

    yaml = f"""---
license: cc-by-4.0
pretty_name: NAICS-GH
task_categories:
- text-classification
language:
- en
tags:
- github
- naics
- industry-classification
- weak-supervision
- llm-labeled
- gpt-4
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: name_repo
    dtype: string
  - name: description
    dtype: string
  - name: topics
    dtype: string
  - name: readme_content
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
{class_names_yaml}
  - name: code
    dtype: string
  splits:
  - name: train
    num_examples: {n_train}
  - name: validation
    num_examples: {n_val}
  - name: test
    num_examples: {n_test}
---

# NAICS-GH

**A multi-region, GPT-4.1-labeled corpus of GitHub repositories tagged
with 2-digit NAICS industry sectors.**

NAICS-GH contains **{n_total:,} public GitHub repositories** drawn from
the USA, the European Union, and Australia, each labeled with one of
the 19 retained 2-digit codes from the North American Industry
Classification System (NAICS 2022). Sector 55 ("Management of
Companies and Enterprises") is absent because too few repositories
passed verification under the 80-sample minimum-class-size threshold.

## How the labels were produced

A two-stage **retrieve-and-verify** pipeline:

1. A Presto/Trino query against GitHub's data warehouse yields ~1.37M
   public repositories (filters: `num_stars >= 1`, non-fork,
   non-spammy-owner, README ≥ 750 bytes, ≥ 6 commits, ≥ 2 contributors,
   non-empty description).
2. **BAAI/bge-large-en-v1.5** embeddings indexed in FAISS retrieve the
   top-20 most semantically similar repositories per NAICS subindustry
   query, producing **28,341 candidate (repository, sector) pairs**.
3. **GPT-4.1** (snapshot `gpt-4.1-2025-04-14`) scores each candidate
   against a structured rubric on a 1–10 scale; only repositories
   scoring at least 8 are kept.

A stratified random sample of 2,421 USA repositories was manually
re-checked by research assistants, confirming **96.98% label
precision** on the USA portion (Wilson 95% CI [96.23%, 97.59%]).

## Splits

| Split | Rows |
|---|---:|
| train | {n_train:,} |
| validation | {n_val:,} |
| test | {n_test:,} |
| **total** | **{n_total:,}** |

Splits are stratified by NAICS sector with `random_state = 42`.
All 19 classes are present in every split.

## Schema

| Column | Type | Description |
|---|---|---|
| `name_repo` | string | Repository short name (no owner prefix) |
| `description` | string | Repository description from GitHub |
| `topics` | string | Semicolon-joined topic tags; empty if none |
| `readme_content` | string | Cleaned README text |
| `label` | int64 | Integer class encoding 0–18 |
| `code` | string | 2-digit NAICS sector code |

## Class labels

| label | code | NAICS sector |
|---:|---|---|
""" + "\n".join(
        f"| {i} | {code} | {name} |"
        for (i, name), code in zip(
            ID2NAME.items(),
            ["11", "21", "22", "23", "31-33", "42", "44-45", "48-49", "51",
             "52", "53", "54", "56", "61", "62", "71", "72", "81", "92"],
        )
    ) + """

## Usage

```python
from datasets import load_dataset

ds = load_dataset("aquiro1994/naics-gh")
print(ds)

# Inspect a sample
row = ds["train"][0]
print(row["name_repo"], "->", row["code"])
```

Each repository is most easily passed to a classifier by serializing
its four text columns:

```python
text = (
    f"Repository: {row['name_repo']} | "
    f"Description: {row['description']} | "
    f"Topics: {row['topics']} | "
    f"README: {row['readme_content']}"
)
```

This is the input format used by the published RoBERTa-large
checkpoint at
[`alexanderquispe/naics-github-classifier`](https://huggingface.co/alexanderquispe/naics-github-classifier).

## License

- **Labels and metadata:** CC-BY-4.0.
- **Underlying repository content** (READMEs, descriptions) remains
  governed by each repository's own license; the `code` column is a
  derived label, not an extract of any single repository's content.

## Citation

```bibtex
@inproceedings{xu2026naicsgh,
  title  = {Industry Classification of GitHub Repositories
            Using the North American Industry Classification System (NAICS)},
  author = {Xu, Kevin and Quispe, Alexander},
  year   = {2026},
  note   = {Dataset available at https://huggingface.co/datasets/aquiro1994/naics-gh}
}
```

## Limitations and caveats

- **Validation coverage**: The 96.98% precision figure is established
  on the USA portion of the corpus only. EU and AU repositories were
  labeled by the same GPT-4.1 pipeline but have not yet been included
  in the manual gold sample.
- **English-only retrieval**: BGE-large-en-v1.5 is trained on English
  text; non-English READMEs are under-represented.
- **NAICS is a North American taxonomy**: Applying NAICS to EU and AU
  repositories assumes that economic activities map cleanly across
  jurisdictions. Sector 22 (Utilities) is particularly US-centric.
- **Label noise is not uniform**: At score ≥ 8, sectors 31–33
  (Manufacturing) and 42 (Wholesale Trade) have ~73% precision in the
  USA gold sample; raise the inclusion threshold to score ≥ 9 for
  stricter applications.
- **Repository content licensing**: The dataset releases labels
  *about* public repositories, plus excerpts of their READMEs. The
  underlying repositories remain governed by their own licenses.
"""

    path.write_text(yaml, encoding="utf-8")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN environment variable not set.")

    df = pd.read_parquet(SOURCE)
    print(f"Loaded {len(df):,} rows from {SOURCE.name}")
    assert len(df) == 6588, f"expected 6,588 rows, got {len(df)}"

    splits = make_splits(df)
    print(
        "Splits: "
        f"train={len(splits['train']):,}, "
        f"validation={len(splits['validation']):,}, "
        f"test={len(splits['test']):,}"
    )

    STAGING.mkdir(parents=True, exist_ok=True)
    data_dir = STAGING / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Drop pandas index, sort columns to canonical order
    canonical_cols = [
        "name_repo", "description", "topics", "readme_content", "label", "code",
    ]
    for name, frame in splits.items():
        out = frame[canonical_cols].reset_index(drop=True)
        path = data_dir / f"{name}-00000-of-00001.parquet"
        out.to_parquet(path, index=False)
        print(f"  Wrote {path.relative_to(STAGING)}  ({len(out):,} rows, {path.stat().st_size/1024:.0f} KB)")

    readme = STAGING / "README.md"
    write_dataset_card(readme, splits)
    print(f"  Wrote {readme.relative_to(STAGING)}  ({readme.stat().st_size/1024:.0f} KB)")

    print(f"\nCreating repo {REPO_ID} (public={not PRIVATE})...")
    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=token,
        private=PRIVATE,
        exist_ok=True,
    )

    print("Uploading files...")
    api = HfApi(token=token)
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(STAGING),
        commit_message="Initial NAICS-GH v1.0 upload (6,588 rows, train/val/test splits, Croissant metadata)",
    )
    print(f"\n✓ Done. Public URL: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
