---
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
          '0': Agriculture, Forestry, Fishing and Hunting
          '1': Mining, Quarrying, and Oil and Gas Extraction
          '2': Utilities
          '3': Construction
          '4': Manufacturing
          '5': Wholesale Trade
          '6': Retail Trade
          '7': Transportation and Warehousing
          '8': Information
          '9': Finance and Insurance
          '10': Real Estate and Rental and Leasing
          '11': Professional, Scientific, and Technical Services
          '12': Administrative and Support and Waste Management and Remediation Services
          '13': Educational Services
          '14': Health Care and Social Assistance
          '15': Arts, Entertainment, and Recreation
          '16': Accommodation and Food Services
          '17': Other Services (except Public Administration)
          '18': Public Administration
  - name: code
    dtype: string
  splits:
  - name: train
    num_examples: 4611
  - name: validation
    num_examples: 659
  - name: test
    num_examples: 1318
---

# NAICS-GH

**A multi-region, GPT-4.1-labeled corpus of GitHub repositories tagged
with 2-digit NAICS industry sectors.**

NAICS-GH contains **6,588 public GitHub repositories** drawn from
the USA, the European Union, and Australia, each labeled with one of
the 19 retained 2-digit codes from the North American Industry
Classification System (NAICS 2022). Sector 55 ("Management of
Companies and Enterprises") is absent because too few repositories
passed verification under the 80-sample minimum-class-size threshold.

## How the labels were produced

A two-stage **retrieve-and-verify** pipeline:

1. A Presto/Trino query against GitHub's data warehouse yields ~1.37M
   public repositories (filters: `num_stars >= 1`, non-fork,
   non-spammy-owner, README â‰¥ 750 bytes, â‰¥ 6 commits, â‰¥ 2 contributors,
   non-empty description).
2. **BAAI/bge-large-en** embeddings indexed in FAISS retrieve the
   top-k most semantically similar repositories per NAICS subindustry
   query (k = max(20, ceil(400/n)) for a sector with n subindustries;
   k = 20 for 17 of the 20 sectors), producing **31,178 unique
   candidate (repository, sector) pairs** after per-sector
   deduplication.
3. **GPT-4.1** (snapshot `gpt-4.1-2025-04-14`) scores each candidate
   against a structured rubric on a 1â€“10 scale; only repositories
   scoring at least 8 are kept.

A stratified random sample of 2,421 USA repositories was manually
re-checked by research assistants, confirming **96.98% label
precision** on the USA portion (Wilson 95% CI [96.23%, 97.59%]).

## Splits

| Split | Rows |
|---|---:|
| train | 4,611 |
| validation | 659 |
| test | 1,318 |
| **total** | **6,588** |

Splits are stratified by NAICS sector with `random_state = 42`.
All 19 classes are present in every split.

## Schema

| Column | Type | Description |
|---|---|---|
| `name_repo` | string | Repository short name (no owner prefix) |
| `description` | string | Repository description from GitHub |
| `topics` | string | Semicolon-joined topic tags; empty if none |
| `readme_content` | string | Cleaned README text |
| `label` | int64 | Integer class encoding 0â€“18 |
| `code` | string | 2-digit NAICS sector code |

## Class labels

| label | code | NAICS sector |
|---:|---|---|
| 0 | 11 | Agriculture, Forestry, Fishing and Hunting |
| 1 | 21 | Mining, Quarrying, and Oil and Gas Extraction |
| 2 | 22 | Utilities |
| 3 | 23 | Construction |
| 4 | 31-33 | Manufacturing |
| 5 | 42 | Wholesale Trade |
| 6 | 44-45 | Retail Trade |
| 7 | 48-49 | Transportation and Warehousing |
| 8 | 51 | Information |
| 9 | 52 | Finance and Insurance |
| 10 | 53 | Real Estate and Rental and Leasing |
| 11 | 54 | Professional, Scientific, and Technical Services |
| 12 | 56 | Administrative and Support and Waste Management and Remediation Services |
| 13 | 61 | Educational Services |
| 14 | 62 | Health Care and Social Assistance |
| 15 | 71 | Arts, Entertainment, and Recreation |
| 16 | 72 | Accommodation and Food Services |
| 17 | 81 | Other Services (except Public Administration) |
| 18 | 92 | Public Administration |

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
- **English-only retrieval**: BGE-large-en is trained on English
  text; non-English READMEs are under-represented.
- **NAICS is a North American taxonomy**: Applying NAICS to EU and AU
  repositories assumes that economic activities map cleanly across
  jurisdictions. Sector 22 (Utilities) is particularly US-centric.
- **Label noise is not uniform**: At score â‰¥ 8, sectors 31â€“33
  (Manufacturing) and 42 (Wholesale Trade) have ~73% precision in the
  USA gold sample; raise the inclusion threshold to score â‰¥ 9 for
  stricter applications.
- **Repository content licensing**: The dataset releases labels
  *about* public repositories, plus excerpts of their READMEs. The
  underlying repositories remain governed by their own licenses.
