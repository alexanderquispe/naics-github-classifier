# NAICS-GH Training Data Pipeline

End-to-end documentation of how the NAICS-GH labeled corpus was
produced — from raw regional GitHub dumps to the parquet file that
fine-tunes RoBERTa-large in the sibling `naics-github-train` repo.

This is the canonical reference. The paper text in `paper/main.tex`
should be consistent with this document; if they disagree, this
document is right and the paper is stale.

---

## 0. Diagram (data flow)

```
                       ┌─────────────────────────┐
                       │  NAICS sector titles    │
                       │  6-digit subindustries  │
                       │   (JSON, 20 sectors)    │
                       └────────────┬────────────┘
                                    │
  ┌──────────────┐   ┌──────────────┴───────────────┐   ┌──────────────┐
  │  USA dump    │   │  EU dump                     │   │  AU dump     │
  │  511k repos  │   │  531k repos                  │   │  331k repos  │
  └──────┬───────┘   └──────────────┬───────────────┘   └──────┬───────┘
         │                          │                          │
         ▼                          ▼                          ▼
    [Step 1] BGE embeddings (1024-dim, L2-normalized)
             (original 2025 Colab run: bge-large-en **v1** on README-only
              text [:1000]; current pipeline: bge-large-en-**v1.5** on
              description+topics+readme — see funnel section)
         │                          │                          │
         ▼                          ▼                          ▼
    [Step 2] FAISS IndexFlatIP (exact inner-product / cosine)
         │                          │                          │
         ▼                          ▼                          ▼
    [Step 3] Load NAICS JSON (20 sectors → 1,029 subindustry phrases).
             For each sector: split its ';'-string into subindustries.
             For each subindustry: query = "Repositories about {subindustry}"
                                   retrieve top effective_k candidates
                                   (k=20 for 17 sectors; k=24/27/29 for the
                                    3 sectors with <20 subindustries).
             → tag rows with (naics_group, subindustry, similarity)
             → 20,879 raw rows per region (Σ n×k; overlap allowed)
             → dedup on (nwo, naics_group) at GPT-prep
         │                          │                          │
         ▼                          ▼                          ▼
   10,983 candidates        10,531 candidates         9,664 candidates
         │                          │                          │
         ▼                          ▼                          ▼
    [Step 4] GPT-4.1 rubric scoring (1–10) + JSON output
             keep score ≥ 8
         │                          │                          │
         ▼                          ▼                          ▼
   usa_2k_gpt_ab8score      eu_2k_gpt_ab8score        au_2k_gpt_ab8score
   (2,529 rows)             (2,039 rows)              (2,021 rows)
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    ▼
                  [Step 5] pd.concat([au, eu, usa])
                                    │
                                    ▼
                  train_data_naics_github.parquet
                  (7,057 rows · 20 sectors · full provenance)
                                    │
                                    ▼   [Step 6] min_samples_per_class=80
                                    │           drops sector 55 (13 rows)
                                    │           + dedup/quality filter
                                    ▼
                  train_data_gpt_ab8_score_with_code.parquet
                  (6,588 rows · 19 sectors · 4 text + label + code)
                                    │
                                    ▼   [Step 7] scripts/train.py
                                    │           in naics-github-train
                                    ▼
                  RoBERTa-large fine-tuned classifier
                  (86.45% F1 / 86.35% accuracy on 1,318 test rows)
                  huggingface.co/alexanderquispe/naics-github-classifier
```

---

## 0.5 Two-tier architecture: internal regional vs. public merged

The pipeline runs in **two tiers** that share the same model, prompt, and
filtering logic but differ in one dimension: whether jurisdiction (USA /
EU / AU) is retained.

### Tier 1 — Internal canonical pipeline (regional, with country)

- **Where it runs:** inside GitHub, on jurisdiction-tagged source data.
- **How:** the three regional pools are kept separate. Each is embedded,
  indexed, and retrieved independently (the diagram above). Candidates
  carry their region.
- **What it produces:**
  1. The **canonical labeled corpus** (the one the paper benchmarks on
     and the one published to Hugging Face — *with the country column
     stripped at the final step*).
  2. The **cross-jurisdiction analysis** the paper reports: sector mix
     by jurisdiction, and the distribution of GPT-4.1 rubric scores by
     jurisdiction. These are **aggregate figures**, not row-level
     attributes, and are therefore publishable.
- **Country handling:** retained throughout Tier 1 for analysis, then
  dropped (and rows shuffled) before any artifact leaves GitHub. No
  published row ever carries a jurisdiction label.

### Tier 2 — Public replication pipeline (merged, country-blind)

- **Where it runs:** anywhere, by anyone, on data we can publish.
- **How:** the three regional pools are concatenated into a single
  jurisdiction-blind pool *before* embedding
  (`scripts/merge_and_strip_country.py`), then a single embedding /
  index / retrieval / GPT / filter pass runs over the merged pool.
- **What it produces:** a corpus of **comparable size and quality** to
  the canonical one — *not bit-identical*. The merged retrieval picks
  the best candidates regardless of origin, rather than enforcing a
  per-region quota, so the candidate set differs.
- **Why it exists:** it lets an external reviewer reproduce the
  *methodology* end-to-end without ever touching jurisdiction-tagged
  data.

### Why the two tiers are not identical (and why that is fine)

| Dimension | Tier 1 (internal) | Tier 2 (public) |
|---|---|---|
| Source | 3 regional pools | 1 merged pool |
| Retrieval depth | `base_k`/region × 3 regions | `base_k` once |
| Country column | retained, then stripped on export | never present |
| Candidate set | regional quotas | best-overall |
| Corpus | canonical (benchmarked) | comparable reproduction |

Both tiers use **BGE-large-en-v1.5**, the **combined `description +
topics + readme` input**, the **`"Represent this document/query for
retrieval: "` prefixes**, **GPT-4.1 at temperature 0**, and the
**score ≥ 8** filter. They differ *only* in the partition. That makes
the "comparable, not identical" claim tight and defensible.

> **Disclosure text for the paper / replication README:**
> *"The released corpus was generated internally via a per-jurisdiction
> pipeline. To enable external reproduction without disclosing
> jurisdictional attributes, we additionally provide a
> jurisdiction-blind merged pipeline that reproduces the methodology and
> yields a corpus of comparable size and quality. Aggregate
> jurisdictional statistics are reported in the paper; no per-row
> jurisdiction attribute is released in any artifact."*

> **Efficiency note:** Tier 1 does not require a separate embedding run.
> Per-repo BGE embeddings are independent of which index they live in, so
> the merged-pool embeddings (Step 1) can be **partitioned by jurisdiction
> via `nwo` membership in the three regional source files** and used to
> build three regional FAISS indices. Only retrieval (Step 3) and GPT
> scoring (Step 4) re-run per region.

---

## Pipeline funnel — the numbers that go in the paper

> ⚠️ **Corrected 2026-06-10.** An earlier version of this section
> reported **28,341** candidates (10,983 USA + 9,062 EU + 8,296 AU).
> That figure mixed two different runs: the USA count came from the
> original Colab run, but the EU/AU counts came from a **later local
> rerun with a different embedding configuration** whose candidates
> never fed GPT. The verified original counts below were read directly
> from the executed cell outputs of the Colab notebooks
> (`4_gpt_clasif_repos_stats`, `eu_gpt_clasif_repos_stats`,
> `au_gpt_clasif_repos_stats`, `3_eu_repo_gpt_classif`,
> `1_au_index_metadata_creation`).

The verified original (2025 Colab) funnel:

> From a source pool of $\sim$1.37M public GitHub repositories across
> USA, EU, and AU, the BGE+FAISS retrieval step produces
> **31,178 candidate (repository, NAICS sector) pairs** (10,983 USA +
> 10,531 EU + 9,664 AU). After GPT-4.1 rubric scoring with a
> score-$\geq$8 threshold (6,589 pairs) and class-size cleanup, the
> released NAICS-GH training corpus contains **6,588 high-confidence
> repositories** across 19 of the 20 NAICS sectors.

| Stage | USA | EU | AU | Total |
|---|---:|---:|---:|---:|
| **Source pool** (Step 0 SQL output, unique nwo) | 510,380 | 530,898 | 331,211 | **1,372,489** |
| **Step 3 raw retrieval** (Σ n×k, k = max(20, ⌈400/n⌉), overlap allowed) | 20,879 | 20,879 | 20,879 | **62,637** |
| **Step 3 candidates** (dedup on (nwo, naics_group)) | 10,983 | 10,531 | 9,664 | **31,178** |
| **Step 4 score ≥ 8** (GPT-4.1 verification) | 2,529 | 2,039 | 2,021 | **6,589** |
| **Step 6 training-ready** (`train_data_gpt_ab8_score_with_code.parquet`, 19 sectors) | — | — | — | **6,588** |

**End-to-end retention:** $6{,}588 / 1{,}372{,}489 \approx$ **0.48%** of
the source pool ends up in the training file. From candidates to
training: $6{,}588 / 31{,}178 \approx$ **21.1%** survive GPT
verification + cleanup. (The 6,589 → 6,588 one-row difference arises at
the concat/cleanup step and is still to be pinned down.)

**Original-run configuration (what produced these numbers):**
documents embedded as **README-only, first 1,000 chars**, with
**bge-large-en (v1)**; queries with the asymmetric
`"Represent this query for retrieval: "` prefix; the combined
`description + topics + readme` `input_text` was built **only for the
GPT step**, not for embeddings.

**The replication pipeline intentionally diverges** (both tiers):
**bge-large-en-v1.5** and **desc+topics+readme embedding input** (same
asymmetric prefixes, same k formula, same raw depth of 20,879/region).
In the v1.5 full-text space, subindustry queries within a sector
overlap more, so within-sector dedup is higher (68–75% vs 47–54%) and
candidate counts are lower: **6,607 USA + 6,138 EU + 5,224 AU = 17,969**
(Tier 1 regional, 2026-06-10) and **18,011** (Tier 2 merged at
base_k=1200/min_k=60). The raw-depth identity (20,879 per region)
confirms the protocol matches; the dedup difference is a property of
the improved embedding space, not a bug.

**Faithful replication verified (2026-06-10):** the original
configuration was re-run end-to-end through retrieval in
`naics-tier1-replication` (local RTX 3080). Raw retrieval matched
exactly (20,879 per region, asserted), the AU pool/index matched
exactly (331,374), and the deduped candidate counts came within
±0.1% per region: USA 10,985 (+2), EU 10,529 (−2), AU 9,674 (+10);
total 31,188 vs 31,178 (+0.03%). Residual deltas are FP16 borderline
neighbors (original A100 vs local RTX 3080). This is the
reproducibility evidence for the paper; the GPT re-labeling step runs
from `train_data_for_gpt_{usa,eu,au}.parquet` in that folder.

**Provenance of the superseded 9,062 / 8,296 figures:** local rerun
files `data/processed/retrieved_repos_by_naics.parquet` (EU) and
`au_retrieved_repos_by_naics.parquet` (AU), produced with bge-large-en
v1, desc+topics+readme input, and a **symmetric** (document) prefix on
queries — a config that matches neither the Colab original nor the
current pipeline. Do not mix them into any funnel.

**Notes for paper text:**

- The USA candidate count (10,983) matches slide 8 of the May 22, 2026
  presentation and the executed output of `4_gpt_clasif_repos_stats.ipynb`
  (count of the production `batch_csvs` GPT outputs); the production
  retrieval parquet (`retrieved_repos_by_naics_group_511k.parquet`,
  20,879 raw rows) lives in Google Drive.
- The EU count (10,531) is the executed dedup output of
  `3_eu_repo_gpt_classif.ipynb` cell 20 / `eu_gpt_clasif_repos_stats.ipynb`.
- The AU count (9,664) is the executed dedup output of
  `1_au_index_metadata_creation.ipynb` cell 52 /
  `au_gpt_clasif_repos_stats.ipynb`.
- Score ≥ 8 counts: `usa_2k_gpt_ab8score.parquet` (2,529 rows, verified
  on disk in `Dropbox/github_laptop/.../2_output/`), EU 2,039 and AU
  2,021 from the executed stats notebooks.
- The released training file `train_data_gpt_ab8_score_with_code.parquet`
  has 6,588 rows; it is the input to `scripts/train.py` and produced
  the RoBERTa-large 86.45% F1 / 86.35% accuracy result (best of six
  pretrained encoders benchmarked --- see Step 7).

### ⚠️ Open issue: GPT re-classification for EU + AU (planned)

The intent of the training script
(`Colab Notebooks/naics_training_gptdata.ipynb`) is to read three
all-GPT regional files:

```python
au_data  = pd.read_parquet(f'{DIR}/au_2k_gpt_ab8score.parquet')
eu_data  = pd.read_parquet(f'{DIR}/eu_2k_gpt_ab8score.parquet')
usa_data = pd.read_parquet(f'{DIR}/usa_2k_gpt_ab8score.parquet')
raw_data = pd.concat([au_data, eu_data, usa_data], ignore_index=True)
```

However, only `usa_2k_gpt_ab8score.parquet` is currently confirmed to
be on local disk (in `Dropbox/github_laptop/.../2_output/`); the EU
and AU equivalents live in Google Drive.

Independent inspection of the released `train_data_naics_github.parquet`
(an intermediate 7,057-row artifact) revealed that its per-region
score distributions match **GPT-4.1 for USA but DeepSeek-V3 for EU
and AU** — i.e., that intermediate file was built from a mixed-LLM
source, not from the three all-GPT files the notebook references.

**Planned action:** Alexander will **re-run GPT-4.1 classification on
the EU and AU candidate pools** to produce true `eu_2k_gpt_ab8score.parquet`
and `au_2k_gpt_ab8score.parquet` files, then re-run the concatenation
and training-data preparation so that the released NAICS-GH corpus is
unambiguously GPT-only across all three regions. The 6,588-row count
above will likely shift slightly after re-classification (since GPT-4.1
and DeepSeek-V3 do not agree on every label). Note (2026-06-10): the
planned path is now the Tier-1 regional re-run (17,969 candidates, see
funnel section), which regenerates the GPT scores for ALL three regions
with a single consistent configuration — superseding the partial EU/AU
re-classification described above.

---

## Step 0 — Raw inputs

Three regional GitHub repository dumps and one taxonomy file. All
three regional files share the same schema; the regional differences
are only which repositories appear inside.

### Source extraction (the SQL)

The regional parquets are derived from GitHub's internal Trino-on-Hive
data warehouse via a Presto/Trino query. The canonical extraction
notebooks live in
`C:\Users\Alexander\Dropbox\github_laptop\GitHub\naics-repo-classification\1_scripts\`:

| Notebook | Purpose |
|---|---|
| `10_sql_extract_data.ipynb` | Main extraction with all metadata (one cell per region: US, EU, AU, plus emerging markets) |
| `10_sql_extract_data_readme.ipynb` | Two-stage variant that builds a presample of IDs first, then fetches README content separately |
| `11_sql_data_append.ipynb` | Per-country sampling / append into the merged regional file |
| `us_repos_presample_no_readmes_2025_04_15.ipynb` | Pandas-level filter grid-search on US presample data |

Data freshness: **April 15, 2025** for the US extraction (per
filename); EU readme extraction in `10_sql_extract_data_readme.ipynb`
was saved as `repos_eu_08_15_2025.parquet` (August 15, 2025).

#### Canonical extraction SQL (per-region)

This is the **actual** production SQL used to produce the three
regional parquets (`repos_usa_usa_511k.parquet`,
`repos_eu_531k.parquet`, `repos_au_331k.parquet`). One execution per
region, varying only the country list. Source:
`10_sql_extract_data.ipynb` cell 24 (AU variant shown):

```sql
SELECT
    nwo,
    num_stars,
    is_fork,
    is_archived,
    has_readme,
    content_size AS readme_content_size,
    disk_usage_bytes,
    spdx_license,
    repositories_current.id AS repository_id,
    from_utf8(repositories_current.description) AS description,
    repositories_current.topics,
    from_utf8(readme_current.content) AS readme_content,
    accounts_current.country_account AS country_code,
    COUNT(DISTINCT issues_current.id) AS num_issues,
    COUNT(DISTINCT issues_current.user_dotcom_id) AS num_issue_authors,
    COUNT(DISTINCT github_collab_commit_contributions.user_id) AS num_commit_contributors,
    SUM(github_collab_commit_contributions.commit_count) AS num_commits,
    MAX(github_collab_commit_contributions.created_at) AS last_commit_at
FROM hive.canonical.repositories_current
JOIN hive.canonical.accounts_current
    ON repositories_current.owner_dotcom_id = accounts_current.dotcom_id
LEFT JOIN hive.canonical.issues_current
    ON issues_current.repository_id = repositories_current.id
    AND NOT issues_current.is_user_hidden
JOIN hive.suez.readme_current
    ON readme_current.repository_id = repositories_current.id
    AND LOWER(path) = 'readme.md'
    AND LOWER(filename) = 'readme.md'
LEFT JOIN hive.reference.licenses
    ON licenses.id = repositories_current.license_id
LEFT JOIN delta.snapshots.github_collab_commit_contributions
    ON github_collab_commit_contributions.repository_id = repositories_current.id
WHERE NOT is_spammy_owner
    AND is_public = TRUE
    AND has_readme = TRUE
    AND num_stars >= 1
    AND is_fork = FALSE
    AND accounts_current.country_account IN ( /* see country lists below */ )
    AND content_size >= 750                          -- README ≥ 750 bytes
    AND disk_usage_bytes > 0
    AND repositories_current.description IS NOT NULL
    AND TRIM(from_utf8(repositories_current.description)) != ''
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12
HAVING SUM(github_collab_commit_contributions.commit_count) > 5    -- ≥ 6 commits
    AND COUNT(DISTINCT github_collab_commit_contributions.user_id) >= 2  -- ≥ 2 contributors
ORDER BY repository_id
```

#### Country lists used

| Region | Country codes (ISO-3166-1 alpha-2) |
|---|---|
| **USA** | `'US'` |
| **EU** | `'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK'` (EU-27, no UK) |
| **AU** | `'AU'` |

Jurisdiction is determined by the **`country_account` field of the
owner account** in `hive.canonical.accounts_current`, joined on
`repositories_current.owner_dotcom_id = accounts_current.dotcom_id`.
This is the country on the owning GitHub account, not the country of
top contributors or any geo-IP heuristic.

#### What the SQL filters

| Filter | Value | Effect |
|---|---|---|
| `NOT is_spammy_owner` | — | GitHub's internal spam-owner flag honored. |
| `is_public = TRUE` | — | Public repos only. |
| `has_readme = TRUE` | — | Repository declares a README. |
| `num_stars >= 1` | ≥1 | At least one star (very mild quality signal). |
| `is_fork = FALSE` | — | Forks excluded. |
| `country_account IN (…)` | per-region | Jurisdiction split (see country lists above). |
| `LOWER(path) = 'readme.md' AND LOWER(filename) = 'readme.md'` | case-insensitive | Only canonical top-level README files. |
| `content_size >= 750` | ≥750 bytes | README must be at least 750 bytes (~100 words). |
| `disk_usage_bytes > 0` | >0 | Repository has actual content on disk. |
| `description IS NOT NULL AND TRIM ≠ ''` | — | Non-empty description required. |
| `HAVING SUM(commit_count) > 5` | ≥6 commits | At least 6 commits total. |
| `HAVING COUNT(DISTINCT user_id) >= 2` | ≥2 contributors | At least 2 distinct commit authors. |

**❗ Note:** Kevin's earlier verbal summary of the SQL (which listed
`num_stars >= 5` and `content_size >= 1024` with no country/contributors
clauses) was a simplification. The canonical production SQL is the one
above. Differences from the simplified version:

| Filter | Simplified version | Canonical SQL |
|---|---|---|
| `num_stars` | `>= 5` | `>= 1` |
| `content_size` | `>= 1024` | `>= 750` |
| `path` | `'README.md'` case-sensitive | `LOWER(path) = 'readme.md'` |
| Jurisdiction | not present | `country_account IN (…)` |
| Contributors | not present | `>= 2` via HAVING |
| Commit count | not present | `> 5` via HAVING |
| Description | not present | `IS NOT NULL` |
| Disk usage | not present | `> 0` |

**Underlying pool context:** GitHub has approximately **5,695,121
public repositories with README files**. The canonical SQL with the
EU country list returns approximately 3.56M EU repository IDs before
the README content filters; further filtering and per-country
balancing produces the 531k EU file. Similar accounting holds for
USA and AU.

#### Downstream pandas filtering (exploratory)

`us_repos_presample_no_readmes_2025_04_15.ipynb` shows an exploratory
grid search over additional filters that **may or may not** all be
applied to the final regional parquets — confirm against the
`11_sql_data_append.ipynb` workflow:

```python
filtered_df = df[
    (~df['is_archived']) & (~df['is_fork']) &
    (df['disk_usage_bytes'] > 0) &
    (df['other_lang_md'] == 1) &              # English-language README only
    (df['num_commits'] > 5) &
    (pd.to_datetime(df['last_commit_at']) >= datetime(2015, 1, 1)) &
    (df['num_commit_contributors'] >= 2) &
    (df['readme_content_size'] >= 500)        # tested 500, 750, 1000
]
```

Filter candidates considered downstream (some redundant with SQL):

- `is_archived = FALSE` (archived repos excluded — **NOT** in SQL, only here)
- `other_lang_md == 1` (likely an English-language README flag; **NEW** filter not visible in the SQL extraction above)
- `last_commit_at >= 2015-01-01` (drops very-old abandoned repos)
- `readme_content_size >= {500, 750, 1000}` (final value was 750 in production based on the SQL)

### Repository dumps (output of the SQL above, post jurisdiction split)

| File | Rows | Path |
|---|---:|---|
| `repos_usa_usa_511k.parquet` | 511,001 | `data/raw/` |
| `repos_eu_531k.parquet` | 531,000 | `data/raw/` |
| `repos_au_331k.parquet` | 331,000 | `data/raw/` |

**Per-row schema** as emitted by the SQL:

| Column | Type | Source | Notes |
|---|---|---|---|
| `repository_id` | bigint | `hive.canonical.repositories_current.id` | The numeric GitHub repository ID (stable). |
| `owner` | string | `split_part(nwo, '/', 1)` | Owner login. |
| `repo` | string | `split_part(nwo, '/', 2)` | Short repository name. |
| `description` | string | `from_utf8(description)` | May be NULL. |
| `topics` | string | `array_join(topics, '; ')` | Single string, `'; '`-joined. NULL where the repo declared no topics. |
| `readme_content` | string | `from_utf8(content)` | The literal README.md bytes, UTF-8-decoded. |

Downstream code in `naics-github-classifier` looks for slightly
different column names (`nwo`, `name_repo`, `readme_content`); the
mapping is presumably done in the notebook that loads these files,
because the merged `train_data_naics_github.parquet` contains `nwo`,
`name_repo`, and `readme_content` columns.

**❓ Open questions remaining for Step 0:**

- **Jurisdiction rule.** How is each repository assigned to USA vs EU
  vs AU? Owner-account country? Top-contributor country? Geo-IP of
  primary committer? This is the largest remaining gap in the paper's
  reproducibility story.
- **2-contributors filter.** Was a $\geq 2$ contributors rule applied
  downstream of the SQL, or is the design-discussion comment about
  this filter *aspirational* and not actually reflected in the
  released dataset?

### NAICS taxonomy

| File | Path |
|---|---|
| `naics_titles_by_group_6digit_clean.json` | `data/raw/` |

A pre-existing JSON taxonomy with **20 keys**, one per 2-digit NAICS
sector. Each value is a single string of subindustry titles separated
by `;`, sourced from the 6-digit NAICS hierarchy and grouped by their
parent 2-digit code. The taxonomy is consumed as-is — our pipeline
does not perform any 6-to-2-digit collapsing.

Example:

```json
{
  "11": "Soybean Farming;Oilseed Farming;Dry Pea and Bean Farming;Wheat Farming;...",
  "21": "Crude Petroleum Extraction;Natural Gas Extraction;Surface Coal Mining;...",
  ...
}
```

**Total subindustries across the 20 sectors: 1,029.** Distribution:

| Sector | # subindustries | | Sector | # subindustries |
|---|---:|---|---|---:|
| 11 Agriculture | 64 | | 53 Real Estate | 24 |
| 21 Mining | 21 | | 54 Prof./Sci./Tech. | 49 |
| 22 Utilities | **14** | | 55 Management | 20 |
| 23 Construction | 31 | | 56 Admin/Waste | 44 |
| 31-33 Manufacturing | **346** | | 61 Educational | **17** |
| 42 Wholesale | 69 | | 62 Health Care | 39 |
| 44-45 Retail | 57 | | 71 Arts/Entertainment | 25 |
| 48-49 Transportation | 57 | | 72 Accommodation/Food | **15** |
| 51 Information | 29 | | 81 Other Services | 44 |
| 52 Finance | 35 | | 92 Public Admin | 29 |

Sectors with notable extremes are bolded: Manufacturing has 346
subindustries (the most by far); Utilities (14), Accommodation/Food
(15), and Educational Services (17) have the fewest. The
subindustry count determines the per-query `effective_k` in Step 3
(see §Step 3).

---

## Step 1 — Embedding generation

Convert each repository's text into a 1024-dim semantic vector.

### Script (local repo, English-language reference)
`scripts/01_generate_embeddings.py` → uses `src/embeddings/bge_encoder.py`.

### Notebook (actual production)
`Colab Notebooks/1_eu_index_metadata_creation.ipynb` (EU; AU and USA
have parallel notebooks).

### What it does

1. **Filter rows** with non-empty README
   (`scripts/01_generate_embeddings.py:62-72`).
2. **Build the input text** for each repository:
   ```
   Description: {description} | Topics: {topics} | README: {cleaned_readme}
   ```
   (Implemented in `src/data/preprocessor.py:create_input_text`.)
3. **Clean the README** before composing the input text
   (`src/data/preprocessor.py:clean_readme`):
   - Strip triple-backtick code blocks and single-backtick inline code.
   - Strip HTML tags (`<...>`).
   - Strip URLs (`https?://\S+`).
   - Collapse whitespace.
4. **Truncate** the cleaned README to **1,000 characters** before
   appending to the input text (`max_readme_chars=1000`).
5. **Embed** each input text with **BAAI/bge-large-en-v1.5** (current
   pipeline; see the resolved note below on the original run):
   - Model loaded via `sentence-transformers`.
   - Half-precision FP16 on GPU.
   - Document prefix `"Represent this document for retrieval: "` added
     to every text (`src/embeddings/bge_encoder.py`).
   - Output is **L2-normalized** so inner-product = cosine similarity.
   - Batch size 1024.

### Outputs (per region)

| Artifact | Path / name |
|---|---|
| Full parquet (metadata + embedding column) | `data/processed/embeddings/repos_with_embeddings.parquet` |
| Metadata-only parquet (no vectors) | same name with `_metadata.parquet` suffix |
| Embedding matrix | same name with `_vectors.npy` suffix, shape (N, 1024), float32 |

In production each region produces these under its own filename, e.g.
`eu_repos_with_embeddings_bge_531k.parquet` (per `config/config.yaml`).

### ✅ Resolved (2026-06-10, from executed notebook cells)

- **Model:** the original Colab production used the older
  **`BAAI/bge-large-en` (v1)** — explicit in
  `1_eu_index_metadata_creation.ipynb` cells 12 + 28 and
  `1_au_index_metadata_creation.ipynb` cells 12 + 28
  (`SentenceTransformer('BAAI/bge-large-en')`). Slide 7's
  "bge-large-en-v1.5" was inaccurate. The current pipeline uses
  **v1.5** as a deliberate upgrade.
- **Embedding input:** the original Colab production embedded
  **README-only, truncated to 1,000 chars**
  (`df["readme_clean"] = df["readme_content"].str[:1000]`, cell 11) —
  NOT the combined input text. The combined
  `description + topics + readme` text was built **only for the GPT
  step** (e.g., AU notebook cell 50). The current pipeline embeds the
  combined text as a deliberate improvement; this is the main driver of
  the candidate-count difference documented in the funnel section.

---

## Step 2 — FAISS index build

Make the embeddings searchable by similarity.

### Script
`scripts/02_build_faiss_index.py` → `src/embeddings/faiss_indexer.py`.

### What it does

1. Load embeddings as a `float32` (N, 1024) matrix.
2. Build `faiss.IndexFlatIP(1024)` — exact inner-product index. Because
   vectors are L2-normalized in Step 1, IP scores equal cosine
   similarities in `[-1, 1]`.
3. (Optional) move the index to GPU for build, then back to CPU for
   save.
4. Persist the index alongside its metadata.

**No approximation, no quantization, no IVF.** At 0.5M–1M vectors per
region the exact search is still fast enough (sub-second per query) and
the absence of approximation removes one source of label noise.

### Outputs (per region)

| Artifact | Path / name |
|---|---|
| FAISS index | `data/processed/faiss_indices/faiss_index.index` |
| Metadata parquet (aligned by row index) | `data/processed/faiss_indices/repo_metadata.parquet` |

In production the per-region files are e.g.
`faiss_index_flatip_cosine_bge_eu_531k.index` and
`repo_metadata_bge_eu_531k.parquet`.

---

## Step 3 — Semantic retrieval per NAICS sector

For every NAICS subindustry, find the most semantically similar
repositories.

### Script
`scripts/03_retrieve_by_naics.py`.

### Notebook (production)
`Colab Notebooks/2_eu_index_naics_match.ipynb` (one per region).

### What it does (detailed walkthrough)

**Step 3.0 — Load the NAICS taxonomy JSON.**

```python
with open('.../naics_titles_by_group_6digit_clean.json') as f:
    industry_dict = json.load(f)
```

After this, `industry_dict` is a Python `dict` with 20 entries: keys are
2-digit NAICS sector codes (`'11'`…`'92'`), values are
`;`-separated strings of subindustry phrases.

**Step 3.1 — Outer loop over sectors.** Iterates 20 times, once per
sector:

```python
for naics_code, subindustry_string in industry_dict.items():
    subindustries = [s.strip() for s in subindustry_string.split(';') if s.strip()]
    num_subindustries = len(subindustries)
```

**Step 3.2 — Compute per-sector `effective_k`.**

```python
effective_k = max(20, math.ceil(400 / num_subindustries)) if num_subindustries <= 20 else 20
```

This implements two regimes:

- **17 sectors with > 20 subindustries** → `effective_k = 20`.
- **3 sectors with ≤ 20 subindustries** get boosted k so that each
  sector retrieves ≈400 candidates in total:
  - 22 Utilities (14 subindustries) → `k = 29`
  - 61 Educational Services (17 subindustries) → `k = 24`
  - 72 Accommodation/Food (15 subindustries) → `k = 27`
- Sector 55 Management has exactly 20 subindustries → `k = 20` (boundary).

**Step 3.3 — Inner loop over subindustry phrases.** For each
subindustry, encode the query, search FAISS, tag the results:

```python
for subindustry in subindustries:
    query = f"Repositories about {subindustry}"
    top_repos = search_repos(query, k=effective_k)
    top_repos["subindustry"] = subindustry
    top_repos["naics_group"] = naics_code
    temp_results.append(top_repos)
```

`search_repos` encodes the query with **BAAI/bge-large-en (v1)** in the
original Colab production (v1.5 in the current pipeline), using the
**asymmetric query prefix** `"Represent this query for retrieval: "`
(notebook cell 28 — distinct from the document prefix of Step 1),
L2-normalizes it, and runs FAISS `IndexFlatIP.search(query_vec,
effective_k)`. Each returned row is annotated with two new columns:

- **`naics_group`** = the 2-digit sector code.
- **`subindustry`** = the specific 6-digit phrase that produced this match.
- **`similarity`** = the cosine score from FAISS (typically 0.64–0.79 in
  the current v1.5 asymmetric setup; the superseded local v1 symmetric
  rerun showed 0.79–0.90).

**Step 3.4 — Concatenate within a sector.** All subindustry-level
DataFrames for one sector are concatenated. A repo that matched several
subindustries in the same sector appears multiple times at this stage.

**Step 3.5 — Concatenate across sectors.** A repository can also
appear under multiple `naics_group` values (e.g., a farm-management tool
could match Agriculture and Administrative Services).

**Step 3.6 — Dedup on `(nwo, naics_group)`.** The local script
`03_retrieve_by_naics.py:99-104` keeps only one row per
`(repository, sector)` pair. After this, the effective count is the
number of distinct `(nwo, naics_code)` candidates.

### Why this design

The retrieval step is a noisy pre-filter, not a classifier. By
querying each subindustry separately, we cast a wider net per sector
than a single per-sector query would — and we expose to GPT-4.1 not
just a (repo, sector) pair but also which specific subindustry phrase
produced the match (the `subindustry` column), useful for later error
analysis. The embedding-only retrieval costs effectively zero per
query, so the cost of running 1,029 queries per region is negligible
compared to GPT-4.1 verification.

### Outputs (per region)

The Colab production parquets store the **raw, pre-dedup** retrieval
(20,879 rows each — "overlap allowed"; the `drop_duplicates` cell is
commented out). Dedup on `(nwo, naics_group)` happens at GPT-prep
(e.g., `1_au_index_metadata_creation.ipynb` cell 52,
`3_eu_repo_gpt_classif.ipynb` cell 20).

| Region | File | Raw rows | Deduped candidates |
|---|---|---:|---:|
| USA | `retrieved_repos_by_naics_group_511k.parquet` | 20,879 | 10,983 |
| EU | `retrieved_repos_by_naics_group_eu_531k.parquet` | 20,879 | 10,531 |
| AU | `retrieved_repos_by_naics_group_au_331k.parquet` | 20,879 | 9,664 |

### ✅ Resolved (2026-06-10, from executed notebook outputs)

- Production used the SAME dynamic k as the local script:
  `effective_k = max(20, ceil(400/n))` (written in the notebooks as
  `max(20, ceil(400/n)) if n <= 20 else 20`, which is equivalent).
  It is NOT a flat k=20: sectors 22, 61, 72 ran k=29/24/27.
- Raw retrieval is therefore exactly Σ n×k = 20,879 per region — the
  identity that lets us verify any rerun's protocol independently of
  its embedding configuration.

---

## Step 4 — GPT-4.1 verification

Score each candidate `(repo, sector)` pair against the structured
rubric.

### Script
`scripts/04_classify_repos.py` → `src/classification/gpt_classifier.py`
and `src/classification/prompt_builder.py`.

### Notebook (production)
`Colab Notebooks/3_eu_repo_gpt_classif.ipynb`. Per slide 9 and per
notebook line `model="gpt-4.1"`, this is the model that produced the
released labels.

### What it does, per candidate row

The production logic lives in the Colab notebook
`3_eu_repo_gpt_classif.ipynb` (cells 18, 22, 26–29). The local script
`scripts/04_classify_repos.py` is similar but **differs in several
production-relevant details** — see the comparison table at the end of
this section. The numbers in the released dataset came from the Colab
production code; describe that one in the paper.

1. **Compose the LLM input text** (Colab cell 18). For every row in the
   retrieval output, the notebook builds a single `input_text` column
   that concatenates three fields with explicit labels:

   ```python
   train_data['input_text'] = (
       'description: ' + train_data['description'].fillna('') +
       ', topics: '    + train_data['topics'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x)).fillna('') +
       ', readme: '    + train_data['readme_content'].fillna('')
   )
   ```

   This combined string — **not the bare README** — is what the
   classifier sends to GPT-4.1 under the `<readme>…</readme>` tag.
   The repository name (`nwo`, `name_repo`) is *not* included.

2. **Preprocess and truncate** (Colab cell 27, `preprocess_readme()`):
   strip Markdown code blocks (` ``` … ``` `), inline code (`` ` … ` ``),
   images (`![…](…)`), links (`[…](…)`), HTML tags, and normalize
   whitespace. Then truncate to **3,000 whitespace-separated tokens**
   (i.e., word-count proxy, not `tiktoken`).

3. **Build the prompt** (Colab cell 22, `build_prompt()`). Structure:
   - **System message:** `"You are a domain expert in economic
     classification systems with a focus on NAICS industry {naics_code}."`
   - **User message** containing:
     - Task header: classify into NAICS Sector `{code}`.
     - The composite text inside `<readme>…</readme>` tags
       (labeled "REPOSITORY README CONTENT" — name is historical).
     - The candidate NAICS sector code and its full semicolon-separated
       subindustry definition, inside `<naics_sector>…</naics_sector>` tags.
     - A four-criterion classification rubric:
       1. Industry-specific software
       2. Sector-relevant functionality
       3. Industry-domain applications
       4. Sector-specific data/research
     - A 1–10 scoring guide (9–10 core; 7–8 strong; 5–6 moderate;
       3–4 weak; 1–2 none).
     - Instruction to reply as a JSON object with keys `match`, `score`,
       `rationale`.

4. **Call the API** (Colab cell 26):
   - Endpoint: `https://api-model-lab.githubcopilot.com/chat/completions`.
   - Model alias `gpt-4.1` resolved to snapshot `gpt-4.1-2025-04-14`.
   - `temperature = 0` (fully deterministic).
   - `max_tokens` dynamically computed: `min(3500, max_allowed - input_tokens)`
     where `max_allowed = 128000` for GPT-4.1. Input tokens are counted
     with `tiktoken.encoding_for_model("gpt-4.1")` (fallback
     `cl100k_base`); this is the **only** place `tiktoken` is used.
   - Retries: up to 5 attempts on any non-200 response, with **linear**
     backoff `time.sleep(5 * (attempt + 1))` (not exponential, not
     restricted to HTTP 429).

5. **Parse the JSON** out of GPT's reply via a regex
   `re.search(r'\{[\s\S]*\}', reply)` followed by `json.loads`. The
   notebook also has a fallback parser for Anthropic-style responses.

6. **Persist the row** to per-batch CSVs under
   `batch_csvs_eu/` (or the equivalent per-region directory) so a
   crashed run can resume.

### Where the local script differs from production

The local `scripts/04_classify_repos.py` was a later re-implementation
and **does not perfectly match** the Colab production run that produced
the released labels:

| Parameter | Colab production (`3_eu_repo_gpt_classif.ipynb`) | Local script (`04_classify_repos.py`) |
|---|---|---|
| Combined input | description + topics + readme | description + topics + readme (matches) |
| README truncation | 3,000 **words** via whitespace split | 3,000 **tokens** via `tiktoken` `cl100k_base` |
| `temperature` | **`0`** | `0.1` |
| `max_tokens` (output) | dynamic, capped at **3,500** | hard-coded **200** |
| Retry policy | 5 attempts, linear backoff `5×attempt`, any non-200 | 5 attempts, exponential backoff `retry_delay × 2**attempt`, only on HTTP 429 |
| System message | present (see step 3 above) | not used |

**The paper must describe the Colab values**, since those are what
produced the released NAICS-GH labels.

### Filter

Only rows with `score >= 8` are retained. (Sector-specific evidence
suggests this is the right threshold — see the per-sector precision
table in the paper.)

### Outputs (per region)

| Region | File | Rows kept (`score≥8`) |
|---|---|---:|
| USA | `usa_2k_gpt_ab8score.parquet` | 2,538 |
| EU | `eu_2k_gpt_ab8score.parquet` | TBD |
| AU | `au_2k_gpt_ab8score.parquet` | TBD |

These three files live in
`/content/drive/MyDrive/naics_repo_classification/2_output/` on
Google Drive. They are **not currently in any local repository**;
exporting them for the public release is the open
reproducibility action item.

### ❓ Open questions

- Confirm `gpt-4.1` is the exact LLM family used and the snapshot
  date / version pinned for production.
- The slides say "GPT-4.1" but the OpenAI Models API has at various
  times labeled this differently; for the paper we should state the
  exact model string.

---

## Step 5 — Per-region merge

### Notebook
`Colab Notebooks/naics_training_gptdata.ipynb` (lines 903–908, verified):

```python
au_data  = pd.read_parquet(f'{DIR}/au_2k_gpt_ab8score.parquet')
eu_data  = pd.read_parquet(f'{DIR}/eu_2k_gpt_ab8score.parquet')
usa_data = pd.read_parquet(f'{DIR}/usa_2k_gpt_ab8score.parquet')
raw_data = pd.concat([au_data, eu_data, usa_data], ignore_index=True)
```

### What it does

Concatenate the three regional outputs row-wise. No transformation, no
dedup at this stage — the three regional files are disjoint by source
pool, so duplicates at this step are essentially zero on `nwo`.

### Output

| File | Path | Rows | Sectors |
|---|---|---:|---:|
| `train_data_naics_github.parquet` | `naics-github-train/data/raw/` | **7,057** | **20** |

**Columns**: `nwo, name_repo, description, topics, readme_content,
code, match, score, rationale, repo_url`.

Sector counts:
- Sector 55 ("Management of Companies and Enterprises"): 13 rows.
- All other 19 sectors: between 85 and 579 rows.

This file is the **canonical released dataset**. It contains the full
provenance — the GPT rationale and score for every row — and includes
sector 55.

---

## Step 6 — Released dataset (`train_data_gpt_ab8_score_with_code.parquet`)

This is the **canonical public release** of NAICS-GH. It is also the
file consumed by the fine-tuning pipeline in `naics-github-train`.
Local copy:
`C:\Users\Alexander\Documents\GitHub\naics-github-train\data\raw\train_data_gpt_ab8_score_with_code.parquet`
(12 MB). Public mirror (currently private; will be set to public for
the paper):
`https://huggingface.co/datasets/aquiro1994/naics-sector-classification`.

### Schema and shape

- **Rows:** **6,588** (no nulls in any column).
- **Columns (6):**

| Column | dtype | Description | Non-null | Unique |
|---|---|---|---:|---:|
| `name_repo` | string | Repository short name (no owner prefix) | 6,588 | 6,141 |
| `description` | string | Repository description from GitHub | 6,588 | 6,231 |
| `topics` | string | Semicolon-joined topic tags (or empty string) | 6,588 | 1,926 |
| `readme_content` | string | Cleaned README text | 6,588 | 6,192 |
| `label` | int64 | Integer encoding 0–18 | 6,588 | 19 |
| `code` | string | 2-digit NAICS sector code | 6,588 | 19 |

The columns `nwo`, `match`, `score`, `rationale`, `repo_url` from
upstream files are **not present**. The dataset deliberately omits
LLM-provenance to keep the public release model-ready.

### Label ↔ code mapping (sorted by label)

| label | code | label | code |
|---:|---|---:|---|
| 0 | 11 | 10 | 53 |
| 1 | 21 | 11 | 54 |
| 2 | 22 | 12 | 56 |
| 3 | 23 | 13 | 61 |
| 4 | 31-33 | 14 | 62 |
| 5 | 42 | 15 | 71 |
| 6 | 44-45 | 16 | 72 |
| 7 | 48-49 | 17 | 81 |
| 8 | 51 | 18 | 92 |
| 9 | 52 | | |

19 classes total — sector 55 ("Management of Companies and Enterprises")
is intentionally absent because only 13 rows passed the score-≥8 filter,
below the 80-row minimum-class-size threshold.

### Per-sector counts

| Code | Sector | Rows |
|---|---|---:|
| 11 | Agriculture, Forestry, Fishing & Hunting | 466 |
| 21 | Mining, Quarrying, Oil & Gas Extraction | 86 |
| 22 | Utilities | 414 |
| 23 | Construction | 82 |
| 31-33 | Manufacturing | 311 |
| 42 | Wholesale Trade | 121 |
| 44-45 | Retail Trade | 641 |
| 48-49 | Transportation & Warehousing | 558 |
| 51 | Information | 389 |
| 52 | Finance & Insurance | 429 |
| 53 | Real Estate, Rental & Leasing | 284 |
| 54 | Professional, Scientific & Technical Services | 372 |
| 56 | Admin & Support and Waste Management | 184 |
| 61 | Educational Services | 331 |
| 62 | Health Care & Social Assistance | 466 |
| 71 | Arts, Entertainment & Recreation | 453 |
| 72 | Accommodation & Food Services | 333 |
| 81 | Other Services (except Public Admin) | 295 |
| 92 | Public Administration | 373 |
| | **Total** | **6,588** |

Smallest class: 23 Construction (82). Largest: 44-45 Retail Trade (641).
Median: ~370 rows/sector.

### Text-length statistics

| Field | min | median | mean | max |
|---|---:|---:|---:|---:|
| `name_repo` (chars) | 2 | 14 | 16 | 85 |
| `description` (chars) | 2 | 65 | 88 | 1,626 |
| `topics` (chars) | 0 | 0 | 21 | 384 |
| `readme_content` (chars) | 3 | 1,912 | 3,387 | 212,122 |

- **4,533 of 6,588 rows have empty `topics`** (≈ 69%). GitHub topic
  tags are user-declared and frequently absent.
- README content varies enormously (3 chars to 212k chars). The
  median is ~1.9k chars; the mean is pulled up by a long tail.
- All four text columns have a *minimum* length of at least 2 chars
  except `topics` (which can be empty).

### Known duplicates

- **447 duplicates on `name_repo` alone** — same short name across
  different owners (e.g., two distinct repos both named `awesome`),
  or the same repo classified into multiple NAICS sectors.
- **113 duplicates on `(name_repo, code)`** — same `(short-name, sector)`
  appears multiple times. These are different repositories with the
  same short name that landed in the same sector; the dataset's
  canonical identifier upstream is `nwo` (owner/repo), but `nwo`
  was dropped from the release. Downstream users joining by
  `name_repo` should be aware.

### Lineage

This file was built (in `naics_training_gptdata.ipynb`) by:

1. Reading three regional GPT-4.1-scored files
   (`{au,eu,usa}_2k_gpt_ab8score.parquet`) from Google Drive.
2. Concatenating them into one DataFrame.
3. Adding `name_repo` by splitting `nwo` on `/`.
4. Renaming `naics_code` → `code`.
5. Dropping rows where `code == '55'` (13 rows).
6. Filling NaN topics with empty string.
7. Sorting by `code`.
8. Saving as `train_data_gpt_ab8_score.csv` (with `readme_content`
   dropped) and as `train_data_gpt_ab8_score_with_code.parquet`
   (with `readme_content` retained, `label` added).

### Public-release status

- **Local file:** ✅ inspected and characterized (this section).
- **Hugging Face dataset at
  `huggingface.co/datasets/aquiro1994/naics-sector-classification`:**
  ❌ **does NOT mirror this file.** Verified 2026-05-23 by downloading
  with a user token. The HF dataset is an older / different artifact:
  - 18,666 rows (14,019 train + 4,647 test) vs our 6,588.
  - 2 columns (`text` XML-wrapped, `label`) vs our 6.
  - 20 sectors (includes sector 55) vs our 19.
  - Label encoding is in a different order (HF label 0 = sector 44-45,
    not 11).
  - Massively skewed toward sector 51 (Information): 6,354/18,666 ≈ 34%.

  The HF page must be **re-uploaded** with the contents of
  `train_data_gpt_ab8_score_with_code.parquet` before submission. The
  current HF page does not represent the dataset described in this
  document or in the paper.

**Planned action:** Upload `train_data_gpt_ab8_score_with_code.parquet`
to a fresh or rewritten HF dataset repo (under either
`aquiro1994/naics-gh` or a similar name). Include the train/val/test
split (4,611 / 659 / 1,318 with `seed=42`) used by the RoBERTa-large
baseline so that downstream users see the same split as the paper.

---

## Step 7 (legacy section) — Training-ready derivative notes

The fine-tuning pipeline in `naics-github-train` expects a slimmer
file with a `label` column and a minimum-class-size filter applied.

### Code reference
`naics-github-train/config.py:129` sets `min_samples_per_class = 80`.
`naics-github-train/src/data_loader.py` (and the
`naics_training_gptdata.ipynb` notebook) applies that filter and
column trim.

### What it does

1. Drop sectors with fewer than 80 rows (eliminates sector 55, which
   has 13 rows).
2. Apply additional quality / dedup filters that brought the total
   from 7,044 (after sector-55 removal) down to 6,588. The exact
   filter set should be verified against the notebook;
   common candidates are:
   - Drop rows with empty/null `readme_content`.
   - Drop rows where the input text is empty after preprocessing.
   - Dedup on `(name_repo, code)` (training file has 113 such
     duplicates after this step).
3. Trim to four text input columns + `code` + `label`.

### Output

| File | Path | Rows | Sectors |
|---|---|---:|---:|
| `train_data_gpt_ab8_score_with_code.parquet` | `naics-github-train/data/raw/` | **6,588** | **19** |

**Columns**: `name_repo, description, topics, readme_content, label,
code`. The `label` is the integer-encoded `code` (0–18).

### ❓ Open questions

- The exact filtering rule between 7,044 and 6,588 needs to be
  re-derived from the notebook; the documentation above lists the
  likely candidates.

---

## Step 7 — Fine-tuned NAICS classifier (downstream benchmark)

This step is the downstream utility demonstration of NAICS-GH: six
pretrained encoders are fine-tuned on the released training corpus and
compared head-to-head. All six runs share identical hyperparameters
and data; only the base model identifier changes.

### Source

- **Notebook:** `Colab Notebooks/naics_training_gptdata.ipynb`
  (canonical; produced the released numbers).
- **Plain-Python mirror:** `Colab Notebooks/naics_training_gptdata.py`.
- **Local refactor:** `naics-github-train/scripts/train.py` (re-runnable
  version of the same pipeline; consumes
  `train_data_gpt_ab8_score_with_code.parquet`).

### Input construction

For each row, the four text fields are concatenated with explicit
labels (`prepare_naics_dataset` → `create_text_input` in the notebook):

```text
Repository: {name_repo} | Description: {description}
  | Topics: {topics} | README: {readme_content}
```

The resulting string is then run through `clean_repository_text()`,
which:

- Strips badges and shields (`![...](url)` and `[![...](...)](...)`).
- Strips license/copyright headers (`MIT License`, `Apache License`,
  `GPL`, `BSD`, `Copyright …`).
- Collapses URLs to the domain only (keeps business-relevant signal).
- Removes Markdown headers (`#{1,6}`) and bold/italic/code markers.
- Replaces fenced code blocks with a `code-{lang}` placeholder; strips
  inline backticks but keeps the content inside.
- Normalizes a small set of tech-stack mentions to canonical forms
  (`js`→`javascript`, `py`→`python`, `reactjs`→`react`,
  `nodejs`→`nodejs`).
- Normalizes excessive punctuation and whitespace; removes
  installation-command noise (`npm install`, `pip install`,
  `git clone`).

The cleaned string is the `text` column passed to the tokenizer.

### Splits

`create_dataset_splits` performs **stratified** train / validation /
test splitting with `random_state = 42`, `test_size = 0.2`,
`val_size = 0.1`:

| Split | Size |
|---|---:|
| Train | 4,611 |
| Validation | 659 |
| Test | 1,318 |
| **Total** | **6,588** |

All 19 NAICS classes appear in each split.

### Common training hyperparameters (all six models)

Set in `setup_training_arguments()`:

| Argument | Value |
|---|---|
| `num_train_epochs` | 8 |
| `learning_rate` | `1.5e-5` |
| `lr_scheduler_type` | `polynomial` |
| `warmup_ratio` | 0.15 |
| `weight_decay` | 0.02 |
| `per_device_train_batch_size` | 8 |
| `per_device_eval_batch_size` | 16 |
| `gradient_accumulation_steps` | 2 |
| Effective batch size | 16 |
| `max_grad_norm` | 1.0 |
| `bf16` | True |
| `optim` | `adamw_torch_fused` |
| `eval_strategy` / `save_strategy` | `steps` (every 100) |
| `save_total_limit` | 5 |
| `load_best_model_at_end` | True |
| `metric_for_best_model` | `f1` |
| `early_stopping_patience` | 2 |
| `early_stopping_threshold` | 0.001 |
| `seed` | 42 |
| Tokenizer `max_length` | **512** WordPiece tokens (uniform across all 6 models) |
| Evaluation metrics | weighted F1, accuracy, precision, recall |

### Six base models trained

| HF identifier | Local label | Params |
|---|---|---:|
| `roberta-base` | RoBERTa-base | 125M |
| `roberta-large` | RoBERTa-large | 355M |
| `answerdotai/ModernBERT-base` | ModernBERT-base | 139M |
| `answerdotai/ModernBERT-large` | ModernBERT-large | 395M |
| `microsoft/deberta-v3-base` | DeBERTa-v3-base | 183M |
| `microsoft/deberta-v3-large` | DeBERTa-v3-large | 400M |

### Test-set results

| Model | Params | Test F1 | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| RoBERTa-base | 125M | 84.26% | 84.84% | 84.70% | 84.84% |
| **RoBERTa-large** | **355M** | **86.45%** | **86.35%** | **86.68%** | **86.35%** |
| ModernBERT-base | 139M | 84.63% | 84.85% | 84.86% | 84.85% |
| ModernBERT-large | 395M | 84.16% | 84.23% | 84.44% | 84.23% |
| DeBERTa-v3-base | 183M | 85.68% | 85.82% | 86.16% | 85.82% |
| DeBERTa-v3-large | 400M | 85.07% | 85.22% | 85.41% | 85.22% |

All metrics are weighted across the 19 classes on the held-out
1,318-row test set.

### Findings

- **RoBERTa-large is the strongest** baseline at F1 = 86.45%.
- **Encoder family matters more than parameter count** at this
  corpus size. DeBERTa-v3-base (183M) outperforms ModernBERT-large
  (395M), suggesting that the original RoBERTa/DeBERTa pretraining
  objectives transfer better to short-document classification than
  ModernBERT's long-context-focused pretraining, at least at
  $\sim$5k training examples.
- **The large variants do not uniformly beat the base variants.**
  ModernBERT-large is worse than ModernBERT-base, and DeBERTa-v3-large
  is worse than DeBERTa-v3-base. RoBERTa is the only family where
  scaling up improves performance. This is consistent with a small
  training corpus (4.6k examples, 19 classes) where larger models
  risk overfitting.

Published RoBERTa-large checkpoint:
`huggingface.co/alexanderquispe/naics-github-classifier`.

---

## Validation against the human-labeled gold set

This section documents Phase 1 validation work that backs the paper's
§5. It is not a pipeline step in itself --- it is an evaluation of the
output of the data construction pipeline.

### Gold set

- **File:** `data/raw/gold_set_manual_check.xlsx` (local; downloaded from
  Alexander's Google Sheet).
- **Size:** 2,421 rows.
- **Columns:** `manual_check, nwo, name_repo, description, topics,
  readme_content, code, match, sector_name, score, rationale, repo_url`.
- **Sampling frame:** the gold set is a **~95.4% random sample of
  `usa_train_data_gpt_ab8score.parquet`** (2,421 of 2,538 USA GPT-4.1
  rows). Verified empirically:
  - 100% of gold-set `nwo` values are in
    `usa_train_data_gpt_ab8score.parquet`.
  - 99.75% of gold-set `nwo` values are unambiguously in the USA
    regional source pool (`repos_usa_usa_511k.parquet`); the
    remaining 6 rows have `nwo` strings that happen to appear in
    multiple regional pools and are not necessarily EU/AU.
  - Sector 55 is absent in the gold set (consistent with the USA GPT
    file, which also lacks sector 55).
  - The gold-to-USA-GPT score ratio is uniform across scores
    (0.952 / 0.954 / 0.959 for scores 8 / 9 / 10), i.e., the gold
    set is essentially a uniform random subsample of the USA GPT
    output.
- **Scope:** validation covers only the **USA portion of the labeled
  corpus**. EU and AU repositories were labeled by the same GPT-4.1
  pipeline but have not yet been included in any human gold sample;
  per-region validation for those regions is pending future RA work.
- **Annotation protocol:** research assistants reviewed each repo's
  GitHub page and judged whether the LLM-assigned NAICS sector is
  correct (`manual_check = "Correct"`) or incorrect
  (`"Incorrect"`). Single-labeler per row; no inter-annotator
  agreement (Cohen's κ) was computed.

### Scripts (all in `paper/validation/`)

| Script | Purpose |
|---|---|
| `compute_validation_stats.py` | Overall precision + per-sector + per-score tables; writes the CSVs under `paper/validation/output/`. |
| `check_sample_frame.py` | Joins the gold set against the source pools and against the released files to verify sample frame, dedup behavior, and stratification. |
| `generate_tex_tables.py` | Renders `per_sector_precision_table.tex` and `sector_counts_table.tex` for the paper. |

### Headline metric

**GPT-4.1 label precision on USA repositories: 96.98%**
(2,348 / 2,421 gold rows judged correct; 95% Wilson confidence
interval $[96.23\%, 97.59\%]$).

This number is a clean GPT-4.1 measurement on USA-pool repositories.
It is *not* an estimate of label precision for the EU or AU portions
of the corpus, which have not yet been included in a human gold
sample. Per-region validation for EU and AU is pending future RA
work.

### Per-sector precision (Wilson 95% CIs)

Source: `paper/validation/output/per_sector_precision.csv`. Six
sectors reach 100% precision (21 Mining, 48--49 Transportation, 51
Information, 53 Real Estate, 71 Arts/Entertainment, 92 Public
Administration). Two sectors are notably weak:

- **31--33 Manufacturing: 73.0%** (73/100; CI [63.6, 80.7]).
- **42 Wholesale Trade: 73.0%** (27/37; CI [57.0, 84.6]).

Most other sectors fall in the 93--99% range.

### Precision rises monotonically with LLM score

Source: `paper/validation/output/precision_by_score.csv`.

| Score | n | Correct | Precision | 95% CI |
|---:|---:|---:|---:|---|
| 8 | 314 | 285 | 90.76% | [87.05, 93.49] |
| 9 | 1,965 | 1,922 | 97.81% | [97.07, 98.37] |
| 10 | 142 | 141 | 99.30% | [96.12, 99.88] |

This is strong evidence that the LLM's 1--10 score is a meaningful
confidence signal --- raising the threshold from $\geq 8$ to $\geq 9$
removes most remaining errors at the cost of ~13% of rows.

### Sample-frame audit

Source: `paper/validation/output/sector_distribution_gold_vs_train.csv`.

- 100% of gold-set $(nwo, code)$ pairs exist in
  `train_data_naics_github.parquet`.
- 88.97% of gold-set $(name\_repo, code)$ pairs exist in
  `train_data_gpt_ab8_score_with_code.parquet` (the gap is partly
  explained by `name_repo` non-uniqueness and partly by the two files
  reflecting different pipeline snapshots).
- Sector stratification is comparable between the gold set and the
  merged file: all per-sector percentages within $\pm 2$ pp of the
  corpus distribution, with the largest skew at Sector 92 (gold
  +1.94 pp).

### Outputs

All under `paper/validation/output/`:

- `overall_precision.csv`
- `per_sector_precision.csv` (+ `per_sector_precision_table.tex`)
- `precision_by_score.csv`
- `sector_score_counts.csv`, `sector_score_correct.csv`, `sector_score_precision.csv`
- `sector_distribution_gold_vs_train.csv`
- `sector_counts_table.tex` (now from the v1 6,588-row file)

### Open items for validation

1. **Validation for EU and AU.** The current 96.98% number applies
   only to the USA-GPT-4.1 portion. Draw separate gold samples from
   each region (suggest ~500 rows per region) and report per-region
   precision. A combined cross-region precision will require a
   stratified pooled estimate.
2. **Inter-annotator agreement.** Single-labeler protocol means no
   κ. For the camera-ready paper, consider double-labeling at least
   ~200 rows so that a κ can be reported alongside the headline
   precision.
3. **Qualitative error analysis** of the 73 incorrect rows --- which
   NAICS sector did the RA say the repo *should* belong to? Pending
   from the RA team.

---

## Reproducibility summary

To reproduce the released dataset from scratch you need:

1. **Access to the source warehouse query.** The raw extraction
   depends on GitHub's internal `hive.canonical.repositories_current`
   and `hive.suez.readme_current` tables (see §Step 0 SQL). External
   researchers reproducing this work would need to substitute an
   equivalent public source (e.g., GH Archive or a recent BigQuery
   GitHub dump) — the filter clauses transfer with obvious mapping
   (`num_stars`, `is_fork`, `is_public`, README size).
2. The three regional repository dumps as derived from the SQL +
   jurisdiction split (currently in `data/raw/`).
3. The NAICS taxonomy file `naics_titles_by_group_6digit_clean.json`.
4. The pipeline code in this repository.
5. GPU access (the BGE step benefits substantially from one).
6. An API endpoint serving `gpt-4.1` (either OpenAI directly or, as
   used in production, the GitHub Copilot LLM-lab endpoint).
7. The exact prompt at
   `src/classification/prompt_builder.py:build_classification_prompt`.

For partial reproducibility starting from intermediate outputs:

- Skip the warehouse SQL and start from the three regional
  `repos_{usa_usa,eu,au}_*k.parquet` files in `data/raw/`.
- Skip Steps 1–3 and start from the three regional
  `*_2k_gpt_ab8score.parquet` files (currently in Google Drive).
  Concatenate them as in Step 5.
- Skip Steps 1–4 entirely and start from
  `train_data_naics_github.parquet` (the released file).

---

## What is *not* in the released v1.0

- The three regional GPT-validated files (`*_2k_gpt_ab8score.parquet`)
  — they live only on Google Drive at the moment.
- The raw 1.4M-row regional dumps before retrieval — currently in
  `data/raw/` but not part of the public release.
- The FAISS indices — easy to regenerate from Step 1's output.
- The intermediate `retrieved_repos_by_naics_group_*.parquet` files —
  not currently in the local repo for all regions; the EU one is
  referenced in the notebook.

---

## Production parameter summary (one table to rule them all)

| Parameter | Value | Where |
|---|---|---|
| Embedding model (original production) | `BAAI/bge-large-en` (v1) — **corrected 2026-06-10** | `1_eu/1_au_index_metadata_creation.ipynb` cells 12 + 28 (executed) |
| Embedding model (current pipeline) | `BAAI/bge-large-en-v1.5` (deliberate upgrade) | `src/embeddings/bge_encoder.py` |
| Embedding input (original production) | README-only, first 1,000 chars (`readme_clean`) | notebook cell 11 (executed) |
| Embedding input (current pipeline) | combined `description + topics + readme` (deliberate upgrade) | `preprocessor.py:create_input_text` |
| Document prefix | `"Represent this document for retrieval: "` | notebook cell 14; `bge_encoder.py` |
| Query prefix | `"Represent this query for retrieval: "` (asymmetric) | notebook cell 28; `bge_encoder.py` |
| Embedding dim | 1024 | (BGE-large architecture) |
| Embedding batch size | 1024 | `config/config.yaml:14` |
| Embedding precision | FP16 on GPU | `bge_encoder.py` |
| README cleaning (current pipeline) | strip code blocks, HTML, URLs; collapse whitespace | `preprocessor.py:9-40` |
| README truncation (for embedding) | 1,000 chars | notebook cell 11; `preprocessor.py:43` |
| FAISS index type | `IndexFlatIP` | `faiss_indexer.py:45` |
| Retrieval query template | `"Repositories about {subindustry}"` | `03_retrieve_by_naics.py:91` |
| Retrieval k (per subindustry) | 20 for 17 sectors; 24/27/29 for 3 small sectors | dynamic-k formula in `2_eu_index_naics_match.ipynb` cell 17 |
| Number of subindustries | 1,029 across 20 sectors (exact) | `naics_titles_by_group_6digit_clean.json` |
| LLM model (alias) | `gpt-4.1` | EU GPT notebook cell 26 |
| LLM model (resolved snapshot) | `gpt-4.1-2025-04-14` | per `/models` listing in cell 25 of same notebook |
| LLM endpoint | `api-model-lab.githubcopilot.com/chat/completions` | `3_eu_repo_gpt_classif.ipynb` cell 26 |
| LLM system message | `"You are a domain expert in economic classification systems with a focus on NAICS industry {code}."` | cell 26 |
| LLM input field | combined `"description: …, topics: …, readme: …"` | cell 18 |
| LLM input truncation | 3,000 **words** via whitespace split | cell 27 `preprocess_readme()` |
| LLM temperature | **0** (production); `0.1` in local script | cell 26 |
| LLM max\_tokens (output) | dynamic, capped at **3,500** (production); `200` in local script | cell 26 |
| LLM retry policy | 5 attempts, linear backoff `5×attempt`, any non-200 (production) | cell 26 |
| Score keep threshold | ≥ 8 | `05_filter_results.py:36` |
| Min samples per class (training) | 80 | `naics-github-train/config.py:129` |
| Fine-tune model | `roberta-large` | `naics-github-train/README.md:9` |
| Train/Val/Test | 70/10/20 (seed=42) | `naics-github-train/config.py:104-108` |
| Fine-tune learning rate | 1.5e-5 | `config.py:80` |
| Fine-tune batch size | 32 | `config.py:78` |
| Fine-tune epochs | 8 | `config.py:77` |
| Fine-tune mixed precision | BF16 | `config.py:100` |

---

## Open items before submission

The paper's Source-Pool description (P0-7 in `paper/IMPROVEMENTS.md`)
needs the answers to these questions. They are the gating items for a
clean reproducibility story.

### ✅ Answered (after reading `1_scripts/` notebooks)

1. **Snapshot date.** US data freshness is **April 15, 2025**
   (filename of `us_repos_presample_no_readmes_2025_04_15.ipynb`); the
   EU readme extraction was saved 2025-08-15 (`repos_eu_08_15_2025.parquet`).
   Final regional parquets sit on top of those snapshots.
2. **Pre-filter rules.** Fully documented in the canonical SQL above:
   `is_fork=FALSE`, `is_public=TRUE`, `NOT is_spammy_owner`,
   `num_stars>=1`, `has_readme=TRUE`, README path/filename
   `LOWER(...)='readme.md'`, `content_size>=750`, `disk_usage_bytes>0`,
   non-empty description, `HAVING SUM(commit_count)>5`, and
   `HAVING COUNT(DISTINCT user_id)>=2` (for AU; need to verify whether
   the contributors HAVING was also applied to USA and EU).
3. **Jurisdiction rule.** `accounts_current.country_account` on the
   **owner account**, joined via
   `repositories_current.owner_dotcom_id = accounts_current.dotcom_id`.
   USA = `'US'`; EU = 27 explicit country codes (EU-27 minus UK);
   AU = `'AU'`.
6. **k in retrieval.** ✅ **Fully resolved (2026-06-10)**: all three
   regions used `effective_k = max(20, ceil(400/n))` (executed loop
   cells in `2_eu_index_naics_match.ipynb` cell 17,
   `1_eu_index_metadata_creation.ipynb` cell 38,
   `1_au_index_metadata_creation.ipynb` cell 38), yielding exactly
   20,879 raw rows per region.
8. **`>=2` contributors filter.** **Confirmed applied** — in the AU
   SQL it appears in the HAVING clause; in the US pandas filter chain
   it appears as `df['num_commit_contributors'] >= 2`. Need to verify
   whether USA and EU SQL versions also use the HAVING form.

### ❓ Still open

4. ~~**`bge-large-en` version.**~~ ✅ **CORRECTED 2026-06-10: production
   used `BAAI/bge-large-en` (v1), NOT v1.5.** The executed Colab cells
   (`1_eu_index_metadata_creation.ipynb` cells 12 + 28,
   `1_au_index_metadata_creation.ipynb` cells 12 + 28) explicitly load
   `SentenceTransformer('BAAI/bge-large-en')`. The earlier "confirmed
   v1.5" note (2026-05-23, from slide 7) was based on the slide, which
   the executed code contradicts. The current replication pipeline uses
   v1.5 as a documented, deliberate upgrade (see funnel section).
5. ~~**`gpt-4.1` snapshot.**~~ ✅ **Resolved to `gpt-4.1-2025-04-14`**
   (per Alexander, 2026-05-23). The GitHub Copilot LLM-lab `/models`
   endpoint (queried in `3_eu_repo_gpt_classif.ipynb` cell 25) lists
   `gpt-4.1` as an alias alongside the pinned snapshot
   `gpt-4.1-2025-04-14` and a `gpt-4.1-kvcache` variant. Production
   used `model="gpt-4.1"`, which the endpoint resolves to
   `gpt-4.1-2025-04-14`. The paper should cite the pinned snapshot
   string for reproducibility.
7. **7,044 → 6,588 quality filter.** Going from the merged
   `train_data_naics_github.parquet` (7,057 rows minus 13 from
   sector 55) to the training-ready
   `train_data_gpt_ab8_score_with_code.parquet` (6,588 rows) drops
   456 additional rows. Need to derive the exact filter rule from
   `naics_training_gptdata.ipynb`.
9. **`other_lang_md == 1` filter.** Visible in the pandas filter
   chain — appears to be an English-language README flag. Confirm
   whether it was applied to the final regional parquets or was just
   an exploratory column.
10. **Uniformity of SQL across regions.** The AU cell (cell 24) has
    `HAVING ... >= 2 contributors`; the US-only cell (cell 4 / cell 9)
    has only `HAVING SUM(commit_count) > 5` without the contributors
    HAVING. Either the contributors check was added in pandas for the
    US case, or the US SQL was re-run with the HAVING added. Need to
    verify so the paper can say "the same SQL was used for all three
    regions" or describe the per-region differences accurately.
