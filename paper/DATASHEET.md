# Datasheet for NAICS-GH

A "Datasheet for Datasets" following [Gebru et al. (2021)](https://dl.acm.org/doi/10.1145/3458723).
This document accompanies the NAICS-GH dataset (Industry Classification of
GitHub Repositories Using the North American Industry Classification
System).

- **Dataset version:** v1.0
- **Last updated:** 2026-05-22
- **Authors:** Kevin Xu (GitHub), Alexander Quispe (GitHub)
- **Contact:** alexanderquispe@github.com (corresponding)
- **License:** CC-BY-4.0 (labels and metadata); MIT (pipeline code)

---

## 1. Motivation

**For what purpose was the dataset created?**
NAICS-GH was created to enable empirical work on the industrial
composition of open-source software production. GitHub hosts millions
of public repositories but provides no native indication of which
*industry* a repository serves. NAICS-GH maps a representative subset
of repositories from the USA, the European Union, and Australia onto
the 2-digit North American Industry Classification System (NAICS),
which is the industry-classification standard used by US, Canadian, and
Mexican statistical agencies. The dataset is also the training data for
a downstream RoBERTa-large classifier that propagates these labels to
arbitrary repositories at inference time.

**Who created the dataset and on behalf of which entity?**
The dataset was created by Kevin Xu and Alexander Quispe at GitHub.

**Who funded the creation of the dataset?**
GitHub. Compute for LLM inference was paid through GitHub's internal
LLM access; no external grants were used.

**Any other comments?**
The dataset is one half of a two-repository system. This file
documents the labeled corpus; the downstream classifier and its
training code live in a sibling repository, `naics-github-train`.

---

## 2. Composition

**What do the instances represent?**
Each instance is a public GitHub repository, identified by its
`owner/name` (`nwo`) string. The labels attached to each instance
indicate which 2-digit NAICS industry sector the repository serves,
along with the LLM-generated rationale and confidence score.

**How many instances are there in total?**
The released corpus contains **6,588 repositories** in the parquet
file `train_data_gpt_ab8_score_with_code.parquet`. This is the file
that produces the published RoBERTa-large baseline.

**Does the dataset contain all possible instances, or is it a sample
from a larger set?**
It is a sample. The pipeline begins with three regional dumps of
public GitHub repositories (totaling 1,372,489 rows):
- USA: 510,380
- European Union: 530,898
- Australia: 331,211

For each NAICS subindustry (1,029 across the 20 top-level sectors),
the top-k most semantically similar repositories were retrieved using
BGE embeddings + FAISS (k = max(20, ceil(400/n)) for a sector with n
subindustries; k = 20 for 17 of 20 sectors), then LLM-scored. Only
repositories scoring at least 8 on the 1-10 rubric were retained.

**What data does each instance consist of?**
Each instance has six columns in the released file
(`train_data_gpt_ab8_score_with_code.parquet`):

| Column | Type | Description |
|---|---|---|
| `name_repo` | string | Repository short name (no owner prefix) |
| `description` | string | Repository description from GitHub |
| `topics` | string | Semicolon-joined topic tags; empty string if none |
| `readme_content` | string | Cleaned README content |
| `label` | int64 | Integer class encoding 0â€“18 |
| `code` | string | 2-digit NAICS sector code |

The columns `nwo`, `match`, `score`, `rationale`, and `repo_url`
present in intermediate files are intentionally dropped from the
public release so that the corpus is model-ready (no LLM-provenance,
no PII via repo-owner URLs).

**Is there a label or target associated with each instance?**
Yes. The label is the 2-digit NAICS sector code in the `code` column
(string form), or equivalently the integer `label` column (0â€“18,
monotonic in NAICS code order).

**Is any information missing from individual instances?**
- All four text columns and both label columns are fully populated
  (zero NaN values).
- `topics` is empty for ~69% of rows (4,533 of 6,588): not all GitHub
  repositories declare topic tags.
- `readme_content` is preprocessed (code blocks stripped, HTML
  removed, URLs collapsed) and truncated to 3,000 whitespace-separated
  tokens before LLM scoring; the released field reflects the
  *preprocessed* text, not the original.

**Are there recommended data splits?**
For the RoBERTa-large baseline we used 70% train / 10% validation /
20% test ($n = 4{,}611 / 659 / 1{,}318$) with `seed=42`. Splitting was
random and stratified by NAICS sector. Practitioners building their
own classifiers are free to re-split.

**Are there any errors, sources of noise, or redundancies?**
- **Label noise (USA scope only).** Human re-validation of 2,421
  USA repositories found 96.98% GPT-4.1 label precision overall, but
  two sectors (31â€“33 Manufacturing and 42 Wholesale Trade) had only
  ~73% precision at the score â‰¥ 8 threshold. EU and AU portions are
  GPT-4.1-labeled by the same pipeline but have not yet been included
  in a manual gold sample.
- **Duplicate `name_repo`.** Repository short names are not globally
  unique; the released file has 447 rows sharing a `name_repo` with
  another row, of which 113 share both `(name_repo, code)`. These are
  different repositories with identical short names that independently
  passed verification. The upstream intermediate files retain `nwo`
  as a globally unique identifier.
- **READMEs.** Some READMEs are predominantly HTML badges, build
  configs, or templates that carry little semantic signal. The
  preprocessing step removes most of these artifacts but cannot
  recover content that wasn't there.

**Is the dataset self-contained, or does it link to or otherwise rely
on external resources?**
The released parquet is self-contained. The pipeline that generated it
relies on (a) BAAI/bge-large-en (released model, Hugging Face),
(b) FAISS (open source), and (c) GPT-4.1 (snapshot
`gpt-4.1-2025-04-14`) accessed via the GitHub Copilot LLM-lab
endpoint. The repository content (READMEs, descriptions) is preserved
in the file at the time of capture; we do not re-fetch live GitHub
content.

**Does the dataset contain data that might be considered confidential
or that includes content protected by attorney-client privilege or
similar?**
No. All repositories are public on GitHub.

**Does the dataset contain data that, if viewed directly, might be
offensive, insulting, threatening, or might otherwise cause anxiety?**
Possibly, since READMEs contain free-form user-submitted text. We do
not apply explicit content filtering. Anyone re-using the README
content should be aware that public repository content can include
offensive language.

**Does the dataset identify any subpopulations (e.g., by age, gender)?**
No. The dataset captures organizational and industrial-economic
attributes of repositories, not demographic attributes of contributors.

**Is it possible to identify individuals from the data?**
Repository owner names (`nwo` prefix) may identify individual GitHub
account holders, since public repositories often live under personal
accounts. This is the same level of identifiability already present
on GitHub itself; the dataset does not add information beyond what
GitHub publishes.

**Does the dataset contain data that might be considered sensitive in
any way?**
The dataset records that a given GitHub user's public repository
serves a given industry sector. We do not consider this sensitive
beyond the existing public nature of GitHub, but downstream uses
should respect the principle that aggregating labels about individuals
can amplify identifiability.

---

## 3. Collection Process

**How was the data associated with each instance acquired?**
Source repositories were extracted from GitHub's internal Trino-on-Hive
data warehouse via a Presto/Trino SQL query against the
`hive.canonical.repositories_current`, `hive.canonical.accounts_current`,
`hive.suez.readme_current`, and
`delta.snapshots.github_collab_commit_contributions` tables. The
query yielded three regional dumps (USA: 510,380; EU: 530,898;
AU: 331,211 â€” totaling 1,372,489 repositories). Each repository's
NAICS label was then *derived* via a two-stage pipeline (see
Section 4). The label is not directly observed but is the output of
an algorithm that combines semantic retrieval with an LLM-scored
rubric.

**What mechanisms or procedures were used to collect the data?**
- The source SQL filters on `is_public = TRUE`, `is_fork = FALSE`,
  `NOT is_spammy_owner`, `num_stars >= 1`, README size â‰¥ 750 bytes,
  â‰¥ 6 commits, â‰¥ 2 distinct commit contributors, and non-empty
  description. Jurisdiction is assigned by the owner account's
  `country_account` field.
- Embeddings were computed with `BAAI/bge-large-en` (1024-dim,
  L2-normalized, FP16 on GPU).
- FAISS `IndexFlatIP` was used for exact inner-product search,
  equivalent to cosine similarity on the normalized vectors.
- LLM scoring was issued through the GitHub Copilot LLM-lab endpoint
  (`api-model-lab.githubcopilot.com/chat/completions`) using GPT-4.1
  (alias resolved to snapshot `gpt-4.1-2025-04-14`), with
  `temperature = 0` and dynamic `max_tokens` capped at 3,500.

**If the dataset is a sample from a larger set, what was the sampling
strategy?**
For each of the 1,029 NAICS subindustry phrases across 20 sectors,
the top-*k* nearest-neighbor repositories under BGE cosine similarity
were retrieved per region. The default *k* = 20; for the three
sectors with fewer than 20 subindustries the formula
`max(20, ceil(400/n))` boosts *k* to 24/27/29 so each sector retrieves
roughly 400 candidates. After deduplication and GPT-4.1 filtering at
score â‰¥ 8, we kept all surviving repositories. The sampling strategy
is therefore "top-k semantic retrieval per industry query," not
uniform random sampling.

**Who was involved in the data collection process?**
The pipeline was designed and operated by the authors. Research
assistants conducted the manual gold-set re-check (n=2,421 USA
repositories). Each row was reviewed by exactly one annotator; no
inter-annotator agreement coefficient was computed.

**Over what timeframe was the data collected?**
The USA source dump was extracted on April 15, 2025; the EU README
extraction snapshot is dated August 15, 2025. LLM scoring was
conducted between mid-2025 and early-2026. The dataset v1.0 was
finalized in early 2026.

**Were any ethical review processes conducted?**
No formal IRB review, since the dataset uses only public data and
contains no human subjects. The release was approved internally at
GitHub.

**Did you collect the data directly from the individuals in question,
or obtain it via third parties or other sources?**
The data was derived from publicly available GitHub content. We did
not contact repository owners individually.

**Were the individuals in question notified about the data collection?**
No. The data is public; we considered notification infeasible and
unnecessary at the scale of millions of source repositories.

**Did the individuals in question consent to the collection and use of
their data?**
GitHub users consent to public visibility of their public repositories
under GitHub's Terms of Service. They do not specifically consent to
industry-sector labeling, but the labels are derived computationally
from public content.

**If consent was obtained, were the consenting individuals provided
with a mechanism to revoke their consent in the future?**
We will provide a takedown mechanism: any repository owner can request
removal of their repository from the released dataset by emailing the
corresponding author. We will issue a v1.x with the requested removal
and document the change.

---

## 4. Preprocessing, Cleaning, Labeling

**Was any preprocessing/cleaning/labeling of the data done?**
Yes, extensively. The pipeline is:

1. **README preprocessing.** For embedding input, the *raw* README is
   truncated to its first 1,000 characters (no markup stripping at
   this stage). For LLM scoring, code blocks, HTML tags, badge URLs,
   and image markdown are stripped and the text is truncated to 3,000
   whitespace-separated tokens (word-count proxy, not BPE tokens).
2. **Composite text construction.** LLM scoring uses the combined
   string `"description: {desc}, topics: {topics}, readme: {readme}"`,
   constructed at row time and inserted into the `<readme>â€¦</readme>`
   block of the prompt. Embedding uses the truncated README only;
   description and topics enter the pipeline at the scoring stage.
3. **Embedding.** BGE-large-en with the asymmetric BGE prefixes:
   `"Represent this document for retrieval: "` for documents and
   `"Represent this query for retrieval: "` for queries.
4. **Retrieval.** For each NAICS subindustry, the query
   `"Repositories about {subindustry}"` is embedded and the top-*k*
   nearest repositories are retrieved (default *k* = 20; the three
   sectors with fewer than 20 subindustries use a boosted
   *k* = `max(20, ceil(400/n))` = 24, 27, or 29).
5. **LLM scoring.** Each (repository, sector) candidate is presented
   to GPT-4.1 with the rubric prompt (see the paper appendix). The
   model returns a JSON object nested under the outer key
   `"NAICS {code}"`, with string-valued `match` (`"Yes"`/`"No"`),
   string-valued `score` (`"1"`-`"10"`), and free-form `rationale`.
   Downstream code coerces `match` to a boolean and `score` to an
   integer.
6. **Filtering.** Only repositories scoring at least 8 are retained.
7. **Per-region merge.** The three regional outputs are concatenated.
8. **Minimum-class filter (training derivative only).** Sectors with
   fewer than 80 repositories are dropped, removing sector 55.

**Was the raw data saved in addition to the preprocessed data?**
Yes. The original parquets of unfiltered GitHub repositories were
preserved (USA, EU, AU dumps), as well as the three intermediate
per-region GPT outputs (`{usa,eu,au}_2k_gpt_ab8score.parquet`). These
are not in v1.0 of the public release but will be added to Zenodo at
camera-ready.

**Is the software used to preprocess/clean/label the data available?**
Yes, in this repository (MIT License). The relevant entry points are
`scripts/01_generate_embeddings.py` through `scripts/05_filter_results.py`,
with prompts in `src/classification/prompt_builder.py`.

---

## 5. Uses

**Has the dataset been used for any tasks already?**
Yes:
- Fine-tuning six pretrained encoders (RoBERTa, ModernBERT, and
  DeBERTa-v3 in base and large variants) on the released corpus.
  RoBERTa-large is strongest at 86.45% F1 / 86.35% accuracy on a
  held-out 20% test set; the fine-tuned checkpoint is available on
  Hugging Face at `alexanderquispe/naics-github-classifier`.
- Internal industry-mix analyses of open-source production.

**Is there a repository that links to any or all papers or systems that
use the dataset?**
At release time, the paper repository will track downstream uses. The
released Hugging Face dataset card will also link to known references.

**What (other) tasks could the dataset be used for?**
- Studying the industrial composition of open-source contributions
  across regions or over time.
- Building classifiers that map *new* repositories to NAICS sectors.
- Measuring AI-coding-assistant adoption by industry (the downstream
  classifier is already being used for this).
- Comparative open-source ecosystem analysis vs. economic
  classification statistics.

**Is there anything about the composition of the dataset or the way it
was collected and preprocessed that might impact future uses?**
- **NAICS is a North American taxonomy.** Applying it to EU and AU
  repositories assumes mapping consistency that may not hold for some
  sectors (notably Sector 22, "Utilities," which has US-specific
  regulatory structure).
- **English-only embeddings.** Repositories with non-English READMEs
  are under-represented.
- **Score-conditional reliability.** Manufacturing and Wholesale Trade
  labels are only ~73% precise at score = 8 and should be filtered at
  score â‰¥ 9 for high-precision applications.

**Are there tasks for which the dataset should not be used?**
- **Targeting individual contributors.** The dataset is intended for
  aggregate analysis; using it to profile individual developers is
  outside its intended use.
- **Inferring repository quality, popularity, or value.** The dataset
  encodes industry classification, not quality.
- **Treating labels as ground truth without conditioning on score.**
  Two sectors have substantially higher error rates, and the LLM
  score should be used as a confidence signal.

---

## 6. Distribution

**Will the dataset be distributed to third parties outside of the
entity on behalf of which the dataset was created?**
Yes. The dataset is publicly released.

**How will the dataset be distributed?**
- **Hugging Face Datasets:**
  [`aquiro1994/naics-gh`](https://huggingface.co/datasets/aquiro1994/naics-gh)
  â€” public, CC-BY-4.0, with stratified train/validation/test splits
  ready for `datasets.load_dataset`. Croissant metadata generated
  automatically by HF.
- **Zenodo:** DOI-citable archive (DOI to be reserved at release).
- **GitHub:** Code, prompts, and reproduction scripts at the two
  repositories listed in Â§1.

**When will the dataset be distributed?**
At paper acceptance / camera-ready, expected late 2026 or 2027.

**Will the dataset be distributed under a copyright or other
intellectual-property license?**
Yes:
- Labels and metadata: **CC-BY-4.0**.
- Pipeline code: **MIT**.
- Repository content (READMEs and descriptions) excerpted in the
  dataset remains governed by each repository's own license; we
  preserve `spdx_license` in the released file so users can filter
  by license type.

**Have any third parties imposed IP-based or other restrictions on the
data associated with the instances?**
Each source repository's content is subject to its own license. The
dataset preserves SPDX license identifiers so users can comply with
upstream license terms (e.g., excluding non-commercial-licensed
repositories from commercial uses).

**Do any export controls or other regulatory restrictions apply to the
dataset?**
None known.

---

## 7. Maintenance

**Who will be supporting/hosting/maintaining the dataset?**
Alexander Quispe (corresponding author) and Kevin Xu, with GitHub's
support.

**How can the owner/curator/manager be contacted?**
alexanderquispe@github.com.

**Is there an erratum?**
A `CHANGELOG.md` will track all post-release errata at the Hugging
Face dataset page and in the GitHub paper repository.

**Will the dataset be updated?**
- **v1.x** patches for take-down requests, typo fixes, and metadata
  corrections.
- **v2.0** planned to add: India (1M repos), finer-grained 3- to
  6-digit NAICS labels, and multilingual README support. Timeline to
  be confirmed.

**If the dataset relates to people, are there applicable limits on the
retention of the data associated with the instances?**
We retain the labels indefinitely subject to take-down requests.

**Will older versions of the dataset continue to be supported/hosted/maintained?**
Yes, on Zenodo. Each tagged version has its own DOI; older versions
remain citable but will not receive further updates.

**If others want to extend/augment/build on/contribute to the dataset,
is there a mechanism for them to do so?**
Yes. Pull requests to the GitHub paper repository, issues on the
Hugging Face dataset page, and email to the corresponding author are
all accepted. Substantive contributions (e.g., new regions,
multilingual labels) will be co-credited in the next dataset release.
