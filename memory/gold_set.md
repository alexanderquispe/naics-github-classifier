---
name: gold-set
description: Human-validated gold set for the NAICS-GH paper — 2,422 manually checked observations with a correct/incorrect column.
metadata:
  type: project
---

Confirmed 2026-05-22 by Alexander:

- **Size:** 2,422 observations sampled randomly from the labeled dataset.
- **Annotation:** A column was added with a manual check indicating whether the GPT label is correct or incorrect.
- **Source location:** Google Sheet at https://docs.google.com/spreadsheets/d/1r7fi6DBIeMsVXE7XfOJNZZt8fqwU8TgD/edit
- **Local file:** Not yet exported. Cannot be accessed directly from this environment (no Drive auth). Alexander will export to CSV/Parquet for analysis.

**Open questions to clarify before computing validation stats:**

1. From which pool was the 2,422 drawn — the 6,588-row training-ready file, the 7,057-row merged file, or a pre-filter pool (the ~10,983 USA candidates from semantic match)? This determines whether the validation measures the final dataset's quality or the pipeline's noisier intermediate.
2. Is the manual check single-labeler (just Alexander or one RA) or double-labeled (so we can compute Cohen's κ for inter-annotator agreement)?
3. What does "correct/incorrect" mean exactly — does it use the same rubric as the GPT prompt (NAICS sector alignment), or a simpler binary judgment?
4. Are correct/incorrect labels balanced across NAICS sectors, or skewed?

**Why this matters:** This is the headline evidence for the paper. The number we report as "GPT label precision" comes directly from this sheet. The size (n=2,422) is more than large enough for sector-stratified estimates with reasonable CIs.

**How to apply:** Once the sheet is exported locally, compute (a) overall precision of GPT labels at score ≥ 8, (b) per-sector precision, (c) precision conditional on score = 8 vs 9 vs 10. These three numbers go in the paper's validation table.

**Headline validation results (computed 2026-05-22, n=2,421):**
- **Overall precision: 96.98%** (2,348/2,421; 95% Wilson CI [96.23%, 97.59%])
- 6 sectors at 100% precision: 21 Mining, 48-49 Transportation, 51 Information, 53 Real Estate, 71 Arts/Entertainment, 92 Public Admin.
- 2 weak sectors: **31-33 Manufacturing (73.0%)** and **42 Wholesale Trade (73.0%)**. Both deserve discussion in the paper.
- Precision is monotonic in GPT score: score 8 → 90.76%, score 9 → 97.81%, score 10 → 99.30%. This is strong evidence the rubric is meaningful.
- Cross-tab insight: Manufacturing labels at score 8 are very weak (20%); at score 9 they're 80.5%; at score 10 they're 100%. Same pattern in Wholesale Trade.
- Output tables saved at `paper/validation/output/{overall_precision,per_sector_precision,precision_by_score,sector_score_*}.csv`.

**Caveat:** Sample frame join check failed because the training file (`train_data_gpt_ab8_score_with_code.parquet`) has only `name_repo` (no `nwo`), while the gold set has both. A name_repo-based join is still possible but `name_repo` may not be globally unique. Re-run the check using `(name_repo, code)` as a composite key when needed.

**Sample frame check completed 2026-05-22:** (`paper/validation/check_sample_frame.py`)
- **Gold is a 100% subset of the 7,057-row merged file** (`train_data_naics_github.parquet`) when joining on `(nwo, code)`. Zero code mismatches.
- Match rate drops to **88.97% against the 6,588-row training-ready file** when joining on `(name_repo, code)` — partly because `name_repo` is not globally unique, and partly because 469 rows were dropped between the two files.

**Corrected sample-frame understanding (2026-05-23):** Earlier framing said the gold set was "drawn from the hybrid 7,057-row corpus" — that was technically true but misleading. **The gold set is in fact a 95.4% random subsample of `usa_train_data_gpt_ab8score.parquet`** (2,421 of 2,538 USA GPT-4.1 rows). Empirical evidence:
- 100% of gold-set `nwo` values are in `usa_train_data_gpt_ab8score.parquet`.
- 99.75% are unambiguously in the USA regional source pool; the other 6 rows have `nwo` strings that collide across regional pools.
- Sector 55 is absent (consistent with USA GPT, which also lacks 55).
- Score-distribution ratios are uniform at 0.952 / 0.954 / 0.959 across scores 8 / 9 / 10.

**Validation scope (locked):** The 96.98% precision figure is a clean GPT-4.1 measurement on USA repositories only. It does NOT cover EU or AU. The EU and AU portions used DeepSeek-V3 in v0 (out of scope) and are planned for GPT-4.1 re-classification; separate gold-label sampling will be needed for those regions.

**For the public release:** Use `train_data_gpt_ab8_score_with_code.parquet` (6,588 rows) as the canonical "NAICS-GH" dataset. The 7,057-row `train_data_naics_github.parquet` is an intermediate artifact with full provenance.
