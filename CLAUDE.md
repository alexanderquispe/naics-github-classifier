# CLAUDE.md - Project Rules and Lessons Learned

## Project Overview
NAICS GitHub Classifier: Classifies GitHub repositories into NAICS industry sectors using embeddings (FAISS) + LLM verification.

## Pipeline Steps
1. `01_generate_embeddings.py` — Embed repos using BGE (GPU). Input: description + topics + readme.
2. `02_build_faiss_index.py` — Build FAISS index (GPU).
3. `03_retrieve_by_naics.py` — Retrieve candidate repos per NAICS sector via semantic search.
4. `04_classify_repos.py` — LLM classifies each repo-NAICS pair.

## Critical Rules

### The original notebooks are the reference — with three documented divergences
- Original Colab notebooks: `C:\Users\Alexander\Dropbox\naics_github\Colab Notebooks\`
  (`1_eu_index_metadata_creation.ipynb`, `1_au_index_metadata_creation.ipynb`,
  `2_eu_index_naics_match.ipynb` = USA 511k run, `3_eu_repo_gpt_classif.ipynb`).
- When in doubt about ORIGINAL behavior, check the executed cell OUTPUTS in
  the notebooks, not the slides (slide 7 claimed v1.5; the executed code used v1).
- The replication pipeline INTENTIONALLY diverges from the original in exactly
  three ways (full audit 2026-06-10, see PIPELINE.md funnel section):
  1. Embedding model: original `bge-large-en` (v1) → ours `bge-large-en-v1.5`.
  2. Embedding input: original embedded README-ONLY (first 1,000 chars,
     notebook cell 11) → ours embeds description + topics + readme.
  3. (Consequence) within-sector dedup rises 47–54% → 68–75%, so candidate
     counts drop: original 10,983/10,531/9,664 = 31,178; ours
     6,607/6,138/5,224 = 17,969 (Tier 1) / 18,011 (Tier 2 merged 1200/60).
     Raw retrieval is ALWAYS 20,879/region (Σ n×k) — use that identity to
     verify protocol fidelity of any rerun.
- Do NOT cite 28,341 as the original candidate count — that figure mixed the
  Colab USA run with a superseded local EU/AU rerun. The verified original
  total is 31,178.

### Embedding input text: description + topics + readme (deliberate improvement)
- Format: `"description: {desc}, topics: {topics}, readme: {readme}"`
- This is OUR pipeline's behavior (`create_input_text()` in
  `src/data/preprocessor.py`) and a deliberate upgrade. The ORIGINAL Colab
  run embedded only `readme_content[:1000]` — the combined input_text existed
  in the notebooks ONLY for the GPT step. Keep embedding all three; just
  don't claim the original did.

### LLM classification input must include description + topics + readme
- Combined format as above. Do NOT send only the README to the LLM.
- The original notebook DOES build `input_text` (desc + topics + readme)
  before sending to GPT (e.g., AU notebook cell 50) — original and ours
  agree on this step.

### Retrieval query format
- Use: `"Repositories about {subindustry}"`
- Do NOT use: `"{subindustry} software tools applications"`
- The original notebook uses the "Repositories about" prefix.
- k per subindustry: `max(20, ceil(400/n))` — confirmed identical in the
  original notebooks and `03_retrieve_by_naics.py` (base_k=400/min_k=20;
  Tier 2 merged run uses 1200/60 = 3× to match three regional runs).

### BGE prefixes are ASYMMETRIC
- Documents: `"Represent this document for retrieval: "`
- Queries: `"Represent this query for retrieval: "` (a DIFFERENT prefix —
  notebook cell 28 and current `bge_encoder.py` agree).
- A previous version of this file said queries reuse the document prefix.
  That described an old local-code regression (which produced the superseded
  9,062/8,296 EU/AU candidate files), not the notebook.

### Output directories must exist before writing
- Always create output directories with `mkdir -p` BEFORE running scripts.
- The embedding save crashed once because `embeddings_au/` didn't exist.

### Parallel workers and batch resume
- When using `--workers` > 1 with `--start-idx`, the batch numbering must account for previously saved batches.
- `as_completed` returns results out of order — batch_0000 from a test run WILL be overwritten by a resumed run.
- Solution: Either clean old batches before a full rerun, or don't mix test runs with resumed runs in the same output directory.

### AU dataset naming convention
- All AU output files must include "au" in their names.
- Examples: `au_retrieved_repos_by_naics.parquet`, `au_deepseek_v3_batch_csvs/`, `au_deepseek_v3_matches.csv`

### DeepSeek API specifics
- `deepseek-reasoner` (R1): No `temperature` parameter, needs `max_tokens=8000`, ~28s/repo.
- `deepseek-chat` (V3): Supports `temperature`, `max_tokens=200`, ~5.6s/repo.
- DeepSeek has no fixed rate limit — safe to parallelize with 10+ workers.

### Environment
- Conda env: `naics-classifier` at `C:\Users\Alexander\anaconda3\envs\naics-classifier`
- GPU: NVIDIA RTX 3080 Laptop, CUDA 11.8
- Python executable: `C:\Users\Alexander\anaconda3\envs\naics-classifier\python.exe`
- The older `naics` env is deprecated — do not use it for new work.
- PowerShell reports "Exit code 1" for tqdm stderr output — this is NOT always an actual error. Check the output.

### Polars over pandas for large-parquet work
- EXCEPTION: the faithful-replication folder (`naics-tier1-replication`)
  uses pandas ON PURPOSE — its dedup keep-first / merge / row-order
  semantics must match the original notebooks exactly, and the
  replication targets (10,983/10,531/9,664) depend on them. Do not
  convert those scripts to polars.
- For loading, merging, or transforming the multi-million-row source parquets, prefer Polars (`pl.scan_parquet` + lazy ops) over pandas. It is 2-5x faster and uses less peak memory because lazy evaluation avoids materialising intermediates.
- Polars also catches schema drift loudly (e.g. AU stores `topics` as `List[String]`, USA/EU as `String`) where pandas silently turns the column into `object` dtype. That's the kind of bug we want to surface, not hide.
- pandas is still fine for small tables (gold set, validation outputs, the 6.5k-row training corpus).

### Cost awareness
- User has a limited budget. Always estimate costs before running large LLM jobs.
- deepseek-chat: ~$0.0003/repo, deepseek-reasoner: ~$0.0006/repo
