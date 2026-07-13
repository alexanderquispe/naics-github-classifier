# NAICS-GH Replication Package — Plan

**Goal.** A reviewer or independent researcher can, from a fresh clone of
`naics-github-classifier`, reproduce every result reported in the paper
at a level of effort proportional to what they want to verify. The
package must satisfy NeurIPS 2025 D&B's reproducibility bar (Hugging
Face hosting, Croissant metadata, MIT-licensed code, datasheet, fixed
seeds, hyperparameter list).

This plan does **not** rewrite the paper; the paper is the
specification. This plan exists so we know exactly what assets to
publish, which scripts to rerun, and what is still missing.

## Data-release policy (binding)

**Per-row country attributes are not published.** Per the affiliation
constraints of the corresponding author, no published artifact may
carry a per-repository jurisdiction (USA / EU / AU) label, even via a
`country_code` column. The source SQL extracts three regional pools
for operational convenience; the published pipeline immediately
concatenates them into a single merged pool and drops the country
attribute (see `scripts/merge_and_strip_country.py`). The paper
reports the **aggregate** jurisdictional composition of the source
pool but no downstream per-region figures.

All assets listed below are the **merged, country-blind versions**.
The three regional intermediates are never published.

## Reproduction policy: two-tier (internal regional + public merged)

We run the pipeline in **two tiers** (see `PIPELINE.md` §0.5 for the
full architecture). They share model, prompt, and filtering and differ
only in whether jurisdiction is retained.

### Tier 1 — Internal canonical pipeline (regional, with country)

- Runs inside GitHub on jurisdiction-tagged data.
- Three regional pools embedded/indexed/retrieved separately, then GPT
  scored and filtered (`score >= 8`).
- Produces **(a)** the canonical labeled corpus that the paper
  benchmarks on and that ships to Hugging Face *with the country column
  stripped*, and **(b)** the cross-jurisdiction analysis reported in the
  paper (sector mix by jurisdiction; GPT score distribution by
  jurisdiction — both aggregate, hence publishable).
- Uses the **improved** pipeline (BGE v1.5, combined input, corrected
  query prefix), not the legacy v1.0/README-only configuration that
  produced the original 6,588-row corpus.
- **Efficiency:** reuses the merged-pool embeddings (Step 1) partitioned
  by `nwo` membership in the three regional source files; only retrieval
  and GPT scoring re-run per region.

### Tier 2 — Public replication pipeline (merged, country-blind)

- Runs anywhere, by anyone, on publishable data.
- Single merged pool (`merge_and_strip_country.py`) → one
  embed/index/retrieve/GPT/filter pass.
- Produces a corpus of **comparable size and quality**, not
  bit-identical: merged retrieval selects best-overall candidates
  rather than enforcing per-region quotas.
- This is the L2–L4 reproduction path shipped to reviewers (code +
  merged source pool + merged intermediates on Zenodo).

### What is published vs. retained

| Artifact | Country retained? | Published? |
|---|---|---|
| Tier 1 regional intermediates | yes | **no** (internal only) |
| Cross-jurisdiction figures (aggregate) | aggregate only | **yes** (in paper) |
| Canonical corpus (HF) | stripped before export | **yes**, country-blind |
| Tier 2 merged pipeline + intermediates | never present | **yes**, country-blind |

No published row in any artifact carries a per-row jurisdiction label.
The two-tier discrepancy ("comparable, not identical") is disclosed in
the paper and the Zenodo README.

---

## Four levels of reproduction

A reviewer should be able to pick the level that matches the time and
budget they have. The package is built so that each level depends only
on artifacts from earlier levels.

| Level | What it proves | Time | Compute | Cost | Inputs needed |
|---|---|---|---|---|---|
| **L1 — Smoke test** | The corpus loads, the published classifier works | 5 min | CPU laptop | $0 | HF `aquiro1994/naics-gh` + HF model `alexanderquispe/naics-github-classifier` |
| **L2 — Re-train downstream** | The reported 86.45% F1 reproduces from the released training corpus | 15 min | 1× A100 | $0 (if your own GPU) | HF `aquiro1994/naics-gh` + `naics-github-train` repo |
| **L3 — Re-label from candidates** | The GPT-4.1 scoring step reproduces from the retrieval output | 6–10 hr | API only | $50–80 GPT-4.1 spend | merged `retrieval_candidates.parquet` (country-blind, published to Zenodo) + OpenAI key |
| **L4 — Re-retrieve from source** | The BGE+FAISS retrieval reproduces from the SQL-extracted source pool | 1–2 hr | 1× GPU | $0 | merged `source_pool.parquet` (country-blind, published to Zenodo) + GPU |
| **L5 — Re-extract from warehouse** | The SQL itself reproduces (the source pool) | 1 hr | warehouse access | depends | Internal GitHub `hive` warehouse access — NOT reproducible by external reviewers |

L1 and L2 are the **public-facing reproduction paths**: any reviewer in
the world can run them today. L3 and L4 are **optional deep-dive paths**
gated only by storage hosting (which we can solve). L5 is intentionally
**not reproducible** by outsiders; we document the SQL but cannot
release the warehouse data.

---

## Asset inventory

Everything the package needs, with current status and target hosting.

### A. Code (already in this repo)

| Asset | Path in repo | Status | Action before submission |
|---|---|---|---|
| Pipeline scripts (5 numbered) | `scripts/01_*` … `scripts/05_*` | Functional, parameterized via CLI | Add a single `scripts/run_all.sh` that operates on the merged source pool |
| BGE encoder + FAISS indexer | `src/embeddings/` | Functional | Pin `BAAI/bge-large-en-v1.5` exactly (currently defaults to `bge-large-en`) |
| GPT classifier + prompt builder | `src/classification/` | Functional | Reflect the production prompt (with `"NAICS {code}"` nested JSON schema); strip hardcoded PAT |
| Preprocessor + token counter | `src/data/`, `src/utils/` | Functional | No change |
| Country-strip / merge utility | `scripts/merge_and_strip_country.py` | Written | Run once to produce the merged source/intermediate parquets for publication |
| Config defaults | `config/config.yaml` | Region-hardcoded to EU | Collapse to a single `config/config.yaml` pointed at the merged inputs (no per-region configs in the public package) |
| Test stubs | `tests/__init__.py` | Empty | Add 2–3 smoke tests (loader, encoder, prompt builder) |
| Paper sources | `paper/` | Complete and compiles to 26-page PDF | No change |
| `PIPELINE.md` | repo root | 60-page comprehensive | Cross-link from `README.md` |
| `requirements.txt` | repo root | Loose version ranges | **Pin versions** to those tested with the published RoBERTa-large checkpoint |
| `environment.yml` (conda) | absent | — | Add for users who prefer conda |
| `Dockerfile` | absent | — | Optional but reviewer-friendly |
| `.env.example` | already present | OK | Document `OPENAI_API_KEY` vs. `GITHUB_TOKEN` clearly |

### B. Datasets

| Artifact | Size | Currently | Target hosting | Why |
|---|---:|---|---|---|
| **Released training corpus** (6,588 rows, no country column) | 12 MB | ✅ `huggingface.co/datasets/aquiro1994/naics-gh` | HF (live) | L1, L2 entry point |
| **Pre-split train / val / test parquets** | 12 MB | ✅ already in the HF dataset | HF (live) | Reproduce the 86.45% F1 exactly |
| **NAICS taxonomy JSON** | 39 KB | `data/raw/naics_titles_by_group_6digit_clean.json` | HF + GitHub | Drives Step 3 retrieval queries |
| **Gold-set XLSX** (2,421 manually re-checked, country-stripped) | 4 MB | `data/raw/gold_set_manual_check.xlsx` (still has `nwo`) | HF as a second config (`gold_set`); strip `nwo` first | Re-derive precision numbers |
| **Merged source pool** (~1.37M rows, country-blind, all three regions concatenated and shuffled) | ~1 GB | Three regional files in `data/raw/`; produce merged via `scripts/merge_and_strip_country.py` | **Zenodo only** (too large for HF) | L4 entry: re-retrieve from scratch |
| **BGE embeddings of the merged pool** (1024-dim, FP16) | ~3 GB | To be generated by rerun (step 6) | Zenodo only (too large for HF) | Skip the GPU embedding step on replication |
| **FAISS index of the merged pool** | ~3 GB | To be generated by rerun (step 6) | Zenodo only (too large for HF) | Skip the index-build step on replication |
| **Merged retrieval output** (~28k candidates, country-blind) | ~40 MB | To be generated by rerun (step 7) | Zenodo + HF config (`retrieval_candidates`) | L3 entry: skip the GPU step, run only GPT-4.1 verification |
| **Merged GPT-4.1 score file** (country-blind, rerun) | ~10 MB | To be generated by rerun (step 8) | Zenodo + HF config (`gpt_scores`) | L3 output / L2 input: skip the expensive GPT step entirely |
| **Rerun training corpus** (country-blind, produced fresh on this machine) | ~12 MB | To be generated by rerun (step 9) | Zenodo only — paper benchmark stays on the *frozen* HF corpus | Documents the rerun, does not replace `aquiro1994/naics-gh` |
| **Source-extraction SQL** | Tiny | In paper Appendix A + this repo | GitHub | L5: documented but warehouse not reproducible externally |
| **Downloaded reference papers (literature search)** | 75 MB | `paper/literature/pdfs/` | Not released | Internal artifact; not needed for replication |

### C. Models

| Model | Status | Hosting | Notes |
|---|---|---|---|
| **BAAI/bge-large-en-v1.5** | upstream | HF hub (theirs) | We just consume it |
| **GPT-4.1** snapshot `gpt-4.1-2025-04-14` | upstream | OpenAI API / GitHub Copilot LLM-lab | API call |
| **RoBERTa-large NAICS classifier** | ✅ published | `huggingface.co/alexanderquispe/naics-github-classifier` | L1 / L2 entry |
| **Other 5 baselines** (RoBERTa-base, ModernBERT-{base,large}, DeBERTa-v3-{base,large}) | not published; trainable on demand | Optional HF uploads | Helpful for ablation reproducibility but not essential |

---

## Outstanding tasks (the punch list)

Each task has an effort tag and the action that completes it. Tasks
are ordered roughly by reproducibility-impact, not by effort.

### High-impact (do before submission)

1. **Pin model and library versions in `requirements.txt`** — change `transformers>=4.30.0` to the exact version used (≈ `transformers==4.46.x` per the training notebook). Same for `torch`, `sentence-transformers`, `faiss-gpu`. Effort: 30 min.

2. **Switch the BGE encoder default to `BAAI/bge-large-en-v1.5`** — the paper claims v1.5; the code defaults to v1.0. Effort: 15 min. (`src/embeddings/bge_encoder.py:23`)

3. **Sync the local prompt builder to the production format** — production uses the nested `"NAICS {code}"` JSON schema with string-valued `"score"` / `"match"`. The local script's `prompt_builder.py` outputs the older flat schema. Effort: 30 min.

4. **Remove the hardcoded PAT** from any code path. The Colab notebook had it; the local repo should not. Effort: 10 min.

5. **Rerun step: merge raw parquets → country-blind source pool.** Run `scripts/merge_and_strip_country.py` over the three regional `data/raw/repos_*.parquet` files. Output: `data/processed/source_pool_merged.parquet` (~1.37M rows). Effort: 5 min, free.

6. **Rerun step: BGE embeddings + FAISS index.** Run `scripts/01_generate_embeddings.py` on the merged pool, then `scripts/02_build_faiss_index.py`. Output: `data/processed/embeddings/repos_with_embeddings_vectors.npy` and `data/processed/faiss_indices/index.faiss`. Effort: ~25 min on the RTX 3080, free.

7. **Rerun step: retrieve candidates.** Run `scripts/03_retrieve_by_naics.py` over the merged FAISS index using the 1,029 NAICS subindustry queries. Output: `data/processed/retrieval_candidates.parquet` (~28k rows). Effort: ~5 min, free.

8. **Rerun step: GPT-4.1 verification.** Run `scripts/04_classify_repos.py` over the merged retrieval candidates. **This is the only paid step — estimated $340–420 in GPT-4.1 spend.** Effort: ~8–12 hours wall-clock with the existing 10-worker setup. Requires explicit budget sign-off before starting.

9. **Rerun step: filter to final corpus.** Run `scripts/05_filter_results.py` over the GPT scores. Output: `data/processed/train_data_rerun.parquet`. Effort: 5 min, free.

10. **Write `scripts/run_all.sh`** — one shell script that runs steps 5→9 in sequence over the merged country-blind source pool. Effort: 1 hour.

11. **Pin a Zenodo DOI for the rerun-trace artifacts** — the merged source pool, the BGE embeddings, the FAISS index, the merged retrieval-candidates file, the merged GPT-4.1 score file, and the rerun training corpus. Effort: 1 hour (mostly waiting for Zenodo).

12. **Add `environment.yml`** for conda users. Effort: 20 min.

### Medium-impact

13. **Write `tests/` smoke tests** that exercise the loader, encoder, FAISS indexer, and prompt builder. Effort: 2 hours.

14. **Add a `Dockerfile`** that builds an image with all dependencies pinned and exposes one entrypoint per pipeline stage. Effort: 1.5 hours.

15. **Add a top-of-`README.md` quickstart** that runs L1 in under 1 minute:

    ```bash
    pip install datasets transformers
    python -c "from datasets import load_dataset; ds = load_dataset('aquiro1994/naics-gh'); print(ds)"
    ```

16. ~~Split `config.yaml` into three region configs~~ — **superseded by Option B**: the public pipeline operates on the merged country-blind pool, so a single `config.yaml` is the right surface.

17. **Move `CLAUDE.md` and `FORALEXANDER.md` out of the public root** — they're internal notes. Either delete or move under `internal/`. Effort: 5 min.

18. **Cross-link `PIPELINE.md` from the top of `README.md`** so readers find the detailed methodology. Effort: 2 min.

### Low-impact / nice-to-have

19. **Generate Croissant metadata explicitly** — HF currently auto-generates from the YAML; explicit `croissant.json` in the repo root is cleaner. Effort: 1 hour with `mlcroissant`.

20. **Pre-baked Jupyter notebook** in `notebooks/` that walks a reader through the L1 path with explanations. Effort: 1 hour.

21. **Continuous integration** (`.github/workflows/test.yml`) that runs the smoke tests on every push. Effort: 30 min.

---

## The reviewer's reproduction flow

This is what the package's `README.md` should ultimately tell a
reviewer. The current `README.md` is structured for a developer; the
public-facing version should foreground the L1 / L2 / L3 / L4 ladder.

### L1 — Smoke test (5 minutes)

```bash
pip install datasets transformers torch
python - <<'PY'
from datasets import load_dataset
from transformers import pipeline

ds = load_dataset("aquiro1994/naics-gh", split="test")
clf = pipeline("text-classification",
               model="alexanderquispe/naics-github-classifier")

row = ds[0]
text = (f"Repository: {row['name_repo']} | Description: {row['description']} | "
        f"Topics: {row['topics']} | README: {row['readme_content']}")
pred = clf(text)[0]
print("Predicted:", pred["label"], "  Actual:", row["code"])
PY
```

### L2 — Reproduce the benchmark (15 minutes on A100)

```bash
git clone https://github.com/alexanderquispe/naics-github-train
cd naics-github-train
pip install -r requirements.txt
python scripts/train.py --model roberta-large --batch-size 32 --epochs 8 \
       --dataset aquiro1994/naics-gh
# Expected: weighted F1 ≈ 86.45% on the test split.
```

### L3 — Re-label from candidates (6–10 hours)

```bash
git clone https://github.com/alexanderquispe/naics-github-classifier
cd naics-github-classifier
pip install -r requirements.txt

# Download the Zenodo-hosted merged retrieval candidates (~40 MB)
wget https://zenodo.org/records/<DOI>/files/retrieval_candidates.parquet \
     -O data/processed/retrieval_candidates.parquet

# Set your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# Run GPT-4.1 verification over the merged candidate set
python scripts/04_classify_repos.py --input data/processed/retrieval_candidates.parquet
python scripts/05_filter_results.py
```

### L4 — Re-retrieve from scratch (1–2 hours)

```bash
# Download the Zenodo-hosted merged source pool (~1 GB)
wget https://zenodo.org/records/<DOI>/files/source_pool.parquet \
     -O data/raw/source_pool.parquet

# Embed, index, retrieve, classify, filter
bash scripts/run_all.sh   # one-shot over the merged pool
```

### L5 — Documented only

The Presto/Trino SQL that produced the regional source parquets is in
Appendix A of the paper. It runs against an internal GitHub data
warehouse and **cannot be reproduced by external users**. Reviewers can
inspect the SQL but must take the merged source pool as given. The
per-row country attribute used by the SQL is dropped during the
merge-and-strip step (`scripts/merge_and_strip_country.py`) and is
**not** part of any published artifact.

---

## What "done" looks like for the replication package

Tick this checklist at submission:

- [ ] `requirements.txt` pins all versions used for the published model
- [ ] `environment.yml` (conda) exists
- [ ] Single-command `scripts/run_all.sh` works end-to-end over the merged pool
- [ ] `scripts/merge_and_strip_country.py` has been run and its output verified to contain no jurisdictional column
- [ ] `tests/` has smoke tests for each module (loader, encoder, indexer, prompt)
- [ ] `Dockerfile` builds and the L1 smoke test passes inside the image
- [ ] `.env.example` documents both API key options
- [ ] PAT removed from all notebooks; `.env`-driven everywhere
- [ ] Top-of-`README.md` shows the L1 smoke test as the first code block
- [ ] HF dataset `aquiro1994/naics-gh` is live, Croissant validates, and the schema carries no country column
- [ ] HF model `alexanderquispe/naics-github-classifier` is live
- [ ] Zenodo DOI exists for the merged source pool (L4)
- [ ] Zenodo DOI exists for the merged retrieval-candidates parquet (L3)
- [ ] Zenodo DOI exists for the merged GPT-4.1 score parquet (L3)
- [ ] All Zenodo records linked from `README.md`
- [ ] BGE encoder model name in code matches paper (`-v1.5`)
- [ ] Prompt builder produces the production-schema JSON (nested under `"NAICS {code}"`)
- [ ] `CLAUDE.md`, `FORALEXANDER.md`, `nul` removed or moved under `internal/`
- [ ] `PIPELINE.md` linked from `README.md`
- [ ] All cross-references in the paper resolve and the PDF compiles cleanly
- [ ] Paper appendices A, B, C, D, E, F, H complete (G pending RA work)
- [ ] Datasheet for Datasets ships in the supplementary archive

---

## Estimated effort to "submission-ready"

Assuming the user (Alexander) drives the dataset uploads and 1 RA day is
available; everything else is Claude time:

| Stage | Effort | Owner |
|---|---|---|
| Code refactor (tasks 1–4, 10, 13, 14, 15) | 4 hours | Claude |
| Pipeline wrapper script + region configs (task 7) | 1 hour | Claude |
| Zenodo uploads for source + intermediate parquets (tasks 5, 6, 8) | 2 hours human + waiting | Alexander |
| README.md quickstart rewrite (task 12) | 30 min | Claude |
| Smoke tests + Dockerfile (tasks 10, 11) | 3 hours | Claude |
| environment.yml + Croissant explicit JSON (tasks 9, 16) | 1 hour | Claude |
| **Total** | **~12 hours** | mixed |

Realistic calendar: **3 working days** if Alexander handles the Zenodo
side in parallel with Claude doing the code work.

---

## Open questions for Alexander

These need a decision before we can finalize:

1. **Are you willing to publicly release the merged country-blind source pool** (~1 GB, 1.37M rows)? Per Option B the per-row country attribute is stripped before publication; the artifact carries only the aggregate composition already reported in the paper. The merge utility (`scripts/merge_and_strip_country.py`) refuses to write any output that still carries a jurisdictional column.

2. **What account hosts the Zenodo records?** `alexanderquispe` or a project-level Caltech account? (Microsoft affiliation is the reason for Option B; the Zenodo upload should be under a Caltech-affiliated identity, not a Microsoft one.)

3. **Do we want to also publish the gold-set XLSX** as a second config on the HF dataset (so the 96.98% precision is exactly reproducible from public data)? My recommendation: yes — but strip `nwo` first so it carries only sector codes and the manual-check column.

4. **Are we OK rotating both shared HF tokens** (current chat + earlier chat) right now? If they're still live, anyone reading the chat history can use them.

5. **Final Zenodo DOI naming convention.** Suggest one record per artifact group (source / candidates / scores) so each can be updated independently without invalidating older citations. Under Option B, each group is a single merged file rather than three regional files.
