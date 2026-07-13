---
name: dataset-lineage
description: Final dataset lineage for the NAICS-GH paper — verified original funnel 31,178 retrieval candidates → 6,588 training samples; superseded 28,341 figure mixed two runs.
metadata:
  type: project
---

**Verified pipeline funnel (corrected 2026-06-10 from executed Colab
notebook outputs; supersedes the 2026-05-23 version of this note):**

| Stage | USA | EU | AU | Total |
|---|---:|---:|---:|---:|
| Source pool (post-SQL, unique nwo) | 510,380 | 530,898 | 331,211 | 1,372,489 |
| Step 3 raw retrieval (Σ n×k, k = max(20, ⌈400/n⌉)) | 20,879 | 20,879 | 20,879 | 62,637 |
| Step 3 candidates (dedup on nwo × sector) | 10,983 | 10,531 | 9,664 | **31,178** |
| Step 4 GPT-4.1 score ≥ 8 | 2,529 | 2,039 | 2,021 | **6,589** |
| Step 6 training-ready (19 sectors) | — | — | — | **6,588** |

End-to-end retention: 6,588 / 1,372,489 ≈ 0.48%.
Candidate → final: 6,588 / 31,178 ≈ 21.1%.

**Do NOT use 28,341.** That figure (10,983 + 9,062 + 8,296) mixed the
Colab USA run with a later local EU/AU rerun (bge v1, symmetric query
prefix, full input_text) whose candidates never fed GPT.

**Original-run config:** bge-large-en (v1), README-only embeddings
(first 1,000 chars), asymmetric query prefix, dynamic k. The combined
desc+topics+readme text was used only for the GPT step.

**Why this matters:** The mixed-LLM issue stands — EU and AU labels in
the 7,057-row intermediate file came from DeepSeek-V3, not GPT-4.1, so
re-classification is required for a clean GPT-only narrative.

**Current plan (supersedes the EU/AU-only re-run):** the Tier-1
regional pipeline in `naics-tier1-internal` (bge-large-en-v1.5,
desc+topics+readme embeddings) is staged through retrieval —
6,607 USA + 6,138 EU + 5,224 AU = **17,969 candidates** — and
Alexander's team runs GPT-4.1 on all three regions with the internal
API. The paper's funnel will be regenerated end-to-end from that run;
the original 31,178 → 6,588 funnel is reported as the v1 historical
run. See PIPELINE.md "Pipeline funnel" for the full audit trail.
Related: [[tier1-internal-folder]].
