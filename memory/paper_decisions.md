---
name: paper-decisions
description: Locked decisions for the NAICS-GH paper: dataset paper (option A), public release approved, target venue NeurIPS Datasets & Benchmarks.
metadata:
  type: project
---

Decisions confirmed 2026-05-15 by Alexander:

- **Paper type:** Option A — dataset paper anchored on the NAICS-labeled GitHub corpus (EU + AU + USA + IN regions).
- **Public release:** Approved. The labeled dataset and pipeline can be released publicly.
- **Primary target venue:** NeurIPS Datasets & Benchmarks (D&B track). Fallback: ACL Rolling Review / EMNLP resource track, or TMLR.
- **Co-author:** Kevin Xu (see [[collaborator-kevin-xu]]).

**Why:** D&B explicitly rewards LLM-weak-supervision dataset releases and requires the methodology section, so the pipeline gets published as part of the dataset paper. Public release means we can target the strongest possible venue rather than a sample-only release.

**How to apply:** Frame all future work (validation, ablations, datasheet, downstream demo) as serving a D&B submission. Default to NeurIPS LaTeX style for the writeup. Treat the labeled corpus + pipeline as the primary artifact; ModernBERT fine-tune is a downstream utility demo, not the main contribution. See also [[project-goal]].

**LLM scope (confirmed 2026-05-15):** The paper's main and final dataset is the GPT-labeled corpus only. Do NOT discuss, compare against, or reference the DeepSeek (V3/R1) outputs in the paper or in conversations about the paper. The DeepSeek files exist on disk for internal exploration but are out of scope. This means inter-LLM agreement validation, if used, must come from a different second model (e.g., an open-weights model run by us) — not from the existing DeepSeek outputs.

**Phase 0 closed (2026-05-22):** Charter at `paper/CHARTER.md` is locked. Confirmed: dataset = NAICS-GH (6,588 rows, USA+EU+AU, GPT-labeled, 19 sectors); both authors are GitHub-affiliated; Alexander corresponding. Open follow-up: Alexander to locate the three regional `*_2k_gpt_ab8score.parquet` files in Google Drive for the public release. Next: Phase 1 — validate labels against the RA gold-label spreadsheet.
