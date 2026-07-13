---
name: project-goal
description: Generate a NAICS-labeled training dataset from GitHub repos via BGE+FAISS retrieval and LLM scoring, to fine-tune a ModernBERT classifier downstream.
metadata:
  type: project
---

The repo `naics-github-classifier` is the data-generation half of a two-repo system. It uses BGE embeddings + FAISS retrieval to surface candidate repos per NAICS sector, then an LLM (GPT-4.x or DeepSeek V3/R1) to score each candidate (1–10). Score ≥ 8 entries are exported as the training set for a downstream ModernBERT fine-tune (separate repo `naics-github-train`).

Datasets currently processed: EU (531k), AU (331k), USA (511k), India (1M, raw only). Outputs exist for EU and AU via deepseek-chat (V3) — see `data/output/{eu,au}_train_data_deepseek_v3_ab8score.parquet`. The original GPT-4 run produced `train_data_gpt_ab8score.parquet`.

**Why:** Framing the project goal makes it easier to scope a paper around the *dataset* and the *labeling methodology*, rather than the downstream classifier.

**How to apply:** When discussing a paper, anchor the contribution in the labeling pipeline + released dataset, not the ModernBERT fine-tune (which lives in a sibling repo).
