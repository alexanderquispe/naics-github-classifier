# NAICS-GH paper — `paper/` directory

This directory contains the draft paper, bibliography, validation
analysis, and a (forthcoming) datasheet for the NAICS-GH dataset.

## Layout

```
paper/
├── CHARTER.md                       # Phase 0 charter (locked)
├── README.md                        # This file
├── main.tex                         # Manuscript skeleton
├── references.bib                   # Bibliography
├── figures/                         # Figures used in the paper
└── validation/
    ├── compute_validation_stats.py  # Phase 1: precision / per-sector / by-score
    ├── check_sample_frame.py        # Phase 1: gold ⊆ corpus join check
    └── output/                      # CSV tables consumed by main.tex
```

## Target venue and style

**Primary target:** NeurIPS Datasets & Benchmarks Track.

The skeleton in `main.tex` currently compiles with the default
`article` class so you can iterate without the NeurIPS style file.
Before submitting:

1. Download the official NeurIPS D&B style files from the call for
   papers page (typically named `neurips_data_<year>.sty`).
2. Drop the `.sty` file in this directory.
3. In `main.tex`, comment out `\documentclass{article}` and uncomment
   the `\usepackage[preprint]{neurips_data_<year>}` line (with the
   matching year).
4. For camera-ready, remove the `[preprint]` option.

## How to compile

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Output: `main.pdf`.

## Open TODOs visible in `main.tex`

Search for `TODO` in `main.tex` and `references.bib`. Current pending
items:

- **Tables.** Auto-generate from `validation/output/*.csv`:
  - `tab:sector-counts` (per-sector row counts in the released corpus)
  - `tab:per-sector-precision` (the per-sector validation table)
- **Citations.** All `\cite{TODO-*}` entries need to be replaced with
  real, verified references.
- **Appendices.** Schema, NAICS sector definitions, full LLM prompt,
  training hyperparameters, qualitative error analysis (waiting on RAs).
- **Datasheet.** To be drafted as `paper/DATASHEET.md` and converted to
  LaTeX for `app:datasheet`.
- **Release URLs.** Once Hugging Face and Zenodo identifiers exist,
  replace the `TODO` URLs in `\section{Release and Reproducibility}`.

## Numbers already plugged in

These are verified against on-disk data and the Phase 1 validation run
(2026-05-22):

| Item | Value | Source |
|---|---|---|
| Released corpus size | 6,588 repositories | `naics-github-train/data/raw/train_data_gpt_ab8_score_with_code.parquet` |
| Number of NAICS sectors covered | 19 of 20 (sector 55 absent) | per-sector counts |
| Validation sample size | 2,421 | `data/raw/gold_set_manual_check.xlsx` |
| Overall precision | **96.98%** (95% CI [96.23%, 97.59%]) | `validation/output/overall_precision.csv` |
| Precision at score 8 / 9 / 10 | 90.76% / 97.81% / 99.30% | `validation/output/precision_by_score.csv` |
| Weakest sectors | 31-33 Manufacturing & 42 Wholesale Trade (73.0%) | `validation/output/per_sector_precision.csv` |
| RoBERTa-large test F1 | 86.45% | `Colab Notebooks/naics_training_gptdata.ipynb` |
| RoBERTa-large test accuracy | 86.35% | `Colab Notebooks/naics_training_gptdata.ipynb` |
| Six encoders compared | RoBERTa, ModernBERT, DeBERTa-v3 (base + large) | same notebook |
| Train/Val/Test split | 4,611 / 659 / 1,318 | `naics-github-train/README.md` |
