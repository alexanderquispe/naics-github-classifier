# NAICS-GH Paper — Improvement Plan

**Created:** 2026-05-22 after end-to-end review.

This document tracks every concrete improvement the paper needs before
submission to NeurIPS Datasets & Benchmarks, grouped by priority and
ownership.

Legend:
- **C** = Claude can do (code/scripts/writing edits)
- **U** = User input required (Alexander or Kevin)
- **R** = RAs (qualitative error analysis, double-labeling if needed)
- **E** = Requires a re-run experiment (compute time)

---

## Headline assessment

The paper has a solid spine (clear pipeline, headline 96.98% precision,
trained downstream model, datasheet drafted) but it currently reads
like a strong workshop paper, not a top-tier D&B submission. Three
gaps will draw the most reviewer fire:

1. **The 96.98% is precision, but the paper writes it as if it were
   end-to-end accuracy.** There is no recall estimate, no false-negative
   analysis, no characterization of *what got missed*. This is the
   methodological gap reviewers will flag first.

2. **The dataset is multi-region but the paper never breaks down
   results by region.** Counts, precision, and the qualitative
   discussion all collapse USA/EU/AU into one number.

3. **Half the appendices are empty TODO blocks.** Real D&B reviewers
   read appendices; empty schema/prompt/training-details sections look
   like the work isn't finished.

Beyond these, there's a long tail of polish items (citations, prior
work, figures, region-specific framing) that individually are minor
but collectively determine whether the paper looks "submission-ready"
or "still drafting."

---

## P0 — submission blockers

These must be fixed before submission. Reviewers will reject the
paper without them.

| # | Item | Owner | Effort |
|---|---|---|---|
| P0-1 | **Add a pipeline figure** (Fig. 1). The presentation slides already have a good diagram. Reproduce as TikZ or import the PNG. Dataset papers without a pipeline figure look incomplete. | C | 1 hour |
| P0-2 | **Reframe "precision" vs "recall"** throughout. Add an explicit paragraph in §5 saying we measure precision-at-score, not end-to-end accuracy. Plan a recall experiment (see P1-1). | C | 30 min |
| P0-3 | **Specify the exact LLM version**. Replace every "GPT-4-class" / "GPT-4.x" with the exact model and snapshot — `gpt-4.1`. Note the API endpoint used. | C+U | 15 min |
| P0-4 | **Per-region breakdowns**. Add per-region counts (USA/EU/AU) to the dataset description, and per-region precision to the validation section. Currently the paper collapses regions. | C+E | 1 hour |
| P0-5 | **Real prior-work citation**. Replace `TODO-prior-repo-classification`. Either find a real reference (Sharma et al. "Cataloging GitHub Repositories" EASE 2017 is a candidate) or explicitly state that no prior work on standardized industry classification of repositories exists. | C+U | 1 hour |
| P0-6 | **Inter-annotator agreement (IAA)**. The paper says "three RAs" and "the second author adjudicated" — both unverified. If RAs single-labeled, document that and call it a limitation. If any rows were double-labeled, compute Cohen's κ. | U | 30 min |
| P0-7 | **Source-pool description**. Specify how the 511k / 531k / 331k pools were filtered: which date the snapshot was taken, were forks/archived repos excluded, what jurisdiction-of-organization rule was applied. The current text is vague. | U+C | 30 min |
| P0-8 | **Fill the empty appendices** A–E (NAICS definitions, full LLM prompt, schema, stratification, training details). Most are paste-ins from existing files. | C | 2 hours |
| P0-9 | **Compute / cost statement**. D&B requires it. Estimate GPT API spend (in dollars) and GPU-hours for embedding + fine-tuning. | U | 15 min |
| P0-10 | **Sample-frame note**. Acknowledge that the gold sample (n=2,421) was drawn from the **7,057-row release file**, not the 6,588-row training derivative; only 88.97% of gold rows are in the training file. The current text glosses this. | C | 15 min |

---

## P1 — highly recommended

These materially strengthen the paper. Reviewers will likely ask for
them.

| # | Item | Owner | Effort |
|---|---|---|---|
| P1-1 | **Recall estimation experiment**. Hand-label 100–200 *uniform-random* (not retrieval-filtered) repos from the source pool; measure what fraction were retrieved at all. Single headline number, half a page. | U+E | 1 day RA work |
| P1-2 | **Per-region precision** (E). Re-run `compute_validation_stats.py` with an added region grouping. Determines whether GPT does worse on EU/AU than USA. | C+E | 30 min if region info is recoverable; up to 2 hours otherwise |
| P1-3 | **Score-threshold tradeoff plot**. Curve of dataset size vs precision as the score threshold rises from 7 to 10. Gives downstream users a knob. | C | 1 hour |
| P1-4 | **Examples table**. Show 5–6 representative repos (1–2 per region) with NAICS label, GPT score, and rationale. Makes the paper concrete. | C | 30 min |
| P1-5 | **Strengthen related work**. Add: GitHub repository topic classification (Sharma et al. 2017, others), LLM-as-labeler (Wang et al. 2023, Chiang & Lee), code-language-model classifiers. The current Related Work is 3 thin paragraphs. | C+U | 2 hours |
| P1-6 | **"Downstream uses" subsection** in main text. Currently folded into the introduction. D&B reviewers want a dedicated section: industry-mix studies, AI-coding-assistant adoption analysis (the `naics-github-train` repo already does this — reference it). | C | 1 hour |
| P1-7 | **Score-distribution figure**. Bar chart or histogram of scores 8/9/10 broken out by sector or region. | C | 30 min |
| P1-8 | **README quality / star bias** acknowledgment. Add to limitations: repos with sparse READMEs are systematically harder to classify, biasing the corpus toward better-documented projects. | C | 15 min |
| P1-9 | **LLM-cutoff bias** acknowledgment. GPT-4.1's training data has a cutoff; recent repos may be labeled differently from older ones. | C | 15 min |
| P1-10 | **Embedding query prefix details**. Mention the BGE prefix `"Represent this document for retrieval: "` and the query template `"Repositories about {subindustry}"` in §3.3. They're non-trivial implementation choices. | C | 15 min |

---

## P2 — polish

These improve the paper but are not critical. Do these after P0 and
P1 are complete.

| # | Item | Owner | Effort |
|---|---|---|---|
| P2-1 | **Rephrase title** to foreground the dataset: e.g., "NAICS-GH: A Multi-Region Industry-Tagged Corpus of GitHub Repositories" or similar. | U | discussion |
| P2-2 | **Table styling consistency**. Use 3-decimal precision throughout (Table 3 still uses 4). | C | 5 min |
| P2-3 | **Conclusion: link to downstream Copilot/Codex/Claude analysis** in `naics-github-train`. Concretizes what the dataset enables. | C | 15 min |
| P2-4 | **`\citet` vs `\citep` consistency**. Make sure parenthetical vs in-text citations are used appropriately. | C | 15 min |
| P2-5 | **Cite NAICS source** at first mention; cite Wilson 1927 at first mention of Wilson intervals. | C | 5 min |
| P2-6 | **Cite ModernBERT** somewhere (e.g., baselines we considered) or remove the bib entry. | C | 5 min |
| P2-7 | **Tighten abstract** to ~200 words. | C | 15 min |
| P2-8 | **Heading style** consistency (`\paragraph{}` vs `\subsection{}` — pick one pattern per section). | C | 15 min |
| P2-9 | **Reproducibility statement** at end of §3: one paragraph saying what's needed to reproduce (data, prompts, configs, hardware). | C | 30 min |
| P2-10 | **Visible TODO scrub**. Search for `TODO` in `main.tex` and `references.bib`; resolve or convert to footnotes. | C | 30 min |

---

## Workflow plan — recommended order

### Week 1: Information gathering and structural fixes (P0-3, P0-6, P0-7, P0-9 with Alexander)

Have a 1-hour conversation with Kevin and write down:
- Exact GPT model/snapshot used (P0-3) — `gpt-4.1` per slides, confirm
- Whether RAs double-labeled anything; if not, confirm and document (P0-6)
- Source-pool filtering details (P0-7) — date of snapshot, fork/archive
  rules, jurisdiction rule used
- Rough total GPT spend in dollars and GPU-hours for fine-tuning (P0-9)

I can then write these into the paper in ~1 hour.

### Week 1: Mechanical paper improvements (P0-1, P0-2, P0-5, P0-8, P0-10, all of P2)

All Claude-doable. Order:
1. Pipeline figure (P0-1)
2. Fill empty appendices (P0-8) — paste in NAICS definitions, the
   verbatim prompt, the schema table, stratification table, training
   hyperparams.
3. Precision-vs-recall reframing (P0-2) and sample-frame note (P0-10).
4. Real prior-work citation or "no-prior-work" paragraph (P0-5).
5. All P2 polish items.

### Week 2: Per-region experiments (P0-4, P1-2)

This requires recovering the region label for each row in the gold
set + the training file. The 7,057-row merged file has full `nwo`
(owner/repository) — we can join back to the three regional dumps to
recover region. Once that's done, the per-region tables and precision
numbers come automatically.

### Week 2: Recall experiment (P1-1) — needs RA time

This is the only major piece that requires net-new labeling work.
Sample 100–200 repos uniformly at random from the USA source pool;
have an RA assign a NAICS sector (or "no sector") for each; measure
what fraction of the sector-X repos appeared in the retrieved
candidate pool. Half a day of RA work, gives the paper a credible
recall number.

### Week 3: Figures and qualitative depth (P1-3, P1-4, P1-6, P1-7)

Once the data is in shape, add the threshold-tradeoff curve, the
examples table, the downstream-uses subsection, and the
score-distribution figure.

### Week 3: Related-work pass (P1-5)

This benefits from doing a real literature search rather than relying
on what we know off the top of our heads. Allocate a couple of hours
to read the Sharma 2017 paper and 3-5 LLM-as-labeler papers, then
write the strengthened Related Work in one sitting.

### Week 4: Final pass, datasheet conversion review, submission

Polish, proofread, run the full build, verify the PDF renders cleanly
under `[final]` (camera-ready), assemble supplementary materials.

---

## What I need from you to proceed

In one short message you can unblock most of the P0 items:

1. **GPT model**: confirm `gpt-4.1` is the exact model used? Or was it
   `gpt-4`, `gpt-4-turbo`, or a specific snapshot like
   `gpt-4-1106-preview`?
2. **RA labeling**: was the 2,421-row gold set single-labeled
   (each row seen by exactly one RA) or did you have any overlap for
   IAA?
3. **Source pool date**: what month/year was each regional GitHub
   snapshot taken?
4. **Source pool filters**: did you exclude forks? archived
   repositories? repos without READMEs? Just confirm the rules used.
5. **Jurisdiction filter**: how did you determine that a repository
   belongs to "USA" vs "EU" vs "AU"? Owner location? Top contributor?
   Some other heuristic?
6. **Cost estimate**: rough total dollars spent on GPT API calls for
   the labeling pipeline?

Once you answer these I can do all the P0 paper edits in one session.
