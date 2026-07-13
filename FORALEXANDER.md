# FORALEXANDER.md -- The NAICS GitHub Classifier, Explained

---

## What This Project Actually Does (The Big Picture)

Imagine you have half a million GitHub repositories and you want to answer one question: **"What industry does each repo belong to?"**

Not "what programming language" or "how many stars" -- but what *real-world economic sector* it serves. Is this repo about agriculture? Finance? Healthcare? Construction?

That's what NAICS codes are for. NAICS (North American Industry Classification System) is the standard the U.S. government uses to categorize every business in the economy into 20 sectors. This project takes GitHub repos and labels them with those same sector codes -- automatically, at scale.

**Why?** Because the end goal is to build training data for a ModernBERT classifier (a separate project: `naics-github-train`). That model needs thousands of labeled examples of "this repo belongs to NAICS sector 11 (Agriculture)" or "this repo belongs to sector 52 (Finance)." You can't hand-label half a million repos. So this pipeline does it semi-automatically using two clever tricks stacked on top of each other:

1. **Semantic search** to find *candidate* repos for each industry (fast but noisy)
2. **GPT-4 validation** to score each candidate (slow but accurate)

Think of it like hiring: Step 1 is the resume screen (quick, casts a wide net), and Step 2 is the actual interview (expensive, but gives you real signal).

---

## The Technical Architecture (How the Machine Works)

The pipeline is a clean 5-step assembly line. Each step produces files on disk that the next step consumes. No databases, no servers -- just scripts, files, and APIs.

```
Raw Repos (parquet)
      |
      v
[Step 1] BGE Embeddings -- turn README text into 1024-dim vectors
      |
      v
[Step 2] FAISS Index -- build a searchable index of those vectors
      |
      v
[Step 3] Semantic Search -- for each NAICS sector, find the 400 closest repos
      |
      v
[Step 4] GPT-4 Classification -- ask GPT to score each repo-sector pair (1-10)
      |
      v
[Step 5] Filter -- keep only score >= 8, deduplicate, export
      |
      v
Training Dataset (CSV/Parquet)
```

Each step is one Python script in `scripts/`, numbered `01` through `05`. You run them in order. If one crashes, you restart just that step -- the previous steps' outputs are already on disk.

---

## The Codebase Structure

```
naics-github-classifier/
├── config/config.yaml          # All tunable knobs in one file
├── data/
│   ├── raw/                    # Input: parquet repo dumps + NAICS definitions JSON
│   ├── processed/              # Intermediate: embeddings, FAISS index, candidate pairs
│   └── output/                 # Final: filtered training data + batch CSVs
├── src/                        # The actual library code
│   ├── data/                   # loader.py (read parquet/JSON), preprocessor.py (clean text)
│   ├── embeddings/             # bge_encoder.py (encode text), faiss_indexer.py (search)
│   ├── classification/         # gpt_classifier.py (call GPT), prompt_builder.py, batch_processor.py
│   └── utils/                  # logger.py, token_counter.py
├── scripts/                    # The 5 pipeline scripts (01-05)
├── notebooks/                  # Jupyter exploration (placeholder)
├── figures/                    # Generated charts
└── tests/                      # Unit tests (placeholder)
```

The separation between `src/` (library) and `scripts/` (executables) is intentional. The scripts are thin -- they parse CLI args, load config, and call into `src/` modules. All the real logic lives in `src/`. This means you can also `import` these modules from a Jupyter notebook or a different script without duplicating code.

---

## How the Parts Connect

### Step 1: Turning Text Into Numbers (`bge_encoder.py`)

A README is just text. FAISS needs numbers. So we use **BAAI/bge-large-en**, an embedding model that converts text into a 1024-dimensional vector. Two texts that are "about the same thing" will have vectors that point in similar directions (high cosine similarity).

The trick is in the preprocessing (`preprocessor.py`): we strip code blocks, HTML, URLs, and other noise from READMEs before embedding. A README full of `<div>` tags and base64-encoded images is useless for understanding what a project *does*. We also combine the repo's description, topics, and cleaned README into one text blob -- more signal, better embeddings.

**Analogy:** This is like creating a fingerprint for each repo. The fingerprint captures "what this repo is about" in a way that's mathematically comparable.

### Step 2: Making Search Fast (`faiss_indexer.py`)

With 531K embeddings, you can't compare every repo to every query by brute force (well, you can, but FAISS makes it orders of magnitude faster). We build an `IndexFlatIP` index -- "flat" means exact search (no approximation), "IP" means inner product (which equals cosine similarity when vectors are L2-normalized).

**Why not approximate search?** At 531K vectors, exact search with FAISS is still fast enough (seconds per query). Approximate methods (IVF, HNSW) add complexity and tuning overhead without a meaningful speedup at this scale.

### Step 3: Finding Candidates (`03_retrieve_by_naics.py`)

This is the clever part. For each NAICS sector (like "11 - Agriculture"), we have a definition string like:

> "Soybean Farming; Oilseed Farming; Wheat Farming; Corn Farming; Rice Farming..."

The script splits this by semicolons to get individual subindustries ("Soybean Farming", "Oilseed Farming", etc.), appends "software tools applications" to each one, encodes the query with BGE, and searches the FAISS index for the closest repos.

**Why split by subindustry?** Because "Agriculture" as a single query is too vague -- it might miss repos about specific niches like aquaculture or forestry equipment. By querying each subindustry separately, we cast a wider, more targeted net. We then retrieve ~20 repos per subindustry query, targeting ~400 total per sector.

The output is ~4.3 million repo-NAICS candidate pairs. Most of these will be junk -- the semantic search is intentionally loose. That's fine. GPT will sort the wheat from the chaff in Step 4.

### Step 4: The Expensive Part (`gpt_classifier.py`)

For each candidate pair, we send GPT-4 a prompt that says (paraphrasing):

> "Here's a GitHub repo README. Here's a NAICS sector definition. On a scale of 1-10, how relevant is this repo to this industry? Return JSON with match (bool), score (1-10), and rationale."

The prompt in `prompt_builder.py` is carefully structured with scoring rubrics (9-10 = core industry software, 1-2 = no connection) so GPT gives consistent scores.

**Dual API support** is a nice engineering touch: the classifier auto-detects whether you have an OpenAI API key or a GitHub Copilot token, and uses whichever is available. Different endpoints, different default models (gpt-4-turbo vs gpt-4.1), same interface.

**Resumability** is critical here. At 4.3M API calls, this step takes days or weeks. The batch processor saves results every 500 repos to CSV files (`batch_0000.csv`, `batch_0001.csv`, ...). If the process crashes at repo 250,000, you restart with `--start-idx 250000` and lose nothing.

### Step 5: Filtering (`05_filter_results.py`)

Load all batch CSVs, keep only score >= 8, deduplicate (if a repo matched multiple NAICS sectors, keep the highest-scored one), and export the final training dataset. Also generates some matplotlib/seaborn charts for sanity checking.

---

## Technologies Used and Why

| Technology | What It Does Here | Why This Choice |
|---|---|---|
| **BAAI/bge-large-en** | Text embeddings (1024-dim) | One of the best open-source embedding models for English retrieval tasks. Strong performance on MTEB benchmarks. |
| **FAISS** | Vector similarity search | Facebook's gold standard for similarity search. GPU support, battle-tested at billion-scale. |
| **GPT-4** | Classification scoring | Best available LLM for nuanced judgment tasks. The scoring rubric approach works well with instruction-following models. |
| **Parquet** | Data storage | Columnar format = fast reads, good compression, schema enforcement. Way better than CSV for 500K+ rows. |
| **PyYAML** | Configuration | One config file, no hardcoded paths or parameters scattered across scripts. |
| **tiktoken** | Token counting | Accurate token counts for cost estimation and README truncation (3K token limit for GPT input). |
| **python-dotenv** | Secrets management | API keys stay in `.env`, not in code or config files. |

---

## Technical Decisions Worth Understanding

### Why Semantic Search + LLM Instead of Just LLM?

Sending 531K repos to GPT-4 would cost ~$16M at current rates. Semantic search narrows the candidates to ~4.3M *pairs* (still expensive, but 100x cheaper than the naive approach of checking every repo against every sector). The semantic search is free (runs locally) and fast.

**Lesson:** Always think about the cost structure of your pipeline. When you have an expensive step (API calls), invest in cheap pre-filtering to reduce the input volume.

### Why Score 1-10 Instead of Just Yes/No?

A boolean classifier would lose information. With a continuous score, you can adjust the threshold later. Started with score >= 7 and getting too much noise? Bump it to 8. Need more training data? Drop to 6. The score gives you a knob to turn without re-running the expensive GPT step.

**Lesson:** When you're generating data, preserve as much signal as possible. Downstream filtering is cheap; re-generation is expensive.

### Why Dual API Support?

This project was built at GitHub, where engineers have access to the GitHub Copilot API (free, different endpoint). External collaborators use the OpenAI API. Rather than maintaining two codebases, the classifier auto-detects which key is available and adjusts the endpoint and default model.

**Lesson:** Design for your actual users. If two groups need different auth mechanisms, abstract the difference away behind a common interface.

### Why Batch CSVs Instead of a Database?

Simplicity. CSVs are debuggable (open in Excel, grep for a repo name), portable (no DB setup), and the batch numbering gives you free checkpointing. A database would add dependency complexity for no real benefit at this scale.

**Lesson:** Don't introduce infrastructure you don't need. Files on disk are a perfectly valid "database" for batch pipelines.

---

## Lessons, Pitfalls, and Hard-Won Wisdom

### 1. README Preprocessing Is Everything

The single most impactful part of this pipeline isn't the fancy embedding model or GPT-4 -- it's the `preprocessor.py` that strips code blocks, HTML, and URLs from READMEs before embedding.

**Why it matters:** A README that's 80% HTML badge markup and webpack config examples will embed as "web development tooling" regardless of what the project actually does. Cleaning the text first means the embeddings actually capture the project's *purpose*, not its *boilerplate*.

**Pitfall to avoid:** Don't skip preprocessing and assume the model will "figure it out." Garbage in, garbage out, no matter how good the model is.

### 2. Token Budgets Are a Hard Constraint

GPT-4 has a context window. READMEs can be enormous (some are 50K+ characters). The pipeline truncates to 3,000 tokens using tiktoken. This is a design tradeoff: more context = better classification, but also higher cost and slower inference.

**Pitfall:** If you truncate too aggressively, you might cut off the part of the README that actually explains what industry the project serves (often buried below installation instructions). The current approach of cleaning *then* truncating helps because it removes low-signal content first.

### 3. Rate Limits Will Humble You

At 4.3M API calls, you *will* hit rate limits. The retry logic (exponential backoff, max 5 retries, special handling for HTTP 429) isn't optional -- it's essential. Without it, your pipeline dies at 3 AM and you lose a night of processing.

**Best practice:** Always build retry logic for any API-dependent pipeline. And always save intermediate results. The batch-save-every-500 pattern is simple but saves you from losing hours of work.

### 4. The Resume Pattern

The `--start-idx` flag on the classification script is a small feature with huge impact. Long-running pipelines *will* crash -- network blips, API outages, OOM errors, laptop goes to sleep. The ability to pick up where you left off turns a catastrophic failure into a minor inconvenience.

**How good engineers think:** Before writing a long-running process, ask "what happens when this crashes at 60% completion?" If the answer is "start over from scratch," add checkpointing.

### 5. Config as Code

Every tunable parameter lives in `config/config.yaml` -- batch sizes, model names, file paths, thresholds, token limits. No magic numbers in the scripts. This means:
- You can reproduce any run by checking the config file
- Changing a parameter doesn't require reading code
- Different team members can use different configs without merge conflicts

**Lesson:** If you find yourself editing a Python file to change a threshold, that threshold should be in a config file.

### 6. The Auto-Detection Pattern

`gpt_classifier.py` checks for `OPENAI_API_KEY` first, then `GITHUB_TOKEN`, and uses whichever it finds. This is better than requiring a `--backend` flag because it "just works" for most users. The explicit flag is still there as an override.

**How good engineers think:** Optimize for the common case (auto-detect), but always provide an escape hatch (explicit flag).

### 7. Cost Awareness

The `token_counter.py` utility exists specifically so you can estimate costs *before* running the full pipeline. At ~$0.03 per 1K input tokens for GPT-4, 4.3M classifications adds up fast. Running on the 5% sample first (26.5K repos) is the responsible way to validate the approach before committing real money.

**Lesson:** Always estimate costs before scaling up. Run on a sample first. This applies to API costs, compute costs, and your own time.

---

## How the Pipeline Thinks About Scale

| Dataset | Repos | Candidate Pairs | Estimated GPT Cost |
|---|---|---|---|
| 5% sample | 26.5K | ~215K | ~$6.5K |
| EU full | 531K | ~4.3M | ~$130K |
| AU | 331K | ~2.7M | ~$80K |
| India | 1M | ~8M | ~$240K |

The pipeline was designed to work at the 531K scale first, then scale to multi-million. The file-based architecture (parquet in, parquet/CSV out) means you don't need to change anything to run on a bigger dataset -- just point to a different input file in the config.

---

## The Prompt Engineering (The Secret Sauce)

The classification prompt (`prompt_builder.py`) is deceptively important. It doesn't just ask "is this repo related to agriculture?" It provides:

1. The full NAICS sector definition (so GPT knows what "sector 11" actually means)
2. A scoring rubric (9-10 = core industry software, 1-2 = no connection)
3. A structured output format (JSON with match, score, rationale)
4. Instructions to be conservative (when in doubt, score lower)

The `temperature=0.1` setting makes GPT nearly deterministic -- you want consistent scoring, not creative writing. And the `max_tokens=200` cap prevents GPT from rambling in the rationale field.

**Lesson:** Prompt engineering for data pipelines is different from prompt engineering for chatbots. You want *consistency* and *parseable output*, not eloquence. JSON output format + low temperature + clear rubric = reliable structured data.

---

## What Makes This Codebase Well-Structured

1. **Separation of concerns**: `src/` is the library, `scripts/` are the entry points. Each module does one thing.
2. **No hardcoded paths**: Everything comes from `config.yaml` or CLI args.
3. **Defensive I/O**: Batch saving, retry logic, resume capability. The code assumes things will go wrong.
4. **Progressive refinement**: The pipeline narrows from 531K repos -> 4.3M candidates -> filtered training set. Each step reduces noise.
5. **Dual API support**: One interface, two backends. The calling code doesn't know or care which API it's talking to.

---

## Quick Reference: Running the Full Pipeline

```bash
# 1. Setup
pip install -r requirements.txt && pip install -e .
echo "OPENAI_API_KEY=sk-..." > .env

# 2. Generate embeddings (GPU recommended)
python scripts/01_generate_embeddings.py --input data/raw/repos_eu_sample_5pct.parquet

# 3. Build search index
python scripts/02_build_faiss_index.py

# 4. Find candidates per NAICS sector
python scripts/03_retrieve_by_naics.py

# 5. Classify with GPT (the slow, expensive step)
python scripts/04_classify_repos.py

# 6. Filter to high-confidence labels
python scripts/05_filter_results.py --min-score 8
```

---

## Final Thought

This project is a good example of a pattern that comes up constantly in ML engineering: **you rarely have labeled data, so you build a pipeline to generate it.** The pipeline itself is more engineering than science -- file I/O, API wrangling, error handling, cost management. The "AI" parts (embeddings, GPT classification) are almost plug-and-play. The hard part is everything around them: making the pipeline reliable, resumable, cost-effective, and producing clean output.

That's the real lesson here. The models are the easy part. The plumbing is where the work is.
