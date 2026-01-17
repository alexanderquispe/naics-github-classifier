# NAICS GitHub Repository Classifier

Automated classification of GitHub repositories into NAICS (North American Industry Classification System) sectors using semantic search and LLM-powered validation.

## Overview

This project builds a training dataset for a NAICS classifier by:

1. **Semantic Search**: Using BGE embeddings + FAISS to find repositories relevant to each NAICS sector
2. **LLM Classification**: Using GPT-4 to validate and score whether repositories truly belong to their matched sectors
3. **Dataset Creation**: Filtering high-confidence predictions (score >= 7) for training data

## Project Structure

```
naics-github-classifier/
├── config/                 # Configuration files
├── data/
│   ├── raw/               # Input data (repos parquet, NAICS definitions)
│   ├── processed/         # Intermediate outputs (embeddings, FAISS indices)
│   └── output/            # Final classification results
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── embeddings/        # BGE encoder and FAISS indexing
│   ├── classification/    # GPT classification pipeline
│   └── utils/             # Utilities (logging, token counting)
├── scripts/               # Pipeline execution scripts (01-05)
├── figures/               # Generated visualizations
└── tests/                 # Unit tests
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended for embeddings)
- ~16GB GPU memory for batch processing
- OpenAI API key OR GitHub Copilot API access

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/naics-github-classifier.git
cd naics-github-classifier
```

### Step 2: Create Conda Environment

```bash
conda create -n naics-classifier python=3.10 -y
conda activate naics-classifier
```

### Step 3: Install PyTorch with CUDA

```bash
# For CUDA 11.8 (check your CUDA version with: nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install Dependencies

```bash
pip install sentence-transformers faiss-cpu tiktoken tqdm pandas pyarrow python-dotenv requests pyyaml matplotlib seaborn
```

### Step 5: Verify GPU Setup

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Configuration

### API Setup

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

**Option A: OpenAI API (General Public)**

If you have an OpenAI API key, add it to `.env`:

```
OPENAI_API_KEY=sk-your-openai-api-key-here
```

Get your API key from: https://platform.openai.com/api-keys

**Option B: GitHub Copilot API (GitHub Employees)**

If you have access to GitHub's internal Copilot API:

```
GITHUB_TOKEN=your-github-token-here
```

The classifier auto-detects which API to use based on available environment variables (OpenAI takes priority if both are set).

## Pipeline Overview

The pipeline consists of 5 sequential steps:

| Step | Script | Description | Time Estimate |
|------|--------|-------------|---------------|
| 1 | `01_generate_embeddings.py` | Convert README text to BGE vectors | ~10 min (GPU) |
| 2 | `02_build_faiss_index.py` | Build searchable vector index | ~1 min |
| 3 | `03_retrieve_by_naics.py` | Find candidate repos for each NAICS sector | ~2 min |
| 4 | `04_classify_repos.py` | GPT-4 validates each repo-NAICS pair | ~5 sec/repo |
| 5 | `05_filter_results.py` | Filter high-confidence results | ~1 min |

## Running the Pipeline

### Quick Start (Test Run)

First, verify everything works with a small test:

```bash
conda activate naics-classifier
cd /path/to/naics-github-classifier

# Step 1: Generate embeddings
python scripts/01_generate_embeddings.py \
    --input data/raw/repos_eu_sample_5pct.parquet \
    --output data/processed/embeddings/repo_embeddings.parquet \
    --batch-size 64

# Step 2: Build FAISS index
python scripts/02_build_faiss_index.py \
    --vectors data/processed/embeddings/repo_embeddings_vectors.npy \
    --embeddings data/processed/embeddings/repo_embeddings_metadata.parquet \
    --output-index data/processed/faiss_indices/repos.index

# Step 3: Retrieve repos by NAICS
python scripts/03_retrieve_by_naics.py \
    --index data/processed/faiss_indices/repos.index \
    --metadata data/processed/embeddings/repo_embeddings_metadata.parquet \
    --naics data/raw/naics_titles_by_group_6digit_clean.json \
    --output data/processed/retrieved_repos_by_naics.parquet

# Step 4: Classify with GPT (test with 50 repos)
python scripts/04_classify_repos.py \
    --input data/processed/retrieved_repos_by_naics.parquet \
    --output-dir data/output/batch_csvs \
    --limit 50

# Step 5: Filter results
python scripts/05_filter_results.py \
    --input-dir data/output/batch_csvs \
    --output data/output/naics_training_data_filtered.csv \
    --min-score 7
```

### Full Classification Run

To classify all retrieved repos (can take several hours):

```bash
# Remove --limit to process all repos
python scripts/04_classify_repos.py \
    --input data/processed/retrieved_repos_by_naics.parquet \
    --output-dir data/output/batch_csvs
```

**Notes:**
- Results are saved in batches of 500 repos
- If interrupted, use `--start-idx` to resume from where you left off
- Progress is logged to `data/output/classification_log.txt`

### Selecting API Backend

You can explicitly select which API to use:

```bash
# Use OpenAI API
python scripts/04_classify_repos.py --backend openai --input data/processed/retrieved_repos_by_naics.parquet

# Use GitHub Copilot API
python scripts/04_classify_repos.py --backend github --input data/processed/retrieved_repos_by_naics.parquet

# Specify a different model
python scripts/04_classify_repos.py --backend openai --model gpt-4o-mini --input data/processed/retrieved_repos_by_naics.parquet
```

## Output Files

After running the pipeline, you'll have:

| File | Description |
|------|-------------|
| `data/processed/embeddings/` | BGE embeddings (vectors + metadata) |
| `data/processed/faiss_indices/repos.index` | FAISS vector index |
| `data/processed/retrieved_repos_by_naics.parquet` | Candidate repo-NAICS pairs |
| `data/output/batch_csvs/` | Classification results (batched CSVs) |
| `data/output/naics_training_data_filtered.csv` | Final training dataset |
| `figures/` | Score distribution and NAICS visualizations |

## Classification Output Format

Each classified repo has:

```json
{
    "repo_url": "https://github.com/user/repo",
    "naics_code": "11",
    "match": true,
    "score": 8,
    "rationale": "This repository implements farm management software..."
}
```

**Scoring Guide:**
- **9-10**: Core industry software with direct sector application
- **7-8**: Strong sector relevance with clear industry use cases
- **5-6**: Moderate sector connection
- **3-4**: Weak sector relevance
- **1-2**: No meaningful sector connection

## Data Sources

- **Repository Dataset**: GitHub repositories with README content (`repos_eu_sample_5pct.parquet` - 26,573 repos)
- **NAICS Definitions**: 20 NAICS sectors with 6-digit industry descriptions (`naics_titles_by_group_6digit_clean.json`)

## Pipeline Architecture

```
┌─────────────────────┐
│  Raw Repositories   │
│  (parquet file)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Step 1: BGE        │
│  Embeddings         │
│  (1024-dim vectors) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Step 2: FAISS      │
│  Index              │
│  (vector search)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Step 3: Semantic   │
│  Retrieval          │
│  (per NAICS sector) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Step 4: GPT-4      │
│  Classification     │
│  (validate matches) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Step 5: Filter     │
│  Score >= 7         │
│  (training data)    │
└─────────────────────┘
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in Step 1:

```bash
python scripts/01_generate_embeddings.py --batch-size 32 --input ...
```

### API Rate Limits

The classifier handles rate limits automatically with exponential backoff. If you hit persistent limits, reduce request frequency or use a different model.

### Missing Dependencies

```bash
pip install sentence-transformers faiss-cpu tiktoken tqdm pandas pyarrow python-dotenv requests
```

### File Not Found Errors

Make sure to run scripts from the project root directory, or use absolute paths.

## License

MIT License

## Author

Alexander Quispe - Caltech / GitHub
