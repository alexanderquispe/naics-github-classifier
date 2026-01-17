# NAICS GitHub Repository Classifier

Automated classification of GitHub repositories into NAICS (North American Industry Classification System) codes using semantic search and LLM-powered validation.

## Overview

This project builds a training dataset for the ModernBERT NAICS classifier by:
1. **Semantic Search**: Using BGE embeddings + FAISS to find repositories relevant to each NAICS sector
2. **LLM Classification**: Using GPT-4 via GitHub Copilot API to validate and score classifications
3. **Dataset Creation**: Filtering high-confidence predictions for training data

## Project Structure

```
naics-github-classifier/
├── config/                 # Configuration files
├── data/
│   ├── raw/               # Original input data
│   ├── processed/         # Intermediate outputs (embeddings, indices)
│   └── output/            # Final training datasets
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── embeddings/        # BGE encoder and FAISS indexing
│   ├── classification/    # GPT classification pipeline
│   └── utils/             # Utilities (logging, token counting)
├── scripts/               # Pipeline execution scripts
├── notebooks/             # Jupyter notebooks for analysis
├── figures/               # Generated visualizations
└── tests/                 # Unit tests
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/naics-github-classifier.git
cd naics-github-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Add your API credentials to `.env`:
```
GITHUB_TOKEN=your_github_token_here
```

3. Adjust parameters in `config/config.yaml` as needed.

## Usage

### Step 1: Generate Embeddings
```bash
python scripts/01_generate_embeddings.py --input data/raw/repos_eu_531k.parquet
```

### Step 2: Build FAISS Index
```bash
python scripts/02_build_faiss_index.py
```

### Step 3: Retrieve Repositories by NAICS
```bash
python scripts/03_retrieve_by_naics.py
```

### Step 4: Classify with GPT
```bash
python scripts/04_classify_repos.py
```

### Step 5: Filter Results
```bash
python scripts/05_filter_results.py --min-score 8
```

## Data Sources

- **Repository Dataset**: European GitHub repositories (531k repos)
- **NAICS Definitions**: 6-digit NAICS codes with sector descriptions

## Pipeline Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Raw Repos      │────▶│  BGE Embeddings │────▶│  FAISS Index    │
│  (531k parquet) │     │  (vectors)      │     │  (search)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Training Data  │◀────│  Filter Score≥8 │◀────│  GPT-4 Classify │
│  (CSV output)   │     │  (validation)   │     │  (API calls)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Key Dependencies

- `torch` - PyTorch for GPU support
- `sentence-transformers` - BGE embedding model
- `faiss-gpu` - Vector similarity search
- `transformers` - Hugging Face models
- `tiktoken` - Token counting for LLM requests

## License

MIT License

## Author

Alexander Quispe - Caltech / GitHub
