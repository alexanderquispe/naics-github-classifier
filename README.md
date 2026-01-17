# NAICS GitHub Repository Classifier

Automated classification of GitHub repositories into NAICS (North American Industry Classification System) codes using semantic search and LLM-powered validation.

## Overview

This project builds a training dataset for the ModernBERT NAICS classifier by:
1. **Semantic Search**: Using BGE embeddings + FAISS to find repositories relevant to each NAICS sector
2. **LLM Classification**: Using GPT-4 (via OpenAI API or GitHub Copilot API) to validate and score classifications
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

2. Add your API credentials to `.env`. Choose one of:

**Option A: OpenAI API** (for external users)
```
OPENAI_API_KEY=your_openai_api_key_here
```
Get your API key from: https://platform.openai.com/api-keys

**Option B: GitHub Copilot API** (for GitHub employees)
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
# Auto-detect backend based on available API key
python scripts/04_classify_repos.py

# Or explicitly specify backend
python scripts/04_classify_repos.py --backend openai
python scripts/04_classify_repos.py --backend github

# Optionally specify model
python scripts/04_classify_repos.py --backend openai --model gpt-4o
```

### Step 5: Filter Results
```bash
python scripts/05_filter_results.py --min-score 8
```

## Input Data Requirements

The pipeline requires two input files in `data/raw/`:

### 1. Repository Dataset (Parquet)

A parquet file containing GitHub repository metadata. **Required columns:**

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `nwo` | string | Repository identifier in "owner/repo" format (e.g., "microsoft/vscode") | **Yes** |
| `readme_content` | string | Full README file content (markdown/text) | **Yes** |
| `description` | string | Repository description from GitHub | Recommended |
| `topics` | list/string | Repository topics/tags | Recommended |

**Optional columns** (used for filtering/analysis):
- `num_stars` - Star count
- `is_fork` - Whether repo is a fork
- `is_archived` - Whether repo is archived
- `spdx_license` - License type
- `num_commits` - Commit count
- `last_commit_at` - Last commit timestamp

**Example structure:**
```
nwo                     | description                          | readme_content
------------------------|--------------------------------------|------------------
microsoft/vscode        | Visual Studio Code                   | # Visual Studio Code...
tensorflow/tensorflow   | An Open Source ML Framework          | # TensorFlow...
```

**Sample file:** `repos_eu_sample_5pct.parquet` (26,573 repos, 69MB)

### 2. NAICS Definitions (JSON)

A JSON file mapping 2-digit NAICS sector codes to their industry descriptions.

**Format:**
```json
{
    "11": "Soybean Farming; Oilseed Farming; Wheat Farming; Corn Farming; ...",
    "21": "Crude Petroleum Extraction; Natural Gas Extraction; Coal Mining; ...",
    "22": "Electric Power Generation; Natural Gas Distribution; Water Supply; ...",
    ...
}
```

**NAICS Sectors (20 total):**

| Code | Sector Name |
|------|-------------|
| 11 | Agriculture, Forestry, Fishing and Hunting |
| 21 | Mining, Quarrying, and Oil and Gas Extraction |
| 22 | Utilities |
| 23 | Construction |
| 31-33 | Manufacturing |
| 42 | Wholesale Trade |
| 44-45 | Retail Trade |
| 48-49 | Transportation and Warehousing |
| 51 | Information |
| 52 | Finance and Insurance |
| 53 | Real Estate and Rental and Leasing |
| 54 | Professional, Scientific, and Technical Services |
| 55 | Management of Companies and Enterprises |
| 56 | Administrative and Support Services |
| 61 | Educational Services |
| 62 | Health Care and Social Assistance |
| 71 | Arts, Entertainment, and Recreation |
| 72 | Accommodation and Food Services |
| 81 | Other Services |
| 92 | Public Administration |

**Sample file:** `naics_titles_by_group_6digit_clean.json`

### Preparing Your Own Data

If you want to use your own repository dataset:

1. **Export to parquet** with at minimum `nwo` and `readme_content` columns
2. **Clean README content** - remove binary/corrupted entries
3. **Filter repositories** - remove forks, archived repos, or those without READMEs if desired

```python
import pandas as pd

# Example: Create input file from your data
df = pd.DataFrame({
    'nwo': ['owner/repo1', 'owner/repo2'],
    'description': ['A cool project', 'Another project'],
    'topics': [['python', 'ml'], ['web', 'api']],
    'readme_content': ['# Project 1\nThis is...', '# Project 2\nThis does...']
})
df.to_parquet('data/raw/my_repos.parquet', index=False)
```

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

## Output Data Format

### Classification Results (CSV)

The final output `data/output/naics_training_data_filtered.csv` contains:

| Column | Type | Description |
|--------|------|-------------|
| `repo_url` | string | Full GitHub URL (e.g., "https://github.com/owner/repo") |
| `naics_code` | string | Matched NAICS sector code (e.g., "11", "54") |
| `match` | boolean | Whether GPT determined this is a true match |
| `score` | integer | Confidence score (1-10) |
| `rationale` | string | GPT's explanation for the classification |

**Example output:**
```csv
repo_url,naics_code,match,score,rationale
https://github.com/farm-ng/farm-ng-core,11,True,9,"Core robotics framework for agricultural automation..."
https://github.com/openag/openag_brain,11,True,8,"Software for controlled environment agriculture..."
```

**Scoring interpretation:**
- **9-10**: Core industry software with direct sector application
- **7-8**: Strong sector relevance with clear industry use cases
- **5-6**: Moderate sector connection
- **3-4**: Weak sector relevance
- **1-2**: No meaningful sector connection

### Intermediate Files

| File | Description |
|------|-------------|
| `data/processed/embeddings/repo_embeddings_vectors.npy` | BGE embeddings (N x 1024 float32) |
| `data/processed/embeddings/repo_embeddings_metadata.parquet` | Repository metadata with input text |
| `data/processed/faiss_indices/repos.index` | FAISS index for similarity search |
| `data/processed/retrieved_repos_by_naics.parquet` | Candidate repo-NAICS pairs from semantic search |
| `data/output/batch_csvs/batch_*.csv` | Raw classification results (batched) |

## Key Dependencies

- `torch` - PyTorch for GPU support
- `sentence-transformers` - BGE embedding model (BAAI/bge-large-en)
- `faiss-cpu` - Vector similarity search
- `tiktoken` - Token counting for LLM requests
- `pandas` / `pyarrow` - Data handling

## License

MIT License

## Author

Alexander Quispe - Caltech / GitHub
