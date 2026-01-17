"""Batch processing for repository classification."""

import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .gpt_classifier import GPTClassifier
from ..data.preprocessor import clean_readme, truncate_to_tokens
from ..utils.logger import setup_logger


class BatchProcessor:
    """
    Process repositories in batches for classification.

    Attributes:
        classifier: GPTClassifier instance
        output_dir: Directory for batch CSV outputs
        batch_size: Number of repos per batch file
    """

    def __init__(
        self,
        classifier: Optional[GPTClassifier] = None,
        output_dir: str = "data/output/batch_csvs",
        batch_size: int = 500,
        max_tokens: int = 3000
    ):
        """
        Initialize batch processor.

        Args:
            classifier: GPTClassifier instance (creates new if None)
            output_dir: Directory for output files
            batch_size: Repos per batch file
            max_tokens: Max tokens for README content
        """
        self.classifier = classifier or GPTClassifier()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.logger = setup_logger("batch_processor")

    def process_dataframe(
        self,
        df: pd.DataFrame,
        readme_col: str = "readme_content",
        naics_col: str = "naics_code",
        naics_def_col: str = "naics_definition",
        repo_url_col: str = "repo_url",
        start_idx: int = 0
    ) -> pd.DataFrame:
        """
        Process a DataFrame of repositories.

        Args:
            df: DataFrame with repos to classify
            readme_col: Column name for README content
            naics_col: Column name for NAICS code
            naics_def_col: Column name for NAICS definition
            repo_url_col: Column name for repository URL
            start_idx: Starting index (for resuming)

        Returns:
            DataFrame with classification results
        """
        results = []
        batch_num = start_idx // self.batch_size

        for idx in tqdm(range(start_idx, len(df)), desc="Classifying repos"):
            row = df.iloc[idx]

            # Preprocess README
            readme = row.get(readme_col, "")
            if readme:
                readme = clean_readme(str(readme))
                readme = truncate_to_tokens(readme, self.max_tokens)

            if not readme:
                results.append({
                    "repo_url": row.get(repo_url_col, ""),
                    "naics_code": row.get(naics_col, ""),
                    "match": False,
                    "score": 0,
                    "rationale": "Empty README"
                })
                continue

            # Classify
            success, result = self.classifier.classify(
                readme_content=readme,
                naics_code=str(row.get(naics_col, "")),
                naics_definition=str(row.get(naics_def_col, ""))
            )

            results.append({
                "repo_url": row.get(repo_url_col, ""),
                "naics_code": row.get(naics_col, ""),
                "match": result.get("match", False) if success else False,
                "score": result.get("score", 0) if success else 0,
                "rationale": result.get("rationale", result.get("error", ""))
            })

            # Save batch
            if len(results) >= self.batch_size:
                self._save_batch(results, batch_num)
                results = []
                batch_num += 1

        # Save remaining
        if results:
            self._save_batch(results, batch_num)

        return self._load_all_batches()

    def _save_batch(self, results: List[Dict], batch_num: int) -> None:
        """Save a batch of results to CSV."""
        batch_df = pd.DataFrame(results)
        batch_path = self.output_dir / f"batch_{batch_num:04d}.csv"
        batch_df.to_csv(batch_path, index=False)
        self.logger.info(f"Saved batch {batch_num} to {batch_path}")

    def _load_all_batches(self) -> pd.DataFrame:
        """Load and concatenate all batch files."""
        batch_files = sorted(self.output_dir.glob("batch_*.csv"))
        if not batch_files:
            return pd.DataFrame()

        dfs = [pd.read_csv(f) for f in batch_files]
        return pd.concat(dfs, ignore_index=True)
