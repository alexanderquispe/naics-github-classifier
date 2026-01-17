"""Data loading functions for repositories and NAICS definitions."""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_repos(filepath: str, columns: List[str] = None) -> pd.DataFrame:
    """
    Load repository dataset from parquet file.

    Args:
        filepath: Path to parquet file
        columns: Optional list of columns to load

    Returns:
        DataFrame with repository data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Repository file not found: {filepath}")

    if columns:
        df = pd.read_parquet(filepath, columns=columns)
    else:
        df = pd.read_parquet(filepath)

    return df


def load_naics_definitions(filepath: str) -> Dict[str, str]:
    """
    Load NAICS sector definitions from JSON file.

    Args:
        filepath: Path to JSON file with NAICS definitions

    Returns:
        Dictionary mapping NAICS codes to semicolon-separated descriptions
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"NAICS definitions file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        naics_data = json.load(f)

    return naics_data


def load_embeddings(filepath: str) -> pd.DataFrame:
    """
    Load pre-computed embeddings from parquet file.

    Args:
        filepath: Path to parquet file with embeddings

    Returns:
        DataFrame with embeddings column
    """
    return pd.read_parquet(filepath)
