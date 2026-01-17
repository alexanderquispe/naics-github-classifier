"""FAISS Index management for similarity search."""

from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np
import pandas as pd


class FAISSIndexer:
    """
    FAISS index manager for efficient similarity search.

    Attributes:
        index: FAISS index
        metadata: DataFrame with repository metadata
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize FAISS indexer.

        Args:
            use_gpu: Use GPU for index operations
        """
        self.use_gpu = use_gpu
        self.index = None
        self.metadata = None

    def build_index(self, embeddings: np.ndarray, normalize: bool = True) -> None:
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: Numpy array of embeddings (n_samples, dim)
            normalize: Normalize embeddings for cosine similarity
        """
        if normalize:
            faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]

        # Use Inner Product for cosine similarity (on normalized vectors)
        self.index = faiss.IndexFlatIP(dim)

        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.index.add(embeddings.astype(np.float32))

    def set_metadata(self, metadata: pd.DataFrame) -> None:
        """
        Set metadata DataFrame for result retrieval.

        Args:
            metadata: DataFrame with repository metadata (indexed by position)
        """
        self.metadata = metadata.reset_index(drop=True)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar items.

        Args:
            query_embedding: Query embedding (1D or 2D array)
            k: Number of results to return

        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)

        return distances[0], indices[0]

    def search_with_metadata(
        self,
        query_embedding: np.ndarray,
        k: int = 100
    ) -> pd.DataFrame:
        """
        Search and return results with metadata.

        Args:
            query_embedding: Query embedding
            k: Number of results

        Returns:
            DataFrame with search results and metadata
        """
        distances, indices = self.search(query_embedding, k)

        if self.metadata is None:
            return pd.DataFrame({"index": indices, "score": distances})

        results = self.metadata.iloc[indices].copy()
        results["similarity_score"] = distances

        return results

    def save(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Save index and optionally metadata.

        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata parquet
        """
        # Move to CPU before saving
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index

        faiss.write_index(cpu_index, str(index_path))

        if metadata_path and self.metadata is not None:
            self.metadata.to_parquet(metadata_path)

    def load(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Load index and optionally metadata.

        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata parquet
        """
        self.index = faiss.read_index(str(index_path))

        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        if metadata_path:
            self.metadata = pd.read_parquet(metadata_path)
