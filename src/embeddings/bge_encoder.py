"""BGE Embedding Encoder for semantic search."""

from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class BGEEncoder:
    """
    Encoder using BAAI/bge-large-en model for semantic embeddings.

    Attributes:
        model: SentenceTransformer model
        device: torch device (cuda/cpu)
        query_prefix: Prefix for query encoding (BGE recommendation)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en",
        device: Optional[str] = None,
        use_half_precision: bool = True
    ):
        """
        Initialize BGE encoder.

        Args:
            model_name: Hugging Face model name
            device: Device to use (auto-detected if None)
            use_half_precision: Use float16 for efficiency
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

        if use_half_precision and device == "cuda":
            self.model.half()

        self.query_prefix = "Represent this document for retrieval: "

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 1024,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Normalize embeddings for cosine similarity

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Add BGE prefix for better retrieval
        texts_with_prefix = [self.query_prefix + t for t in texts]

        embeddings = self.model.encode(
            texts_with_prefix,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embeddings

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query for search.

        Args:
            query: Query text
            normalize: Normalize embedding

        Returns:
            Query embedding
        """
        return self.encode([query], batch_size=1, show_progress=False, normalize=normalize)[0]
