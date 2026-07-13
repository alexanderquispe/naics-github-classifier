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
        doc_prefix: Prefix prepended to documents at indexing time
        query_prefix: Prefix prepended to queries at search time
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
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

        # BGE uses DIFFERENT prefixes for documents (the indexed corpus)
        # and queries (search prompts). Mixing them at retrieval time
        # silently degrades cosine-similarity quality because the encoder
        # was trained to project the two prefixes into asymmetric spaces.
        # The original Colab notebooks had this correct (cells 14 + 28 of
        # 1_eu_index_metadata_creation.ipynb); the local refactor regressed.
        self.doc_prefix = "Represent this document for retrieval: "
        self.query_prefix = "Represent this query for retrieval: "

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 1024,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode documents (the indexed corpus). Uses the document prefix.

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

        texts_with_prefix = [self.doc_prefix + t for t in texts]

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
        Encode a single search query. Uses the QUERY prefix (not the document
        prefix). Bypasses self.encode() to avoid double-prefixing.

        Args:
            query: Query text
            normalize: Normalize embedding

        Returns:
            Query embedding (1D array of shape (1024,))
        """
        formatted_query = self.query_prefix + query

        embedding = self.model.encode(
            formatted_query,
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embedding
