"""
Embedding Computation Module

Uses SentenceTransformer for local embedding computation:
- Batch computation support
- Vector normalization (for cosine similarity)
- Similarity calculation
"""

import numpy as np
import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModule:
    """
    Embedding Computation Module (inspired by A-mem)

    Uses SentenceTransformer for local computation, no API calls needed
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize Embedding Module

        Args:
            model_name: SentenceTransformer model name
                       'all-MiniLM-L6-v2' - fast, dimension 384
                       'all-mpnet-base-v2' - accurate, dimension 768
        """
        self.model_name = model_name
        logger.info(f"Loading Embedding model: {model_name}")

        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded successfully, dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load Embedding model: {str(e)}")
            raise

    @classmethod
    def from_config(cls, config) -> "EmbeddingModule":
        """
        Create Embedding module from Config object

        Args:
            config: Config instance

        Returns:
            EmbeddingModule instance
        """
        return cls(model_name=config.EMBEDDING_MODEL)

    def compute_embedding(self, text: str) -> np.ndarray:
        """
        Compute embedding for a single text

        Args:
            text: Input text (usually combination of summary + context + keywords)

        Returns:
            Normalized embedding vector
        """
        if not text or not text.strip():
            logger.warning("Input text is empty, returning zero vector")
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros(dim)

        # Compute embedding
        embedding = self.model.encode([text])[0]

        # Normalize (for cosine similarity)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity

        Args:
            vec1, vec2: Normalized vectors

        Returns:
            Similarity score [0, 1]
        """
        similarity = float(np.dot(vec1, vec2))
        # Ensure in [0, 1] range (may slightly exceed due to floating point error)
        return max(0.0, min(1.0, similarity))

    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension

        Returns:
            Vector dimension
        """
        return self.model.get_sentence_embedding_dimension()

    def __repr__(self) -> str:
        """Return module summary"""
        return f"EmbeddingModule(model={self.model_name}, dim={self.get_embedding_dimension()})"
