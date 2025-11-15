"""
Vector embedding generation using sentence transformers for semantic similarity.
Runs locally without external API calls, producing normalized vectors for efficient comparison.
"""

import numpy as np
import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModule:
    """
    Converts text into dense vector representations using pre-trained models.
    Handles model loading, vector computation, and normalization for downstream use.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Load the specified transformer model into memory for encoding.
        Different models offer trade-offs between speed and accuracy.
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
        Factory method that extracts model selection from configuration object.
        """
        return cls(model_name=config.EMBEDDING_MODEL)

    def compute_embedding(self, text: str) -> np.ndarray:
        """
        Transform input text into a normalized embedding vector.
        Empty inputs produce zero vectors to avoid downstream errors.
        Normalization ensures cosine similarity can be computed via dot product.
        """
        if not text or not text.strip():
            logger.warning("Input text is empty, returning zero vector")
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros(dim)

        embedding = self.model.encode([text])[0]

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def get_embedding_dimension(self) -> int:
        """Return the dimensionality of vectors produced by this model."""
        return self.model.get_sentence_embedding_dimension()

    def __repr__(self) -> str:
        """Display model name and vector size for identification."""
        return f"EmbeddingModule(model={self.model_name}, dim={self.get_embedding_dimension()})"
