"""
Embedding Manager Module

Handles embedding computation using sentence-transformers.
Provides utilities for computing embeddings and calculating similarity.
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """
    Manages embedding computation and similarity calculations.

    Uses sentence-transformers library with configurable models.
    Default model: all-MiniLM-L6-v2 (lightweight, good performance)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def compute_embedding(self, text: str) -> np.ndarray:
        """
        Compute embedding for a single text.

        Args:
            text: Input text

        Returns:
            np.ndarray: Embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim)

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def batch_compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for multiple texts in batch.

        Args:
            texts: List of input texts

        Returns:
            np.ndarray: Array of embedding vectors, shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        # Replace empty strings with a placeholder to avoid errors
        processed_texts = [text if text and text.strip() else " " for text in texts]
        embeddings = self.model.encode(processed_texts, convert_to_numpy=True)
        return embeddings

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Cosine similarity score (range: -1 to 1)
        """
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(similarity)

    @staticmethod
    def batch_cosine_similarity(query_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a query vector and multiple vectors.

        Args:
            query_vec: Query vector, shape (embedding_dim,)
            vectors: Array of vectors, shape (n_vectors, embedding_dim)

        Returns:
            np.ndarray: Array of similarity scores, shape (n_vectors,)
        """
        if len(vectors) == 0:
            return np.array([])

        # Normalize query vector
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return np.zeros(len(vectors))

        normalized_query = query_vec / query_norm

        # Normalize document vectors
        vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        vector_norms = np.where(vector_norms == 0, 1, vector_norms)
        normalized_vectors = vectors / vector_norms

        # Compute dot product (cosine similarity for normalized vectors)
        similarities = np.dot(normalized_vectors, normalized_query)

        return similarities
