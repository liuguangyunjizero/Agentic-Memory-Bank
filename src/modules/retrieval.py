"""
Hybrid Retrieval Module

Combines BM25 (keyword) and Embedding (semantic) hybrid retrieval:
- Top-K retrieval
- Neighbor expansion (one layer)
- Time sorting
"""

import numpy as np
import logging
from typing import List, Set, Optional, Tuple
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class RetrievalModule:
    """
    Hybrid Retrieval Module (inspired by A-mem's HybridRetriever)

    Retrieval strategy: BM25 (keyword) + Embedding (semantic)
    """

    def __init__(self, alpha: float = 0.5, k: int = 5):
        """
        Initialize retrieval module

        Args:
            alpha: Hybrid retrieval weight (BM25 weight, embedding weight is 1-alpha)
            k: k value for top-k retrieval
        """
        self.alpha = alpha
        self.k = k
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_keywords: List[List[str]] = []  # Preprocessed keyword list
        self.node_ids: List[str] = []  # Node ID list (corresponds to corpus_keywords)
        self._index_dirty: bool = True  # Optimization: index dirty flag (initially True, needs building)

        logger.info(f"Retrieval module initialized: alpha={alpha}, k={k}")

    @classmethod
    def from_config(cls, config) -> "RetrievalModule":
        """
        Create retrieval module from Config object

        Args:
            config: Config instance

        Returns:
            RetrievalModule instance
        """
        return cls(alpha=config.RETRIEVAL_ALPHA, k=config.RETRIEVAL_K)

    def build_index(self, graph):
        """
        Build retrieval index (call after adding nodes)

        Args:
            graph: QueryGraph object
        """
        from src.storage.query_graph import QueryGraph

        if not isinstance(graph, QueryGraph):
            raise TypeError("graph must be a QueryGraph instance")

        # Extract keywords from all nodes as BM25 corpus
        self.corpus_keywords = []
        self.node_ids = []

        for node in graph.nodes_dict.values():
            # Filter empty keywords to avoid BM25 errors
            keywords = [kw for kw in node.keywords if kw]
            if not keywords:
                keywords = ["placeholder"]  # Use placeholder if no keywords at all
            self.corpus_keywords.append(keywords)
            self.node_ids.append(node.id)

        # Build BM25 index
        if self.corpus_keywords:
            self.bm25 = BM25Okapi(self.corpus_keywords)
            logger.info(f"BM25 index built successfully, corpus size: {len(self.corpus_keywords)}")
        else:
            self.bm25 = None
            logger.warning("Corpus is empty, cannot build BM25 index")

        # Optimization: reset dirty flag
        self._index_dirty = False

    def mark_index_dirty(self):
        """
        Optimization: mark index as dirty (needs rebuilding)
        Call after graph modification operations
        """
        self._index_dirty = True

    def hybrid_retrieval(
        self,
        query_embedding: np.ndarray,
        query_keywords: List[str],
        graph,
        exclude_ids: Optional[Set[str]] = None
    ) -> List:
        """
        Hybrid retrieval: return top-k + one layer of neighbors, sorted by timestamp descending

        Args:
            query_embedding: Query embedding vector
            query_keywords: Query keyword list
            graph: QueryGraph object
            exclude_ids: Node IDs to exclude (e.g., known neighbors)

        Returns:
            Retrieval results (candidate nodes + neighbor nodes, sorted by timestamp descending)
        """
        from src.storage.query_graph import QueryGraph

        if not isinstance(graph, QueryGraph):
            raise TypeError("graph must be a QueryGraph instance")

        # Optimization: auto-rebuild index (if index is dirty)
        if self._index_dirty:
            self.build_index(graph)

        exclude_ids = exclude_ids or set()

        if not graph.nodes_dict:
            return []

        # 1. Calculate hybrid scores for all nodes
        all_nodes = list(graph.nodes_dict.values())
        scored_nodes: List[Tuple[str, float]] = []

        # Performance optimization: pre-compute BM25 scores (avoid O(n²) repeated calculation in loop -> O(n))
        bm25_scores_normalized = {}
        if self.bm25 and query_keywords:
            bm25_scores = self.bm25.get_scores(query_keywords)
            max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
            for i, node_id in enumerate(self.node_ids):
                bm25_scores_normalized[node_id] = bm25_scores[i] / max_bm25

        for node in all_nodes:
            if node.id in exclude_ids:
                continue

            # BM25 score (from pre-computed results)
            bm25_score = bm25_scores_normalized.get(node.id, 0.0)

            # Embedding similarity score
            semantic_score = float(np.dot(query_embedding, node.embedding))

            # Hybrid score
            final_score = self.alpha * bm25_score + (1 - self.alpha) * semantic_score
            scored_nodes.append((node.id, final_score))

        # 2. Filter top-k candidate nodes
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        top_k_ids = [node_id for node_id, _ in scored_nodes[:self.k]]

        # 3. Expand one layer of neighbors
        result_ids = set(top_k_ids)
        for node_id in top_k_ids:
            neighbors = graph.get_neighbors(node_id)
            for neighbor in neighbors:
                if neighbor.id not in exclude_ids:
                    result_ids.add(neighbor.id)

        # 4. Get complete node info and sort by timestamp descending
        result_nodes = [graph.nodes_dict[nid] for nid in result_ids if nid in graph.nodes_dict]
        result_nodes.sort(key=lambda n: n.timestamp, reverse=True)

        return result_nodes

    def __repr__(self) -> str:
        """Return module summary"""
        return f"RetrievalModule(alpha={self.alpha}, k={self.k}, indexed={self.bm25 is not None})"
