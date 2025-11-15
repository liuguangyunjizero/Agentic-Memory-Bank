"""
Hybrid search combining keyword matching (BM25) with semantic similarity (embeddings).
Retrieves relevant nodes from the graph and expands results with connected neighbors.
"""

import numpy as np
import logging
from typing import List, Set, Optional, Tuple
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class RetrievalModule:
    """
    Balances precision of keyword search with flexibility of semantic matching.
    Maintains lazy-built indexes for efficiency and supports neighbor expansion.
    """

    def __init__(self, alpha: float = 0.5, k: int = 5):
        """
        Configure the blend between keyword and semantic search.
        Alpha controls the weight given to BM25 versus embedding similarity.
        """
        self.alpha = alpha
        self.k = k
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_keywords: List[List[str]] = []
        self.node_ids: List[str] = []
        self._index_dirty: bool = True

        logger.info(f"Retrieval module initialized: alpha={alpha}, k={k}")

    @classmethod
    def from_config(cls, config) -> "RetrievalModule":
        """Extract retrieval parameters from system configuration."""
        return cls(alpha=config.RETRIEVAL_ALPHA, k=config.RETRIEVAL_K)

    def build_index(self, graph):
        """
        Construct BM25 index from current graph state.
        Should be called after nodes are added but deferred until first query for efficiency.
        """
        from src.storage.query_graph import QueryGraph

        if not isinstance(graph, QueryGraph):
            raise TypeError("graph must be a QueryGraph instance")

        self.corpus_keywords = []
        self.node_ids = []

        for node in graph.nodes_dict.values():
            keywords = [kw for kw in node.keywords if kw]
            if not keywords:
                keywords = ["placeholder"]
            self.corpus_keywords.append(keywords)
            self.node_ids.append(node.id)

        if self.corpus_keywords:
            self.bm25 = BM25Okapi(self.corpus_keywords)
            logger.info(f"BM25 index built successfully, corpus size: {len(self.corpus_keywords)}")
        else:
            self.bm25 = None
            logger.warning("Corpus is empty, cannot build BM25 index")

        self._index_dirty = False

    def mark_index_dirty(self):
        """
        Flag that the index is stale due to graph modifications.
        Next retrieval will trigger automatic rebuild.
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
        Find top-k nodes by combined score, expand with direct neighbors,
        and return sorted by recency. Automatically rebuilds index if needed.
        Exclude_ids allows filtering out known nodes to avoid duplication.
        """
        from src.storage.query_graph import QueryGraph

        if not isinstance(graph, QueryGraph):
            raise TypeError("graph must be a QueryGraph instance")

        if self._index_dirty:
            self.build_index(graph)

        exclude_ids = exclude_ids or set()

        if not graph.nodes_dict:
            return []

        all_nodes = list(graph.nodes_dict.values())
        scored_nodes: List[Tuple[str, float]] = []

        bm25_scores_normalized = {}
        if self.bm25 and query_keywords:
            bm25_scores = self.bm25.get_scores(query_keywords)
            max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
            for i, node_id in enumerate(self.node_ids):
                bm25_scores_normalized[node_id] = bm25_scores[i] / max_bm25

        for node in all_nodes:
            if node.id in exclude_ids:
                continue

            bm25_score = bm25_scores_normalized.get(node.id, 0.0)

            semantic_score = float(np.dot(query_embedding, node.embedding))

            final_score = self.alpha * bm25_score + (1 - self.alpha) * semantic_score
            scored_nodes.append((node.id, final_score))

        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        top_k_ids = [node_id for node_id, _ in scored_nodes[:self.k]]

        result_ids = set(top_k_ids)
        for node_id in top_k_ids:
            neighbors = graph.get_neighbors(node_id)
            for neighbor in neighbors:
                if neighbor.id not in exclude_ids:
                    result_ids.add(neighbor.id)

        result_nodes = [graph.nodes_dict[nid] for nid in result_ids if nid in graph.nodes_dict]
        result_nodes.sort(key=lambda n: n.timestamp, reverse=True)

        return result_nodes

    def __repr__(self) -> str:
        """Show configuration and index status."""
        return f"RetrievalModule(alpha={self.alpha}, k={self.k}, indexed={self.bm25 is not None})"
