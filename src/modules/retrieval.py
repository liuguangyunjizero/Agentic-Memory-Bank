"""
混合检索模块

结合 BM25（关键词）和 Embedding（语义）的混合检索：
- Top-K 检索
- 邻居扩展（一层）
- 时间排序

参考：A-mem 的 HybridRetriever
规范文档：第4.2节
"""

import numpy as np
import logging
from typing import List, Set, Optional, Tuple
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class RetrievalModule:
    """
    混合检索模块（参考 A-mem 的 HybridRetriever）

    检索策略：BM25（关键词）+ Embedding（语义）
    """

    def __init__(self, alpha: float = 0.5, k: int = 5):
        """
        初始化检索模块

        Args:
            alpha: 混合检索权重（BM25 的权重，embedding 权重为 1-alpha）
            k: top-k 检索的 k 值
        """
        self.alpha = alpha
        self.k = k
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_keywords: List[List[str]] = []  # 预处理的关键词列表
        self.node_ids: List[str] = []  # 节点ID列表（与corpus_keywords对应）
        self._index_dirty: bool = True  # ✅ 优化：索引脏标记（初始为True需要构建）

        logger.info(f"检索模块初始化: alpha={alpha}, k={k}")

    @classmethod
    def from_config(cls, config) -> "RetrievalModule":
        """
        从 Config 对象创建检索模块

        Args:
            config: Config 实例

        Returns:
            RetrievalModule 实例
        """
        return cls(alpha=config.RETRIEVAL_ALPHA, k=config.RETRIEVAL_K)

    def build_index(self, graph):
        """
        构建检索索引（在添加节点后调用）

        Args:
            graph: QueryGraph 对象
        """
        from src.storage.query_graph import QueryGraph

        if not isinstance(graph, QueryGraph):
            raise TypeError("graph 必须是 QueryGraph 实例")

        # 提取所有节点的 keywords 作为 BM25 语料
        self.corpus_keywords = []
        self.node_ids = []

        for node in graph.nodes_dict.values():
            # 过滤空的keywords，避免BM25错误
            keywords = [kw for kw in node.keywords if kw]
            if not keywords:
                keywords = ["placeholder"]  # 如果完全没有关键词，使用占位符
            self.corpus_keywords.append(keywords)
            self.node_ids.append(node.id)

        # 构建 BM25 索引
        if self.corpus_keywords:
            self.bm25 = BM25Okapi(self.corpus_keywords)
            logger.info(f"BM25 索引构建完成，语料大小: {len(self.corpus_keywords)}")
        else:
            self.bm25 = None
            logger.warning("语料为空，无法构建 BM25 索引")

        # ✅ 优化：重置脏标记
        self._index_dirty = False

    def mark_index_dirty(self):
        """
        ✅ 优化：标记索引为脏（需要重建）
        在图修改操作后调用
        """
        self._index_dirty = True
        logger.debug("BM25 索引已标记为脏，下次检索时将重建")

    def hybrid_retrieval(
        self,
        query_embedding: np.ndarray,
        query_keywords: List[str],
        graph,
        exclude_ids: Optional[Set[str]] = None
    ) -> List:
        """
        混合检索：返回 top-k + 一层邻居，按 timestamp 降序排列

        Args:
            query_embedding: 查询的 embedding 向量
            query_keywords: 查询的关键词列表
            graph: QueryGraph 对象
            exclude_ids: 需要排除的节点ID（如已知邻居）

        Returns:
            检索结果（候选节点 + 邻居节点，按 timestamp 降序）
        """
        from src.storage.query_graph import QueryGraph

        if not isinstance(graph, QueryGraph):
            raise TypeError("graph 必须是 QueryGraph 实例")

        # ✅ 优化：自动重建索引（如果索引为脏）
        if self._index_dirty:
            logger.debug("检测到索引脏标记，自动重建BM25索引...")
            self.build_index(graph)

        exclude_ids = exclude_ids or set()

        if not graph.nodes_dict:
            return []

        # 1. 计算所有节点的混合分数
        all_nodes = list(graph.nodes_dict.values())
        scored_nodes: List[Tuple[str, float]] = []

        for node in all_nodes:
            if node.id in exclude_ids:
                continue

            # BM25 分数（关键词匹配）
            bm25_score = 0.0
            if self.bm25 and query_keywords:
                bm25_scores = self.bm25.get_scores(query_keywords)
                # 修复：找到node在self.node_ids中的正确索引
                try:
                    node_idx = self.node_ids.index(node.id)
                    bm25_score = bm25_scores[node_idx]
                    # ✅ 改进：除零保护 + 空列表保护
                    max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
                    bm25_score = bm25_score / max_bm25
                except (ValueError, IndexError):
                    # 节点不在索引中，使用0分
                    bm25_score = 0.0

            # Embedding 相似度分数
            semantic_score = float(np.dot(query_embedding, node.embedding))

            # 混合分数
            final_score = self.alpha * bm25_score + (1 - self.alpha) * semantic_score
            scored_nodes.append((node.id, final_score))

        # 2. 筛选 top-k 候选节点
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        top_k_ids = [node_id for node_id, _ in scored_nodes[:self.k]]

        # 3. 扩展一层邻居
        result_ids = set(top_k_ids)
        for node_id in top_k_ids:
            neighbors = graph.get_neighbors(node_id)
            for neighbor in neighbors:
                if neighbor.id not in exclude_ids:
                    result_ids.add(neighbor.id)

        # 4. 获取完整节点信息并按 timestamp 降序排序
        result_nodes = [graph.nodes_dict[nid] for nid in result_ids if nid in graph.nodes_dict]
        result_nodes.sort(key=lambda n: n.timestamp, reverse=True)

        logger.debug(
            f"混合检索完成: top-k={len(top_k_ids)}, "
            f"扩展后={len(result_nodes)}"
        )

        return result_nodes

    def __repr__(self) -> str:
        """返回模块摘要"""
        return f"RetrievalModule(alpha={self.alpha}, k={self.k}, indexed={self.bm25 is not None})"
