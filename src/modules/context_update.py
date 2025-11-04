"""
上下文更新模块

更新节点的 context 和 keywords，并重新计算 embedding

规范文档：第4.4节
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class ContextUpdateModule:
    """Context 更新模块（更新节点并重新计算 embedding）"""

    def __init__(self, graph, embedding_module):
        """
        初始化上下文更新模块

        Args:
            graph: QueryGraph 实例
            embedding_module: EmbeddingModule 实例
        """
        from src.storage.query_graph import QueryGraph
        from src.modules.embedding import EmbeddingModule

        if not isinstance(graph, QueryGraph):
            raise TypeError("graph 必须是 QueryGraph 实例")
        if not isinstance(embedding_module, EmbeddingModule):
            raise TypeError("embedding_module 必须是 EmbeddingModule 实例")

        self.graph = graph
        self.embedding_module = embedding_module
        logger.info("上下文更新模块初始化完成")

    def update_node_context(
        self,
        node_id: str,
        new_context: str,
        new_keywords: List[str]
    ) -> None:
        """
        更新节点的 context 和 keywords，并重新计算 embedding

        Args:
            node_id: 节点ID
            new_context: 新的 context 描述
            new_keywords: 新的 keywords 列表
        """
        if not self.graph.has_node(node_id):
            logger.warning(f"节点不存在，无法更新: {node_id[:8]}...")
            return

        node = self.graph.get_node(node_id)

        # 1. 更新属性
        node.context = new_context
        node.keywords = new_keywords

        # 2. 重新计算 embedding
        text = f"{node.summary} {node.context} {' '.join(node.keywords)}"
        node.embedding = self.embedding_module.compute_embedding(text)

        logger.debug(f"更新节点上下文: {node_id[:8]}... (new context: {new_context[:30]}...)")

    def update_node_summary(
        self,
        node_id: str,
        new_summary: str
    ) -> None:
        """
        更新节点的 summary，并重新计算 embedding

        Args:
            node_id: 节点ID
            new_summary: 新的 summary
        """
        if not self.graph.has_node(node_id):
            logger.warning(f"节点不存在，无法更新: {node_id[:8]}...")
            return

        node = self.graph.get_node(node_id)

        # 1. 更新属性
        node.summary = new_summary

        # 2. 重新计算 embedding
        text = f"{node.summary} {node.context} {' '.join(node.keywords)}"
        node.embedding = self.embedding_module.compute_embedding(text)

        logger.debug(f"更新节点摘要: {node_id[:8]}...")

    def update_full_node(
        self,
        node_id: str,
        new_summary: str,
        new_context: str,
        new_keywords: List[str]
    ) -> None:
        """
        完全更新节点的所有文本属性，并重新计算 embedding

        Args:
            node_id: 节点ID
            new_summary: 新的 summary
            new_context: 新的 context
            new_keywords: 新的 keywords
        """
        if not self.graph.has_node(node_id):
            logger.warning(f"节点不存在，无法更新: {node_id[:8]}...")
            return

        node = self.graph.get_node(node_id)

        # 1. 更新所有属性
        node.summary = new_summary
        node.context = new_context
        node.keywords = new_keywords

        # 2. 重新计算 embedding
        text = f"{node.summary} {node.context} {' '.join(node.keywords)}"
        node.embedding = self.embedding_module.compute_embedding(text)

        logger.debug(f"完全更新节点: {node_id[:8]}...")

    def batch_update_node_contexts(
        self,
        updates: dict
    ) -> None:
        """
        批量更新多个节点的 context 和 keywords，并重新计算 embedding

        这个方法用于优化性能：当多个节点需要更新时，避免重复计算同一节点的embedding

        Args:
            updates: 字典，格式为 {node_id: {"context": str, "keywords": list}}

        Example:
            updates = {
                "node_1": {"context": "新主题1", "keywords": ["关键词1", "关键词2"]},
                "node_2": {"context": "新主题2", "keywords": ["关键词3", "关键词4"]}
            }
        """
        if not updates:
            return

        updated_count = 0
        for node_id, update_data in updates.items():
            if not self.graph.has_node(node_id):
                logger.warning(f"节点不存在，跳过更新: {node_id[:8]}...")
                continue

            node = self.graph.get_node(node_id)

            # 更新属性
            node.context = update_data.get("context", node.context)
            node.keywords = update_data.get("keywords", node.keywords)

            # 重新计算 embedding
            text = f"{node.summary} {node.context} {' '.join(node.keywords)}"
            node.embedding = self.embedding_module.compute_embedding(text)

            updated_count += 1

        logger.info(f"批量更新完成: {updated_count}/{len(updates)} 个节点")

    def __repr__(self) -> str:
        """返回模块摘要"""
        return f"ContextUpdateModule(graph_nodes={self.graph.get_node_count()})"
