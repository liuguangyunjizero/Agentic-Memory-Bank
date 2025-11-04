"""
图操作模块

封装 Query Graph 的 CRUD 操作：
- 节点添加/删除
- 边添加/删除
- 节点合并

规范文档：第4.3节
"""

import logging
from typing import List, Set

logger = logging.getLogger(__name__)


class GraphOperations:
    """图操作模块（封装 Query Graph 的 CRUD 操作）"""

    def __init__(self, graph):
        """
        初始化图操作模块

        Args:
            graph: QueryGraph 实例
        """
        from src.storage.query_graph import QueryGraph

        if not isinstance(graph, QueryGraph):
            raise TypeError("graph 必须是 QueryGraph 实例")

        self.graph = graph
        logger.info("图操作模块初始化完成")

    def add_node(self, node) -> None:
        """
        添加节点

        Args:
            node: QueryGraphNode 实例
        """
        self.graph.add_node(node)
        logger.debug(f"添加节点: {node.id[:8]}... (context: {node.context[:30]}...)")

    def delete_node(self, node_id: str) -> None:
        """
        删除节点及其所有边

        Args:
            node_id: 节点ID
        """
        if not self.graph.has_node(node_id):
            logger.warning(f"节点不存在，无法删除: {node_id[:8]}...")
            return

        self.graph.delete_node(node_id)
        logger.debug(f"删除节点: {node_id[:8]}...")

    def add_edge(self, node_id1: str, node_id2: str) -> None:
        """
        在两个节点间创建 related 边

        Args:
            node_id1: 第一个节点ID
            node_id2: 第二个节点ID
        """
        try:
            self.graph.add_edge(node_id1, node_id2)
            logger.debug(f"添加边: {node_id1[:8]}... <-> {node_id2[:8]}...")
        except ValueError as e:
            logger.error(f"添加边失败: {str(e)}")
            raise

    def remove_edge(self, node_id1: str, node_id2: str) -> None:
        """
        移除边

        Args:
            node_id1: 第一个节点ID
            node_id2: 第二个节点ID
        """
        self.graph.remove_edge(node_id1, node_id2)
        logger.debug(f"移除边: {node_id1[:8]}... <-> {node_id2[:8]}...")

    def get_neighbors(self, node_id: str) -> List:
        """
        获取节点的所有邻居

        Args:
            node_id: 节点ID

        Returns:
            邻居节点列表
        """
        return self.graph.get_neighbors(node_id)

    def update_node_attributes(
        self,
        node_id: str,
        summary: str = None,
        context: str = None,
        keywords: List[str] = None,
        embedding = None
    ) -> None:
        """
        更新节点的属性

        Args:
            node_id: 节点ID
            summary: 新的摘要（可选）
            context: 新的上下文（可选）
            keywords: 新的关键词（可选）
            embedding: 新的 embedding（可选）
        """
        if not self.graph.has_node(node_id):
            logger.warning(f"节点不存在，无法更新: {node_id[:8]}...")
            return

        node = self.graph.get_node(node_id)

        if summary is not None:
            node.summary = summary
        if context is not None:
            node.context = context
        if keywords is not None:
            node.keywords = keywords
        if embedding is not None:
            node.embedding = embedding

        logger.debug(f"更新节点属性: {node_id[:8]}...")

    def merge_nodes(
        self,
        old_node_ids: List[str],
        new_node
    ) -> None:
        """
        合并多个节点：
        1. 创建新节点
        2. 继承所有旧节点的边（去重）
        3. 删除旧节点

        Args:
            old_node_ids: 待合并的旧节点ID列表
            new_node: 新的合并节点
        """
        # 1. 添加新节点
        self.add_node(new_node)

        # 2. 继承边（去重）
        neighbor_ids: Set[str] = set()
        for old_node_id in old_node_ids:
            if not self.graph.has_node(old_node_id):
                logger.warning(f"旧节点不存在，跳过: {old_node_id[:8]}...")
                continue

            neighbors = self.get_neighbors(old_node_id)
            for neighbor in neighbors:
                # 排除被合并的节点和新节点本身
                if neighbor.id not in old_node_ids and neighbor.id != new_node.id:
                    neighbor_ids.add(neighbor.id)

        # 添加继承的边
        for neighbor_id in neighbor_ids:
            self.add_edge(new_node.id, neighbor_id)

        # 3. 删除旧节点
        for old_node_id in old_node_ids:
            if self.graph.has_node(old_node_id):
                self.delete_node(old_node_id)

        logger.info(
            f"合并节点完成: {len(old_node_ids)} 个旧节点 -> 新节点 {new_node.id[:8]}..., "
            f"继承 {len(neighbor_ids)} 条边"
        )

    def get_node(self, node_id: str):
        """
        获取节点

        Args:
            node_id: 节点ID

        Returns:
            节点实例，如果不存在则返回 None
        """
        return self.graph.get_node(node_id)

    def has_node(self, node_id: str) -> bool:
        """
        检查节点是否存在

        Args:
            node_id: 节点ID

        Returns:
            存在返回 True，否则返回 False
        """
        return self.graph.has_node(node_id)

    def has_edge(self, node_id1: str, node_id2: str) -> bool:
        """
        检查两个节点间是否有边

        Args:
            node_id1: 第一个节点ID
            node_id2: 第二个节点ID

        Returns:
            有边返回 True，否则返回 False
        """
        return self.graph.has_edge(node_id1, node_id2)

    def __repr__(self) -> str:
        """返回模块摘要"""
        return f"GraphOperations(nodes={self.graph.get_node_count()}, edges={self.graph.get_edge_count()})"
