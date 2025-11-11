"""
Query Graph - 语义记忆图
"""

import uuid
import time
import numpy as np
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field


@dataclass
class QueryGraphNode:
    """
    Query Graph 节点（参考 A-mem 的 MemoryNote）

    Attributes:
        id: 节点唯一标识符（UUID）
        summary: 结构化详细摘要
        context: 一句话主题描述
        keywords: 关键词列表
        embedding: 语义向量（numpy数组）
        timestamp: 创建时间戳
        merge_description: 合并节点描述（仅合并节点有值）
        links: 邻居节点ID列表（related边）
    """
    # 核心属性（无默认值的字段必须在前）
    id: str
    summary: str
    context: str
    keywords: List[str]
    embedding: np.ndarray
    timestamp: float

    # 可选属性（有默认值的字段必须在后）
    merge_description: Optional[str] = None
    links: List[str] = field(default_factory=list)

    def add_link(self, neighbor_id: str):
        """
        添加邻居链接

        Args:
            neighbor_id: 邻居节点ID
        """
        if neighbor_id not in self.links:
            self.links.append(neighbor_id)

    def remove_link(self, neighbor_id: str):
        """
        移除邻居链接

        Args:
            neighbor_id: 邻居节点ID
        """
        if neighbor_id in self.links:
            self.links.remove(neighbor_id)

    def to_dict(self) -> Dict[str, Any]:
        """
        导出为字典（用于持久化）

        Returns:
            包含所有字段的字典
        """
        return {
            "id": self.id,
            "summary": self.summary,
            "context": self.context,
            "keywords": self.keywords,
            "merge_description": self.merge_description,
            "embedding": self.embedding.tolist(),
            "timestamp": self.timestamp,
            "links": self.links
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "QueryGraphNode":
        """
        从字典创建节点

        Args:
            data: 包含所有字段的字典

        Returns:
            QueryGraphNode 实例
        """
        return QueryGraphNode(
            id=data["id"],
            summary=data["summary"],
            context=data["context"],
            keywords=data["keywords"],
            embedding=np.array(data["embedding"]),
            timestamp=data["timestamp"],
            merge_description=data.get("merge_description"),
            links=data.get("links", [])
        )

    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"QueryGraphNode(id={self.id[:8]}..., "
            f"context='{self.context[:30]}...', "
            f"neighbors={len(self.links)})"
        )


class QueryGraph:
    """
    Query Graph 管理类（邻接表实现，参考 A-mem）

    不使用 NetworkX，使用字典和列表自实现图结构
    """

    def __init__(self):
        """初始化空图"""
        self.nodes_dict: Dict[str, QueryGraphNode] = {}  # {id: Node}

    def add_node(self, node: QueryGraphNode):
        """
        添加节点

        Args:
            node: QueryGraphNode 实例
        """
        self.nodes_dict[node.id] = node

    def get_node(self, node_id: str) -> Optional[QueryGraphNode]:
        """
        获取节点

        Args:
            node_id: 节点ID

        Returns:
            节点实例，如果不存在则返回 None
        """
        return self.nodes_dict.get(node_id)

    def has_node(self, node_id: str) -> bool:
        """
        检查节点是否存在

        Args:
            node_id: 节点ID

        Returns:
            存在返回 True，否则返回 False
        """
        return node_id in self.nodes_dict

    def delete_node(self, node_id: str):
        """
        删除节点及其所有边

        Args:
            node_id: 要删除的节点ID
        """
        if node_id not in self.nodes_dict:
            return

        node = self.nodes_dict[node_id]

        # 1. 从所有邻居的links中移除该节点
        for neighbor_id in node.links:
            if neighbor_id in self.nodes_dict:
                self.nodes_dict[neighbor_id].remove_link(node_id)

        # 2. 删除节点本身
        del self.nodes_dict[node_id]

    def add_edge(self, node_id1: str, node_id2: str):
        """
        添加 related 边（无向边，通过 links 列表实现）

        Args:
            node_id1: 第一个节点ID
            node_id2: 第二个节点ID
        """
        if node_id1 not in self.nodes_dict or node_id2 not in self.nodes_dict:
            raise ValueError("节点不存在，无法添加边")

        # 双向添加链接
        self.nodes_dict[node_id1].add_link(node_id2)
        self.nodes_dict[node_id2].add_link(node_id1)

    def remove_edge(self, node_id1: str, node_id2: str):
        """
        移除边

        Args:
            node_id1: 第一个节点ID
            node_id2: 第二个节点ID
        """
        if node_id1 in self.nodes_dict:
            self.nodes_dict[node_id1].remove_link(node_id2)
        if node_id2 in self.nodes_dict:
            self.nodes_dict[node_id2].remove_link(node_id1)

    def has_edge(self, node_id1: str, node_id2: str) -> bool:
        """
        检查两个节点间是否有边

        Args:
            node_id1: 第一个节点ID
            node_id2: 第二个节点ID

        Returns:
            有边返回 True，否则返回 False
        """
        if node_id1 not in self.nodes_dict:
            return False
        return node_id2 in self.nodes_dict[node_id1].links

    def get_neighbors(self, node_id: str) -> List[QueryGraphNode]:
        """
        获取节点的所有邻居（通过 links 列表）

        Args:
            node_id: 节点ID

        Returns:
            邻居节点列表
        """
        if node_id not in self.nodes_dict:
            return []

        node = self.nodes_dict[node_id]
        return [
            self.nodes_dict[neighbor_id]
            for neighbor_id in node.links
            if neighbor_id in self.nodes_dict
        ]

    def get_all_nodes(self) -> List[QueryGraphNode]:
        """
        获取所有节点

        Returns:
            所有节点的列表
        """
        return list(self.nodes_dict.values())

    def get_node_count(self) -> int:
        """
        获取节点总数

        Returns:
            节点数量
        """
        return len(self.nodes_dict)

    def get_edge_count(self) -> int:
        """
        获取边总数

        Returns:
            边的数量
        """
        # 每条边在邻接表中被计数两次，所以除以2
        total_links = sum(len(node.links) for node in self.nodes_dict.values())
        return total_links // 2

    def clear(self):
        """清空图中所有节点和边"""
        self.nodes_dict.clear()

    def to_dict(self) -> Dict[str, Any]:
        """
        导出为字典（持久化）

        Returns:
            包含所有节点的字典
        """
        return {
            "nodes": [node.to_dict() for node in self.nodes_dict.values()]
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "QueryGraph":
        """
        从字典创建 QueryGraph

        Args:
            data: 包含所有节点的字典

        Returns:
            QueryGraph 实例
        """
        graph = QueryGraph()
        for node_data in data.get("nodes", []):
            node = QueryGraphNode.from_dict(node_data)
            graph.add_node(node)
        return graph

    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"QueryGraph(nodes={self.get_node_count()}, "
            f"edges={self.get_edge_count()})"
        )


def create_node(
    summary: str,
    context: str,
    keywords: List[str],
    embedding: np.ndarray,
    node_id: Optional[str] = None
) -> QueryGraphNode:
    """
    创建新的 Query Graph 节点（辅助函数）

    Args:
        summary: 结构化摘要
        context: 一句话描述
        keywords: 关键词列表
        embedding: 语义向量
        node_id: 节点ID（可选，不提供则自动生成UUID）

    Returns:
        QueryGraphNode 实例
    """
    if node_id is None:
        node_id = str(uuid.uuid4())

    return QueryGraphNode(
        id=node_id,
        summary=summary,
        context=context,
        keywords=keywords,
        embedding=embedding,
        timestamp=time.time()
    )
