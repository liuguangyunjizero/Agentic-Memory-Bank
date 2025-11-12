"""
Query Graph - Semantic Memory Graph

Graph-based memory storage using adjacency list implementation.
"""

import uuid
import time
import numpy as np
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field


@dataclass
class QueryGraphNode:
    """
    Query Graph node (inspired by A-mem's MemoryNote).

    Attributes:
        id: Unique node identifier (UUID)
        summary: Structured detailed summary
        context: One-sentence topic description
        keywords: List of keywords
        embedding: Semantic vector (numpy array)
        timestamp: Creation timestamp
        merge_description: Merge description (only for merged nodes)
        links: List of neighbor node IDs (related edges)
    """
    id: str
    summary: str
    context: str
    keywords: List[str]
    embedding: np.ndarray
    timestamp: float
    merge_description: Optional[str] = None
    links: List[str] = field(default_factory=list)

    def _add_link(self, neighbor_id: str):
        """Add neighbor link (private method for QueryGraph use only)."""
        if neighbor_id not in self.links:
            self.links.append(neighbor_id)

    def _remove_link(self, neighbor_id: str):
        """Remove neighbor link (private method for QueryGraph use only)."""
        if neighbor_id in self.links:
            self.links.remove(neighbor_id)

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for persistence."""
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
        """Create node from dictionary."""
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
        """Return string representation."""
        return (
            f"QueryGraphNode(id={self.id[:8]}..., "
            f"context='{self.context[:30]}...', "
            f"neighbors={len(self.links)})"
        )


class QueryGraph:
    """
    Query Graph manager (adjacency list implementation, inspired by A-mem).
    Does not use NetworkX; uses dict and list for graph structure.
    """

    def __init__(self):
        """Initialize empty graph."""
        self.nodes_dict: Dict[str, QueryGraphNode] = {}

    def add_node(self, node: QueryGraphNode):
        """Add node to graph."""
        self.nodes_dict[node.id] = node

    def get_node(self, node_id: str) -> Optional[QueryGraphNode]:
        """Get node by ID, returns None if not found."""
        return self.nodes_dict.get(node_id)

    def has_node(self, node_id: str) -> bool:
        """Check if node exists in graph."""
        return node_id in self.nodes_dict

    def delete_node(self, node_id: str):
        """Delete node and all its edges."""
        if node_id not in self.nodes_dict:
            return

        node = self.nodes_dict[node_id]

        # Remove this node from all neighbors' links
        for neighbor_id in node.links:
            if neighbor_id in self.nodes_dict:
                self.nodes_dict[neighbor_id]._remove_link(node_id)

        # Delete the node itself
        del self.nodes_dict[node_id]

    def add_edge(self, node_id1: str, node_id2: str):
        """Add undirected related edge between two nodes."""
        if node_id1 not in self.nodes_dict or node_id2 not in self.nodes_dict:
            raise ValueError("Node does not exist, cannot add edge")

        # Add bidirectional link
        self.nodes_dict[node_id1]._add_link(node_id2)
        self.nodes_dict[node_id2]._add_link(node_id1)

    def remove_edge(self, node_id1: str, node_id2: str):
        """Remove edge between two nodes."""
        if node_id1 in self.nodes_dict:
            self.nodes_dict[node_id1]._remove_link(node_id2)
        if node_id2 in self.nodes_dict:
            self.nodes_dict[node_id2]._remove_link(node_id1)

    def has_edge(self, node_id1: str, node_id2: str) -> bool:
        """Check if edge exists between two nodes."""
        if node_id1 not in self.nodes_dict:
            return False
        return node_id2 in self.nodes_dict[node_id1].links

    def get_neighbors(self, node_id: str) -> List[QueryGraphNode]:
        """Get all neighbor nodes via links list."""
        if node_id not in self.nodes_dict:
            return []

        node = self.nodes_dict[node_id]
        return [
            self.nodes_dict[neighbor_id]
            for neighbor_id in node.links
            if neighbor_id in self.nodes_dict
        ]

    def get_all_nodes(self) -> List[QueryGraphNode]:
        """Get all nodes in graph."""
        return list(self.nodes_dict.values())

    def get_node_count(self) -> int:
        """Get total node count."""
        return len(self.nodes_dict)

    def get_edge_count(self) -> int:
        """Get total edge count."""
        # Each edge is counted twice in adjacency list, so divide by 2
        total_links = sum(len(node.links) for node in self.nodes_dict.values())
        return total_links // 2

    def clear(self):
        """Clear all nodes and edges."""
        self.nodes_dict.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for persistence."""
        return {
            "nodes": [node.to_dict() for node in self.nodes_dict.values()]
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "QueryGraph":
        """Create QueryGraph from dictionary."""
        graph = QueryGraph()
        for node_data in data.get("nodes", []):
            node = QueryGraphNode.from_dict(node_data)
            graph.add_node(node)
        return graph

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"QueryGraph(nodes={self.get_node_count()}, "
            f"edges={self.get_edge_count()})"
        )
