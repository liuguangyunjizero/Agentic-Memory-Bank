"""
Semantic memory storage using a graph structure with adjacency lists.
Each node represents structured knowledge with embedding vectors and relationships.
"""

import uuid
import time
import numpy as np
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field


@dataclass
class QueryGraphNode:
    """
    Individual knowledge unit in the graph with semantic representation and connections.
    Contains structured information fields and maintains its own neighbor list.
    """
    id: str
    summary: str
    context: str
    keywords: List[str]
    embedding: np.ndarray
    timestamp: float
    merge_description: Optional[str] = None
    core_information: str = ""
    supporting_evidence: str = ""
    structure_summary: str = ""
    acquisition_logic: Optional[str] = None
    links: List[str] = field(default_factory=list)

    def _add_link(self, neighbor_id: str):
        """Append a connection to another node, avoiding duplicates."""
        if neighbor_id not in self.links:
            self.links.append(neighbor_id)

    def _remove_link(self, neighbor_id: str):
        """Sever a connection to another node if it exists."""
        if neighbor_id in self.links:
            self.links.remove(neighbor_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node including numpy embedding as list."""
        return {
            "id": self.id,
            "summary": self.summary,
            "context": self.context,
            "keywords": self.keywords,
            "merge_description": self.merge_description,
            "core_information": self.core_information,
            "supporting_evidence": self.supporting_evidence,
            "structure_summary": self.structure_summary,
            "acquisition_logic": self.acquisition_logic,
            "embedding": self.embedding.tolist(),
            "timestamp": self.timestamp,
            "links": self.links
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "QueryGraphNode":
        """Deserialize node and reconstruct numpy embedding from list."""
        return QueryGraphNode(
            id=data["id"],
            summary=data["summary"],
            context=data["context"],
            keywords=data["keywords"],
            embedding=np.array(data["embedding"]),
            timestamp=data["timestamp"],
            merge_description=data.get("merge_description"),
            core_information=data.get("core_information", ""),
            supporting_evidence=data.get("supporting_evidence", ""),
            structure_summary=data.get("structure_summary", ""),
            acquisition_logic=data.get("acquisition_logic"),
            links=data.get("links", [])
        )

    def __repr__(self) -> str:
        """Provide brief node summary for debugging."""
        return (
            f"QueryGraphNode(id={self.id[:8]}..., "
            f"context='{self.context[:30]}...', "
            f"neighbors={len(self.links)})"
        )


class QueryGraph:
    """
    Graph container managing nodes and edges using simple Python data structures.
    Implements undirected graph with adjacency list representation.
    """

    def __init__(self):
        """Start with empty graph."""
        self.nodes_dict: Dict[str, QueryGraphNode] = {}

    def add_node(self, node: QueryGraphNode):
        """Insert a new node into the graph."""
        self.nodes_dict[node.id] = node

    def get_node(self, node_id: str) -> Optional[QueryGraphNode]:
        """Look up a node by identifier, returning None for missing nodes."""
        return self.nodes_dict.get(node_id)

    def has_node(self, node_id: str) -> bool:
        """Test whether a node exists in the graph."""
        return node_id in self.nodes_dict

    def delete_node(self, node_id: str):
        """
        Remove a node and clean up all edges pointing to it.
        Updates neighbor lists to maintain graph consistency.
        """
        if node_id not in self.nodes_dict:
            return

        node = self.nodes_dict[node_id]

        for neighbor_id in node.links:
            if neighbor_id in self.nodes_dict:
                self.nodes_dict[neighbor_id]._remove_link(node_id)

        del self.nodes_dict[node_id]

    def add_edge(self, node_id1: str, node_id2: str):
        """
        Create a bidirectional connection between two nodes.
        Raises ValueError if either node doesn't exist.
        """
        if node_id1 not in self.nodes_dict or node_id2 not in self.nodes_dict:
            raise ValueError("Node does not exist, cannot add edge")

        self.nodes_dict[node_id1]._add_link(node_id2)
        self.nodes_dict[node_id2]._add_link(node_id1)

    def remove_edge(self, node_id1: str, node_id2: str):
        """Break the connection between two nodes if it exists."""
        if node_id1 in self.nodes_dict:
            self.nodes_dict[node_id1]._remove_link(node_id2)
        if node_id2 in self.nodes_dict:
            self.nodes_dict[node_id2]._remove_link(node_id1)

    def has_edge(self, node_id1: str, node_id2: str) -> bool:
        """Check whether an edge connects two nodes."""
        if node_id1 not in self.nodes_dict:
            return False
        return node_id2 in self.nodes_dict[node_id1].links

    def get_neighbors(self, node_id: str) -> List[QueryGraphNode]:
        """Retrieve all nodes directly connected to the specified node."""
        if node_id not in self.nodes_dict:
            return []

        node = self.nodes_dict[node_id]
        return [
            self.nodes_dict[neighbor_id]
            for neighbor_id in node.links
            if neighbor_id in self.nodes_dict
        ]

    def get_all_nodes(self) -> List[QueryGraphNode]:
        """Return every node in the graph as a list."""
        return list(self.nodes_dict.values())

    def get_node_count(self) -> int:
        """Count total nodes in the graph."""
        return len(self.nodes_dict)

    def get_edge_count(self) -> int:
        """
        Count total edges in the graph.
        Divides by 2 since each edge appears twice in adjacency lists.
        """
        total_links = sum(len(node.links) for node in self.nodes_dict.values())
        return total_links // 2

    def clear(self):
        """Wipe the entire graph, removing all nodes and edges."""
        self.nodes_dict.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export graph structure for serialization."""
        return {
            "nodes": [node.to_dict() for node in self.nodes_dict.values()]
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "QueryGraph":
        """Reconstruct graph from serialized data."""
        graph = QueryGraph()
        for node_data in data.get("nodes", []):
            node = QueryGraphNode.from_dict(node_data)
            graph.add_node(node)
        return graph

    def __repr__(self) -> str:
        """Display graph statistics."""
        return (
            f"QueryGraph(nodes={self.get_node_count()}, "
            f"edges={self.get_edge_count()})"
        )
