"""
Graph Operations Module
"""

import logging
from typing import List, Set

logger = logging.getLogger(__name__)


class GraphOperations:
    """Graph operations module (encapsulates Query Graph CRUD operations)"""

    def __init__(self, graph, interaction_tree):
        """
        Initialize graph operations module

        Args:
            graph: QueryGraph instance
            interaction_tree: InteractionTree instance
        """
        from src.storage.query_graph import QueryGraph
        from src.storage.interaction_tree import InteractionTree

        if not isinstance(graph, QueryGraph):
            raise TypeError("graph must be a QueryGraph instance")
        if not isinstance(interaction_tree, InteractionTree):
            raise TypeError("interaction_tree must be an InteractionTree instance")

        self.graph = graph
        self.interaction_tree = interaction_tree
        logger.info("Graph operations module initialized successfully")

    def add_node(self, node) -> None:
        """
        Add node

        Args:
            node: QueryGraphNode instance
        """
        self.graph.add_node(node)

    def delete_node(self, node_id: str) -> None:
        """
        Delete node and all its edges

        Args:
            node_id: Node ID
        """
        if not self.graph.has_node(node_id):
            logger.warning(f"Node does not exist, cannot delete: {node_id[:8]}...")
            return

        self.graph.delete_node(node_id)
        # Also delete entry in InteractionTree
        self.interaction_tree.remove_entry(node_id)

    def add_edge(self, node_id1: str, node_id2: str) -> None:
        """
        Create related edge between two nodes

        Args:
            node_id1: First node ID
            node_id2: Second node ID
        """
        try:
            self.graph.add_edge(node_id1, node_id2)
        except ValueError as e:
            logger.error(f"Failed to add edge: {str(e)}")
            raise

    def remove_edge(self, node_id1: str, node_id2: str) -> None:
        """
        Remove edge

        Args:
            node_id1: First node ID
            node_id2: Second node ID
        """
        self.graph.remove_edge(node_id1, node_id2)

    def get_neighbors(self, node_id: str) -> List:
        """
        Get all neighbors of a node

        Args:
            node_id: Node ID

        Returns:
            List of neighbor nodes
        """
        return self.graph.get_neighbors(node_id)

    def merge_nodes(
        self,
        old_node_ids: List[str],
        new_node
    ) -> None:
        """
        Merge multiple nodes:
        1. Create new node (initially isolated, does not inherit edges)
        2. Delete old nodes

        Note: New node no longer automatically inherits old nodes' edges, needs re-analysis

        Args:
            old_node_ids: List of old node IDs to merge
            new_node: New merged node
        """
        # 1. Add new node (does not inherit any edges)
        self.add_node(new_node)

        # 2. Delete old nodes
        deleted_count = 0
        for old_node_id in old_node_ids:
            if self.graph.has_node(old_node_id):
                self.delete_node(old_node_id)
                deleted_count += 1
            else:
                logger.warning(f"Old node does not exist, skipping: {old_node_id[:8]}...")

        logger.info(
            f"Node merge completed: {deleted_count} old nodes -> new node {new_node.id[:8]}... "
            f"(new node starts with no edges)"
        )

    def get_node(self, node_id: str):
        """
        Get node

        Args:
            node_id: Node ID

        Returns:
            Node instance, or None if doesn't exist
        """
        return self.graph.get_node(node_id)

    def has_node(self, node_id: str) -> bool:
        """
        Check if node exists

        Args:
            node_id: Node ID

        Returns:
            True if exists, False otherwise
        """
        return self.graph.has_node(node_id)

    def has_edge(self, node_id1: str, node_id2: str) -> bool:
        """
        Check if edge exists between two nodes

        Args:
            node_id1: First node ID
            node_id2: Second node ID

        Returns:
            True if edge exists, False otherwise
        """
        return self.graph.has_edge(node_id1, node_id2)

    def __repr__(self) -> str:
        """Return module summary"""
        return f"GraphOperations(nodes={self.graph.get_node_count()}, edges={self.graph.get_edge_count()})"
