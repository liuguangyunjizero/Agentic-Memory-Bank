"""
Encapsulates common graph manipulation operations in a single interface.
Coordinates changes across both the graph structure and interaction tree.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class GraphOperations:
    """
    Provides high-level operations that maintain consistency between
    the query graph and its associated interaction tree.
    """

    def __init__(self, graph, interaction_tree):
        """
        Bind to specific graph and tree instances for coordinated updates.
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
        """Insert a node into the graph."""
        self.graph.add_node(node)

    def delete_node(self, node_id: str) -> None:
        """
        Remove a node from both the graph and interaction tree.
        Cleans up all edges automatically via graph's delete logic.
        """
        if not self.graph.has_node(node_id):
            logger.warning(f"Node does not exist, cannot delete: {node_id[:8]}...")
            return

        self.graph.delete_node(node_id)
        self.interaction_tree.remove_entry(node_id)

    def add_edge(self, node_id1: str, node_id2: str) -> None:
        """
        Create a bidirectional relationship between two nodes.
        Raises ValueError if either node doesn't exist.
        """
        try:
            self.graph.add_edge(node_id1, node_id2)
        except ValueError as e:
            logger.error(f"Failed to add edge: {str(e)}")
            raise

    def get_neighbors(self, node_id: str) -> List:
        """Retrieve all nodes directly connected to the specified node."""
        return self.graph.get_neighbors(node_id)

    def merge_nodes(
        self,
        old_node_ids: List[str],
        new_node
    ) -> None:
        """
        Replace multiple nodes with a single merged node.
        The new node starts isolated - edges must be re-established via analysis.
        This allows the system to determine fresh relationships after merging.
        """
        self.add_node(new_node)

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
