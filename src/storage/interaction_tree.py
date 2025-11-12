"""
Interaction Tree - Interaction History Layer

Stores complete context text for each memory node.
"""

from typing import Dict, Any, Optional, List


class InteractionTree:
    """
    Interaction Tree manager.

    Minimalist design: only stores {node_id: text} mapping.
    Text contains complete context without distinguishing reasoning, tool calls, etc.
    """

    def __init__(self):
        """Initialize empty interaction tree."""
        self.node_entries: Dict[str, str] = {}

    def add_entry(self, node_id: str, text: str):
        """
        Add entry (saves complete context directly).

        Args:
            node_id: Associated node ID
            text: Complete context text
        """
        self.node_entries[node_id] = text

    def get_entry(self, node_id: str) -> Optional[str]:
        """
        Get node's complete context.

        Args:
            node_id: Node ID

        Returns:
            Complete context text, None if not found
        """
        return self.node_entries.get(node_id)

    def remove_entry(self, node_id: str):
        """Remove node's entry."""
        if node_id in self.node_entries:
            del self.node_entries[node_id]

    def get_total_entries(self) -> int:
        """Get total entry count."""
        return len(self.node_entries)

    def get_nodes_with_entries(self) -> List[str]:
        """Get all node IDs that have entries."""
        return list(self.node_entries.keys())

    def clear(self):
        """Clear all data (use with caution!)."""
        self.node_entries.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "node_entries": self.node_entries
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "InteractionTree":
        """Create InteractionTree from dictionary."""
        tree = InteractionTree()

        # New format: directly text
        if "node_entries" in data:
            for node_id, entry_data in data["node_entries"].items():
                if isinstance(entry_data, str):
                    tree.node_entries[node_id] = entry_data

        return tree

    def __repr__(self) -> str:
        """Return string representation."""
        return f"InteractionTree(entries={len(self.node_entries)})"
