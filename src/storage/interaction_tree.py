"""
Simple storage layer that preserves complete original context for each memory node.
Enables deep retrieval when summaries are insufficient for detailed analysis.
"""

from typing import Dict, Any, Optional, List


class InteractionTree:
    """
    Flat key-value store mapping node IDs to their full context text.
    Designed for simplicity - no hierarchy or structure beyond basic lookup.
    """

    def __init__(self):
        """Initialize with empty storage."""
        self.node_entries: Dict[str, str] = {}

    def add_entry(self, node_id: str, text: str):
        """
        Store complete context text associated with a node.
        Overwrites any existing entry for the same ID.
        """
        self.node_entries[node_id] = text

    def get_entry(self, node_id: str) -> Optional[str]:
        """
        Retrieve the full context for a node.
        Returns None if no entry exists for the given ID.
        """
        return self.node_entries.get(node_id)

    def remove_entry(self, node_id: str):
        """Delete the context entry for a node if it exists."""
        if node_id in self.node_entries:
            del self.node_entries[node_id]

    def get_total_entries(self) -> int:
        """Count how many context entries are stored."""
        return len(self.node_entries)

    def clear(self):
        """Wipe all stored context entries from memory."""
        self.node_entries.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Package all entries for JSON export."""
        return {
            "node_entries": self.node_entries
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "InteractionTree":
        """
        Rebuild tree from deserialized data.
        Handles simple string entries for each node ID.
        """
        tree = InteractionTree()

        if "node_entries" in data:
            for node_id, entry_data in data["node_entries"].items():
                if isinstance(entry_data, str):
                    tree.node_entries[node_id] = entry_data

        return tree

    def __repr__(self) -> str:
        """Show entry count for quick status check."""
        return f"InteractionTree(entries={len(self.node_entries)})"
