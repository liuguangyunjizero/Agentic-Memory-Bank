"""
Tool for accessing complete original context stored in the interaction tree.
Provides full text retrieval when summaries don't contain sufficient detail.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DeepRetrievalTool:
    """
    Fetches uncompressed original context for graph nodes.
    Used when agents need more detail than what's in the structured summary.
    """

    name = "deep_retrieval"
    description = "Read the complete Interaction Tree content of a Query Graph node."

    parameters = {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "description": "The ID of the Query Graph node to retrieve"
            }
        },
        "required": ["node_id"]
    }

    def __init__(self, interaction_tree):
        """
        Bind to an interaction tree instance for context lookups.
        """
        self.interaction_tree = interaction_tree
        logger.info("DeepRetrievalTool initialized successfully")

    def call(self, params: Dict[str, Any]) -> str:
        """
        Look up and return the full original text for the specified node.
        Returns error message if node not found.
        """
        node_id = params.get("node_id")

        if not node_id:
            error_msg = "Error: node_id parameter is required"
            logger.error(error_msg)
            return error_msg

        logger.info(f"Executing Deep Retrieval: node_id={node_id[:8]}...")

        try:
            text = self.interaction_tree.get_entry(node_id)

            if not text:
                warning_msg = f"No entry found for node_id: {node_id}"
                logger.warning(warning_msg)
                return warning_msg

            logger.info(f"Deep Retrieval completed: 1 entry retrieved")
            return text

        except Exception as e:
            error_msg = f"Deep Retrieval failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
