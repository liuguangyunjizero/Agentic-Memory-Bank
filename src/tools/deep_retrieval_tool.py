"""
Deep Retrieval Tool

Retrieves complete Interaction Tree content for Query Graph nodes.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DeepRetrievalTool:
    """Deep Retrieval Tool: Read complete Interaction Tree content of Query Graph nodes"""

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
        Initialize Deep Retrieval Tool

        Args:
            interaction_tree: InteractionTree instance
        """
        self.interaction_tree = interaction_tree
        logger.info("DeepRetrievalTool initialized successfully")

    def call(self, params: Dict[str, Any]) -> str:
        """
        Execute Deep Retrieval

        Args:
            params: {"node_id": str}

        Returns:
            str: Complete Interaction Tree content (plain text)
        """
        node_id = params.get("node_id")

        if not node_id:
            error_msg = "Error: node_id parameter is required"
            logger.error(error_msg)
            return error_msg

        logger.info(f"Executing Deep Retrieval: node_id={node_id[:8]}...")

        try:
            # Read complete context text
            text = self.interaction_tree.get_entry(node_id)

            if not text:
                warning_msg = f"No entry found for node_id: {node_id}"
                logger.warning(warning_msg)
                return warning_msg

            logger.info(f"Deep Retrieval completed: 1 entry retrieved")
            return text  # Return text directly

        except Exception as e:
            error_msg = f"Deep Retrieval failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
