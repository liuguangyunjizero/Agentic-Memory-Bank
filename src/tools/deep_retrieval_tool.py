"""
Deep Retrieval工具
"""

import logging
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DeepRetrievalTool:
    """Deep Retrieval工具：读取Query Graph节点的完整Interaction Tree内容"""

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

    def __init__(self, interaction_tree, file_utils):
        """
        初始化Deep Retrieval工具

        Args:
            interaction_tree: InteractionTree 实例
            file_utils: FileUtils 实例
        """
        self.interaction_tree = interaction_tree
        self.file_utils = file_utils
        logger.info("DeepRetrievalTool initialized successfully")

    def call(self, params: Dict[str, Any]) -> str:
        """
        执行Deep Retrieval

        Args:
            params: {"node_id": str}

        Returns:
            str: JSON格式的完整Interaction Tree内容
        """
        node_id = params.get("node_id")

        if not node_id:
            error_msg = "Error: node_id parameter is required"
            logger.error(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

        logger.info(f"Executing Deep Retrieval: node_id={node_id[:8]}...")

        try:
            # 读取完整上下文文本
            text = self.interaction_tree.get_entry(node_id)

            if not text:
                warning_msg = f"No entry found for node_id: {node_id}"
                logger.warning(warning_msg)
                return json.dumps(
                    {"node_id": node_id, "text": None, "warning": warning_msg},
                    ensure_ascii=False,
                    indent=2
                )

            # 组装输出（直接返回文本）
            result = {
                "node_id": node_id,
                "text": text
            }

            logger.info(f"Deep Retrieval completed: 1 entry retrieved")
            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"Deep Retrieval failed: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

    def __repr__(self) -> str:
        """返回工具摘要"""
        return f"DeepRetrievalTool(name={self.name})"
