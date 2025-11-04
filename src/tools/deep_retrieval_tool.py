"""
Deep Retrieval工具

从Query Graph节点读取完整的Interaction Tree内容，包括所有Entry和附件。

参考：规范文档第6.1节
"""

import logging
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DeepRetrievalTool:
    """Deep Retrieval工具：读取Query Graph节点的完整Interaction Tree内容"""

    name = "deep_retrieval"
    description = "Read the complete Interaction Tree content of a Query Graph node, including all entries and attachments."

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
        logger.info("DeepRetrievalTool初始化完成")

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

        logger.info(f"执行Deep Retrieval: node_id={node_id[:8]}...")

        try:
            # 1. 读取所有Entry
            entries = self.interaction_tree.get_entries(node_id)

            if not entries:
                warning_msg = f"No entries found for node_id: {node_id}"
                logger.warning(warning_msg)
                return json.dumps(
                    {"node_id": node_id, "entries": [], "warning": warning_msg},
                    ensure_ascii=False,
                    indent=2
                )

            # 2. 组装输出
            result = {"node_id": node_id, "entries": []}

            for entry in entries:
                entry_data = {
                    "entry_id": entry.entry_id,
                    "text": entry.text,
                    "timestamp": entry.timestamp,
                    "metadata": entry.metadata,
                    "attachments": []
                }


                # 3. 处理附件：读取实际文件内容
                for attachment in entry.attachments:
                    file_content = self._load_file_content(attachment)
                    entry_data["attachments"].append({
                        "id": attachment.id,
                        "type": attachment.type.value,
                        "content": attachment.content,  # 文件路径
                        "file_content": file_content     # 实际内容
                    })

                result["entries"].append(entry_data)

            logger.info(f"Deep Retrieval完成: {len(result['entries'])} 个Entry")
            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"Deep Retrieval failed: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

    def _load_file_content(self, attachment) -> str:
        """
        根据附件类型加载文件内容

        Args:
            attachment: Attachment 实例

        Returns:
            文件内容（文本或Base64编码）
        """
        file_path = attachment.content

        try:
            if attachment.type.value == "image":
                # 图片：返回Base64编码
                return self.file_utils.read_image_as_base64(file_path)
            elif attachment.type.value == "document":
                # 文档：返回文本内容
                return self.file_utils.read_document(file_path)
            elif attachment.type.value == "code":
                # 代码：返回文本内容
                return self.file_utils.read_text(file_path)
            else:
                logger.warning(f"未知的附件类型: {attachment.type.value}")
                return ""

        except Exception as e:
            logger.error(f"读取附件失败 {file_path}: {str(e)}")
            return f"[Error loading file: {str(e)}]"

    def __repr__(self) -> str:
        """返回工具摘要"""
        return f"DeepRetrievalTool(name={self.name})"
