"""
Interaction Tree - 交互历史层
"""

from typing import Dict, Any, Optional


class InteractionTree:
    """
    Interaction Tree 管理类

    极简设计：只存储 {node_id: text} 映射
    text 是完整上下文（不区分推理、工具调用、响应等）
    """

    def __init__(self):
        """初始化空的交互树"""
        self.node_entries: Dict[str, str] = {}  # {node_id: text}

    def add_entry(self, node_id: str, text: str):
        """
        添加 Entry（直接保存完整上下文）

        Args:
            node_id: 关联的节点ID
            text: 完整上下文文本
        """
        self.node_entries[node_id] = text

    def get_entry(self, node_id: str) -> Optional[str]:
        """
        获取节点的完整上下文

        Args:
            node_id: 节点ID

        Returns:
            完整上下文文本，如果不存在则返回 None
        """
        return self.node_entries.get(node_id)

    def get_total_entries(self) -> int:
        """
        获取 Entry 总数

        Returns:
            Entry 数量
        """
        return len(self.node_entries)

    def get_nodes_with_entries(self):
        """
        获取所有有 Entry 的节点ID

        Returns:
            节点ID列表
        """
        return list(self.node_entries.keys())

    def clear(self):
        """清空所有数据（慎用！）"""
        self.node_entries.clear()

    def to_dict(self) -> Dict[str, Any]:
        """
        导出为字典

        Returns:
            包含所有数据的字典
        """
        return {
            "node_entries": self.node_entries  # 直接导出 {node_id: text}
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "InteractionTree":
        """
        从字典创建 InteractionTree（支持多种格式向后兼容）

        Args:
            data: 包含所有数据的字典

        Returns:
            InteractionTree 实例
        """
        tree = InteractionTree()

        # 新格式：{node_id: text} 或 {node_id: {...}}
        if "node_entries" in data:
            for node_id, entry_data in data["node_entries"].items():
                if isinstance(entry_data, str):
                    # 新格式：直接是文本
                    tree.node_entries[node_id] = entry_data
                elif isinstance(entry_data, dict):
                    # 中间格式：entry 对象
                    tree.node_entries[node_id] = entry_data.get("text", "")

        # 旧格式兼容：从 node_to_entries 和 entries 转换
        elif "node_to_entries" in data and "entries" in data:
            for node_id, entry_ids in data["node_to_entries"].items():
                if entry_ids:
                    # 取第一个 entry（旧系统每个节点只有一个）
                    entry_data = data["entries"].get(entry_ids[0])
                    if entry_data and isinstance(entry_data, dict):
                        tree.node_entries[node_id] = entry_data.get("text", "")

        return tree

    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return f"InteractionTree(entries={len(self.node_entries)})"
