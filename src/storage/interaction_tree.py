"""
Interaction Tree - 交互历史层

以只读树形结构存储细粒度的原始交互数据：
- Entry：完整文本、时间戳、元数据、附件
- MergeEvent：节点合并事件记录
- 只读特性：不修改历史，只追加

设计目的：
- 完整的审计追踪
- 支持多模态附件（图片、文档、代码）
- 节点合并时保持历史不变

参考：规范文档第3.3节
"""

import uuid
import time
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


class AttachmentType(Enum):
    """附件类型枚举"""
    IMAGE = "image"
    DOCUMENT = "document"
    CODE = "code"


@dataclass
class Attachment:
    """
    多模态附件

    Attributes:
        id: 附件唯一标识符
        type: 附件类型（image/document/code）
        content: 文件路径（非文件内容！）
    """
    id: str
    type: AttachmentType
    content: str  # 文件路径

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Attachment":
        """从字典创建 Attachment"""
        return Attachment(
            id=data["id"],
            type=AttachmentType(data["type"]),
            content=data["content"]
        )


@dataclass
class InteractionEntry:
    """
    Interaction Tree 条目

    Attributes:
        entry_id: 条目唯一标识符（UUID）
        text: 完整文本（推理过程、工具调用摘要等）
        timestamp: 创建时间戳
        metadata: 元数据字典（source, tool_calls 等）
        attachments: 附件列表
    """
    entry_id: str
    text: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[Attachment] = field(default_factory=list)

    def add_attachment(self, attachment: Attachment):
        """
        添加附件

        Args:
            attachment: Attachment 实例
        """
        self.attachments.append(attachment)

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            "entry_id": self.entry_id,
            "text": self.text,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "attachments": [att.to_dict() for att in self.attachments]
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "InteractionEntry":
        """从字典创建 InteractionEntry"""
        return InteractionEntry(
            entry_id=data["entry_id"],
            text=data["text"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
            attachments=[
                Attachment.from_dict(att_data)
                for att_data in data.get("attachments", [])
            ]
        )

    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"InteractionEntry(id={self.entry_id[:8]}..., "
            f"text_len={len(self.text)}, "
            f"attachments={len(self.attachments)})"
        )


@dataclass
class MergeEvent:
    """
    节点合并事件记录

    Attributes:
        event_id: 事件唯一标识符（UUID）
        merged_node_ids: 被合并的节点ID列表
        new_node_id: 合并后的新节点ID
        timestamp: 合并时间戳
        description: 合并描述
    """
    event_id: str
    merged_node_ids: List[str]
    new_node_id: str
    timestamp: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            "event_id": self.event_id,
            "merged_node_ids": self.merged_node_ids,
            "new_node_id": self.new_node_id,
            "timestamp": self.timestamp,
            "description": self.description
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MergeEvent":
        """从字典创建 MergeEvent"""
        return MergeEvent(
            event_id=data["event_id"],
            merged_node_ids=data["merged_node_ids"],
            new_node_id=data["new_node_id"],
            timestamp=data["timestamp"],
            description=data["description"]
        )


class InteractionTree:
    """
    Interaction Tree 管理类

    只读特性：
    - 不修改历史 Entry 的 text 或 attachments
    - 不删除 Entry
    - 节点合并时，只新增 MergeEvent，原 Entry 保持不变
    """

    def __init__(self):
        """初始化空的交互树"""
        self.entries: Dict[str, InteractionEntry] = {}  # {entry_id: entry}
        self.node_to_entries: Dict[str, List[str]] = {}  # {node_id: [entry_ids]}
        self.merge_events: List[MergeEvent] = []  # 合并事件历史

    def add_entry(self, node_id: str, entry: InteractionEntry):
        """
        添加 Entry（只追加，不修改）

        Args:
            node_id: 关联的节点ID
            entry: InteractionEntry 实例
        """
        self.entries[entry.entry_id] = entry

        if node_id not in self.node_to_entries:
            self.node_to_entries[node_id] = []

        self.node_to_entries[node_id].append(entry.entry_id)

    def get_entries(self, node_id: str) -> List[InteractionEntry]:
        """
        读取节点的所有 Entry（按 timestamp 排序）

        Args:
            node_id: 节点ID

        Returns:
            Entry 列表（按时间升序排列）
        """
        entry_ids = self.node_to_entries.get(node_id, [])
        entries = [self.entries[eid] for eid in entry_ids if eid in self.entries]
        return sorted(entries, key=lambda e: e.timestamp)

    def get_entry_by_id(self, entry_id: str) -> Optional[InteractionEntry]:
        """
        通过ID获取单个Entry

        Args:
            entry_id: Entry ID

        Returns:
            InteractionEntry 实例，如果不存在则返回 None
        """
        return self.entries.get(entry_id)

    def record_merge(self, merge_event: MergeEvent):
        """
        记录节点合并事件

        合并逻辑：
        1. 记录合并事件
        2. 删除旧节点的 entries（因为merge都是cross_validate纠错，旧信息已无效）

        Args:
            merge_event: MergeEvent 实例
        """
        self.merge_events.append(merge_event)

        # 删除旧节点的entries（cross_validate纠错场景，旧信息基于错误前提，应该删除）
        for old_node_id in merge_event.merged_node_ids:
            if old_node_id in self.node_to_entries:
                del self.node_to_entries[old_node_id]

    def get_merge_events(self) -> List[MergeEvent]:
        """
        获取所有合并事件

        Returns:
            合并事件列表（按时间排序）
        """
        return sorted(self.merge_events, key=lambda e: e.timestamp)

    def get_total_entries(self) -> int:
        """
        获取 Entry 总数

        Returns:
            Entry 数量
        """
        return len(self.entries)

    def get_nodes_with_entries(self) -> List[str]:
        """
        获取所有有 Entry 的节点ID

        Returns:
            节点ID列表
        """
        return list(self.node_to_entries.keys())

    def clear(self):
        """清空所有数据（慎用！）"""
        self.entries.clear()
        self.node_to_entries.clear()
        self.merge_events.clear()

    def to_dict(self) -> Dict[str, Any]:
        """
        导出为字典

        Returns:
            包含所有数据的字典
        """
        return {
            "entries": {
                eid: entry.to_dict()
                for eid, entry in self.entries.items()
            },
            "node_to_entries": self.node_to_entries,
            "merge_events": [event.to_dict() for event in self.merge_events]
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "InteractionTree":
        """
        从字典创建 InteractionTree

        Args:
            data: 包含所有数据的字典

        Returns:
            InteractionTree 实例
        """
        tree = InteractionTree()

        # 恢复 entries
        for eid, entry_data in data.get("entries", {}).items():
            tree.entries[eid] = InteractionEntry.from_dict(entry_data)

        # 恢复 node_to_entries
        tree.node_to_entries = data.get("node_to_entries", {})

        # 恢复 merge_events
        tree.merge_events = [
            MergeEvent.from_dict(event_data)
            for event_data in data.get("merge_events", [])
        ]

        return tree

    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"InteractionTree(entries={len(self.entries)}, "
            f"nodes={len(self.node_to_entries)}, "
            f"merges={len(self.merge_events)})"
        )


def create_entry(
    text: str,
    node_id: str = None,
    metadata: Dict[str, Any] = None,
    attachments: List[Attachment] = None,
    entry_id: Optional[str] = None
) -> InteractionEntry:
    """
    创建新的 InteractionEntry（辅助函数）

    Args:
        text: 完整文本
        node_id: 关联的节点ID（可选）
        metadata: 元数据字典（可选）
        attachments: 附件列表（可选）
        entry_id: Entry ID（可选，不提供则自动生成UUID）

    Returns:
        InteractionEntry 实例
    """
    if entry_id is None:
        entry_id = str(uuid.uuid4())

    if metadata is None:
        metadata = {}

    if node_id:
        metadata["node_id"] = node_id

    return InteractionEntry(
        entry_id=entry_id,
        text=text,
        timestamp=time.time(),
        metadata=metadata,
        attachments=attachments or []
    )


def create_merge_event(
    merged_node_ids: List[str],
    new_node_id: str,
    description: str,
    event_id: Optional[str] = None
) -> MergeEvent:
    """
    创建新的 MergeEvent（辅助函数）

    Args:
        merged_node_ids: 被合并的节点ID列表
        new_node_id: 合并后的新节点ID
        description: 合并描述
        event_id: 事件ID（可选，不提供则自动生成UUID）

    Returns:
        MergeEvent 实例
    """
    if event_id is None:
        event_id = str(uuid.uuid4())

    return MergeEvent(
        event_id=event_id,
        merged_node_ids=merged_node_ids,
        new_node_id=new_node_id,
        timestamp=time.time(),
        description=description
    )
