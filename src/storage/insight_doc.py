"""
Insight Doc - 任务状态层

以精简的结构化形式管理任务执行状态：
- 任务目标
- 已完成任务列表
- 待办任务列表（通常0-1个）

采用增量式规划，每次只决定下一步。

参考：规范文档第3.1节
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


class TaskType(Enum):
    """任务类型枚举"""
    NORMAL = "NORMAL"                   # 普通子任务
    CROSS_VALIDATE = "CROSS_VALIDATE"   # 交叉验证任务（冲突解决）


@dataclass
class CompletedTask:
    """
    已完成任务的记录

    Attributes:
        type: 任务类型（NORMAL 或 CROSS_VALIDATE）
        description: 任务详细描述
        status: 执行状态（"成功" 或 "失败"）
        context: 1-2句话的浓缩总结
    """
    type: TaskType
    description: str
    status: str  # "成功" 或 "失败"
    context: str  # 1-2句话的浓缩总结

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典（用于持久化）"""
        return {
            "type": self.type.value,
            "description": self.description,
            "status": self.status,
            "context": self.context
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CompletedTask":
        """从字典创建 CompletedTask"""
        return CompletedTask(
            type=TaskType(data["type"]),
            description=data["description"],
            status=data["status"],
            context=data["context"]
        )


@dataclass
class InsightDoc:
    """
    Insight Doc 主类

    任务状态层的核心数据结构，采用增量式规划理念：
    - pending_tasks 大多数时刻只有0-1个元素
    - 不维护优先级队列，每次只决定下一步

    Attributes:
        doc_id: 唯一标识符（用于临时存储映射）
        task_goal: 用户原始问题
        completed_tasks: 已完成的子任务列表
        pending_tasks: 待办任务列表（通常0-1个）
    """
    doc_id: str
    task_goal: str
    completed_tasks: List[CompletedTask] = field(default_factory=list)
    pending_tasks: List[str] = field(default_factory=list)

    def add_completed_task(
        self,
        task_type: TaskType,
        description: str,
        status: str,
        context: str
    ):
        """
        添加已完成的任务

        Args:
            task_type: 任务类型
            description: 任务描述
            status: 执行状态
            context: 浓缩总结
        """
        task = CompletedTask(
            type=task_type,
            description=description,
            status=status,
            context=context
        )
        self.completed_tasks.append(task)

    def set_pending_tasks(self, tasks: List[str]):
        """
        设置待办任务列表

        Args:
            tasks: 待办任务描述列表（通常0-1个）
        """
        self.pending_tasks = tasks

    def clear_pending_tasks(self):
        """清空待办任务列表"""
        self.pending_tasks = []

    def has_pending_tasks(self) -> bool:
        """
        检查是否有待办任务

        Returns:
            有待办任务返回 True，否则返回 False
        """
        return len(self.pending_tasks) > 0

    def is_all_completed_successfully(self) -> bool:
        """
        检查所有已完成任务是否都成功

        Returns:
            全部成功返回 True，否则返回 False
        """
        if not self.completed_tasks:
            return False
        return all(task.status == "成功" for task in self.completed_tasks)

    def to_dict(self) -> Dict[str, Any]:
        """
        导出为字典（用于持久化）

        Returns:
            包含所有字段的字典
        """
        return {
            "doc_id": self.doc_id,
            "task_goal": self.task_goal,
            "completed_tasks": [task.to_dict() for task in self.completed_tasks],
            "pending_tasks": self.pending_tasks
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "InsightDoc":
        """
        从字典创建 InsightDoc

        Args:
            data: 包含所有字段的字典

        Returns:
            InsightDoc 实例
        """
        return InsightDoc(
            doc_id=data["doc_id"],
            task_goal=data["task_goal"],
            completed_tasks=[
                CompletedTask.from_dict(task_data)
                for task_data in data.get("completed_tasks", [])
            ],
            pending_tasks=data.get("pending_tasks", [])
        )

    def get_summary(self) -> str:
        """
        获取任务状态摘要

        Returns:
            格式化的摘要字符串
        """
        summary_lines = [
            f"任务目标: {self.task_goal}",
            f"已完成任务数: {len(self.completed_tasks)}",
            f"待办任务数: {len(self.pending_tasks)}"
        ]

        if self.completed_tasks:
            success_count = sum(
                1 for task in self.completed_tasks if task.status == "成功"
            )
            summary_lines.append(f"成功/失败: {success_count}/{len(self.completed_tasks) - success_count}")

        return "\n".join(summary_lines)

    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"InsightDoc(doc_id={self.doc_id}, "
            f"completed={len(self.completed_tasks)}, "
            f"pending={len(self.pending_tasks)})"
        )
