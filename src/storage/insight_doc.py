"""
Task state management layer that tracks execution progress through incremental planning.
Records completed work and maintains the current active task for the system to execute.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


class TaskType(Enum):
    """Categorizes tasks into regular information gathering or conflict resolution."""
    NORMAL = "NORMAL"
    CROSS_VALIDATE = "CROSS_VALIDATE"


@dataclass
class CompletedTask:
    """
    Represents a finished unit of work with execution outcome.
    Stores enough information to inform future planning without repeating work.
    """
    type: TaskType
    description: str
    status: str
    context: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "description": self.description,
            "status": self.status,
            "context": self.context
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CompletedTask":
        """Reconstruct from dictionary loaded from storage."""
        return CompletedTask(
            type=TaskType(data["type"]),
            description=data["description"],
            status=data["status"],
            context=data["context"]
        )


@dataclass
class InsightDoc:
    """
    Central state tracker for task execution using one-step-at-a-time planning.
    Maintains history of what's been done and a single current task to work on next.
    """
    doc_id: str
    task_goal: str
    completed_tasks: List[CompletedTask] = field(default_factory=list)
    current_task: str = ""
    current_task_keywords: List[str] = field(default_factory=list)
    task_node_map: Dict[str, List[str]] = field(default_factory=dict)
    failed_tasks: Dict[str, str] = field(default_factory=dict)

    def add_completed_task(
        self,
        task_type: TaskType,
        description: str,
        status: str,
        context: str
    ):
        """Append a finished task to the completion history."""
        task = CompletedTask(
            type=task_type,
            description=description,
            status=status,
            context=context
        )
        self.completed_tasks.append(task)

    def set_current_task(self, task: str):
        """Update what the system should work on next, or clear to indicate completion."""
        self.current_task = task

    def to_dict(self) -> Dict[str, Any]:
        """Package entire state for export to JSON."""
        return {
            "doc_id": self.doc_id,
            "task_goal": self.task_goal,
            "completed_tasks": [task.to_dict() for task in self.completed_tasks],
            "current_task": self.current_task,
            "current_task_keywords": self.current_task_keywords,
            "task_node_map": self.task_node_map,
            "failed_tasks": self.failed_tasks
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "InsightDoc":
        """Recreate state object from deserialized JSON data."""
        return InsightDoc(
            doc_id=data["doc_id"],
            task_goal=data["task_goal"],
            completed_tasks=[
                CompletedTask.from_dict(task_data)
                for task_data in data.get("completed_tasks", [])
            ],
            current_task=data.get("current_task", ""),
            current_task_keywords=data.get("current_task_keywords", []),
            task_node_map=data.get("task_node_map", {}),
            failed_tasks=data.get("failed_tasks", {})
        )

    def register_task_node(self, task_description: str, node_id: str) -> None:
        """Link a graph node to the task that created it for traceability."""
        task_key = task_description or self.task_goal or "default"
        nodes = self.task_node_map.setdefault(task_key, [])
        if node_id not in nodes:
            nodes.append(node_id)

    def get_tasks_for_nodes(self, node_ids: List[str]) -> List[str]:
        """Find which tasks generated the specified nodes."""
        result = []
        node_set = set(node_ids)
        for task_desc, nodes in self.task_node_map.items():
            if any(n in node_set for n in nodes):
                result.append(task_desc)
        return result

    def mark_task_failed(self, task_description: str, reason: str = "") -> None:
        """
        Flag a task as having failed to prevent using its outputs.
        Removes associated nodes from the mapping to avoid invalid references.
        """
        if not task_description:
            return
        self.failed_tasks[task_description] = reason or "Failed"
        for task in self.completed_tasks:
            if task.description == task_description:
                task.status = "Failed"
                if reason:
                    task.context = reason
        if task_description in self.task_node_map:
            del self.task_node_map[task_description]

    def enforce_failed_statuses(self, tasks: List[CompletedTask]) -> List[CompletedTask]:
        """Apply failure state to tasks even if planner tries to change their status."""
        if not self.failed_tasks:
            return tasks
        for task in tasks:
            reason = self.failed_tasks.get(task.description)
            if reason is not None:
                task.status = "Failed"
                if reason:
                    task.context = reason
        return tasks

    def __repr__(self) -> str:
        """Provide quick summary of doc state."""
        return (
            f"InsightDoc(doc_id={self.doc_id}, "
            f"completed={len(self.completed_tasks)}, "
            f"current_task={'Yes' if self.current_task else 'No'})"
        )
