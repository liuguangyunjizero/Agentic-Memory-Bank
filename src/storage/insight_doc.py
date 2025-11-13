"""
Insight Doc - Task State Layer

Manages task planning and execution state using incremental planning strategy.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


class TaskType(Enum):
    """Task type enumeration."""
    NORMAL = "NORMAL"
    CROSS_VALIDATE = "CROSS_VALIDATE"


@dataclass
class CompletedTask:
    """
    Record of a completed task.

    Attributes:
        type: Task type (NORMAL or CROSS_VALIDATE)
        description: Detailed task description
        status: Execution status ("success" or "failure")
        context: Condensed 1-2 sentence summary
    """
    type: TaskType
    description: str
    status: str
    context: str

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for persistence."""
        return {
            "type": self.type.value,
            "description": self.description,
            "status": self.status,
            "context": self.context
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CompletedTask":
        """Create CompletedTask from dictionary."""
        return CompletedTask(
            type=TaskType(data["type"]),
            description=data["description"],
            status=data["status"],
            context=data["context"]
        )


@dataclass
class InsightDoc:
    """
    Insight Doc main class - core data structure for task state layer.

    Uses incremental planning:
    - current_task is either empty string (no task) or a single task string
    - No priority queue maintained; decides next step each iteration

    Attributes:
        doc_id: Unique identifier for temporary storage mapping
        task_goal: Original user question
        completed_tasks: List of completed subtasks
        current_task: Current pending task (empty string means no task)
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
        """Add a completed task to the list."""
        task = CompletedTask(
            type=task_type,
            description=description,
            status=status,
            context=context
        )
        self.completed_tasks.append(task)

    def set_current_task(self, task: str):
        """Set the current task (empty string for no task)."""
        self.current_task = task

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for persistence."""
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
        """Create InsightDoc from dictionary."""
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
        """Associate a node with the task description."""
        task_key = task_description or self.task_goal or "default"
        nodes = self.task_node_map.setdefault(task_key, [])
        if node_id not in nodes:
            nodes.append(node_id)

    def get_tasks_for_nodes(self, node_ids: List[str]) -> List[str]:
        """Return task descriptions that produced any of the given nodes."""
        result = []
        node_set = set(node_ids)
        for task_desc, nodes in self.task_node_map.items():
            if any(n in node_set for n in nodes):
                result.append(task_desc)
        return result

    def mark_task_failed(self, task_description: str, reason: str = "") -> None:
        """Mark a task as failed and update its context."""
        if not task_description:
            return
        self.failed_tasks[task_description] = reason or "Failed"
        for task in self.completed_tasks:
            if task.description == task_description:
                task.status = "Failed"
                if reason:
                    task.context = reason
        # Remove obsolete node mapping to avoid reusing invalidated context
        if task_description in self.task_node_map:
            del self.task_node_map[task_description]

    def enforce_failed_statuses(self, tasks: List[CompletedTask]) -> List[CompletedTask]:
        """Ensure tasks marked failed stay failed even after planner updates."""
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
        """Return string representation."""
        return (
            f"InsightDoc(doc_id={self.doc_id}, "
            f"completed={len(self.completed_tasks)}, "
            f"current_task={'Yes' if self.current_task else 'No'})"
        )
