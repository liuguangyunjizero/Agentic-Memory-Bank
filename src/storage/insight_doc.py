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

    def clear_current_task(self):
        """Clear the current task."""
        self.current_task = ""

    def has_current_task(self) -> bool:
        """Check if there is a current task."""
        return bool(self.current_task)

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for persistence."""
        return {
            "doc_id": self.doc_id,
            "task_goal": self.task_goal,
            "completed_tasks": [task.to_dict() for task in self.completed_tasks],
            "current_task": self.current_task
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
            current_task=data.get("current_task", "")
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"InsightDoc(doc_id={self.doc_id}, "
            f"completed={len(self.completed_tasks)}, "
            f"current_task={'Yes' if self.current_task else 'No'})"
        )
