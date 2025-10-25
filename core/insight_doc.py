"""
InsightDoc Module

Manages task execution state in a concise, structured format.
Serves as default context passed to external frameworks.
"""

from datetime import datetime
from typing import List, Optional
from core.models import CompletedTask
from utils.id_generator import generate_task_id


class InsightDoc:
    """
    InsightDoc: Task State Management Layer

    Manages:
    - Task goal (user's original question and understood version)
    - Completed subtasks with their results and impacts
    - Pending tasks
    - Current task being executed

    Designed to be concise (100-200 tokens) for passing to external frameworks.
    """

    def __init__(
        self,
        user_question: str,
        understood_goal: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        """
        Initialize InsightDoc for a new task.

        Args:
            user_question: User's original question/request
            understood_goal: Planning agent's interpretation (optional)
            task_id: Unique task ID (auto-generated if not provided)
        """
        self.task_id = task_id or generate_task_id()
        self.user_question = user_question
        self.understood_goal = understood_goal
        self.completed_tasks: List[CompletedTask] = []
        self.pending_tasks: List[str] = []
        self.current_task: Optional[str] = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def add_pending_task(self, task: str) -> None:
        """
        Add a new task to the pending list.

        Args:
            task: Task description
        """
        if task and task not in self.pending_tasks:
            self.pending_tasks.append(task)
            self.updated_at = datetime.now()

    def add_pending_tasks(self, tasks: List[str]) -> None:
        """
        Add multiple tasks to the pending list.

        Args:
            tasks: List of task descriptions
        """
        for task in tasks:
            self.add_pending_task(task)

    def set_current_task(self, task: str) -> None:
        """
        Set the current task being executed.

        If task is in pending_tasks, remove it from there.

        Args:
            task: Task description
        """
        self.current_task = task
        if task in self.pending_tasks:
            self.pending_tasks.remove(task)
        self.updated_at = datetime.now()

    def complete_current_task(
        self,
        result: str,
        impact: Optional[str] = None
    ) -> None:
        """
        Mark the current task as completed.

        Args:
            result: Result/conclusion of the task
            impact: Impact on subsequent tasks (optional)
        """
        if self.current_task:
            completed = CompletedTask(
                description=self.current_task,
                result=result,
                impact=impact
            )
            self.completed_tasks.append(completed)
            self.current_task = None
            self.updated_at = datetime.now()

    def get_next_task(self) -> Optional[str]:
        """
        Get the next pending task.

        Returns:
            str: Next task description, or None if no pending tasks
        """
        if self.pending_tasks:
            return self.pending_tasks[0]
        return None

    def get_completed_tasks(self) -> List[CompletedTask]:
        """
        Get all completed tasks.

        Returns:
            List[CompletedTask]: List of completed tasks
        """
        return self.completed_tasks.copy()

    def get_pending_tasks(self) -> List[str]:
        """
        Get all pending tasks.

        Returns:
            List[str]: List of pending task descriptions
        """
        return self.pending_tasks.copy()

    def insert_urgent_task(self, task: str, reason: str = "") -> None:
        """
        Insert a high-priority task at the beginning of pending list.

        Typically used for conflict resolution or critical issues.

        Args:
            task: Task description
            reason: Reason for urgency (optional)
        """
        full_task = f"{task} (Urgent: {reason})" if reason else task
        self.pending_tasks.insert(0, full_task)
        self.updated_at = datetime.now()

    def get_current_task_context(self) -> str:
        """
        Generate formatted context for external frameworks.

        Format:
            任务目标：{user_question}
            [理解版本：{understood_goal}]

            已完成的子任务：
            1. {description} → {result} → {impact}
            2. ...

            当前任务：{current_task}

            待办任务：
            - {pending_task_1}
            - {pending_task_2}

        Returns:
            str: Formatted context string
        """
        lines = []

        # Task goal
        lines.append(f"任务目标：{self.user_question}")
        if self.understood_goal:
            lines.append(f"理解版本：{self.understood_goal}")
        lines.append("")

        # Completed tasks
        if self.completed_tasks:
            lines.append("已完成的子任务：")
            for i, task in enumerate(self.completed_tasks, 1):
                lines.append(f"{i}. {task.format()}")
            lines.append("")

        # Current task
        if self.current_task:
            lines.append(f"当前任务：{self.current_task}")
            lines.append("")

        # Pending tasks
        if self.pending_tasks:
            lines.append("待办任务：")
            # Only show first 2 pending tasks to keep it concise
            for task in self.pending_tasks[:2]:
                lines.append(f"- {task}")
            if len(self.pending_tasks) > 2:
                lines.append(f"... (还有 {len(self.pending_tasks) - 2} 个待办任务)")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.

        Returns:
            dict: Serializable dictionary
        """
        return {
            'task_id': self.task_id,
            'user_question': self.user_question,
            'understood_goal': self.understood_goal,
            'completed_tasks': [task.to_dict() for task in self.completed_tasks],
            'pending_tasks': self.pending_tasks,
            'current_task': self.current_task,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'InsightDoc':
        """
        Create InsightDoc from dictionary.

        Args:
            data: Dictionary containing InsightDoc data

        Returns:
            InsightDoc: Reconstructed InsightDoc instance
        """
        doc = cls(
            user_question=data['user_question'],
            understood_goal=data.get('understood_goal'),
            task_id=data['task_id']
        )
        doc.completed_tasks = [
            CompletedTask.from_dict(task) for task in data['completed_tasks']
        ]
        doc.pending_tasks = data['pending_tasks']
        doc.current_task = data.get('current_task')
        doc.created_at = datetime.fromisoformat(data['created_at'])
        doc.updated_at = datetime.fromisoformat(data['updated_at'])
        return doc

    def __str__(self) -> str:
        """String representation for debugging."""
        return self.get_current_task_context()
