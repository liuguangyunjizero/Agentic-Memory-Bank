"""
Planning Agent

Responsibility: Incremental planning, decides only the next task each time
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from src.agents.base_agent import BaseAgent
from src.storage.insight_doc import InsightDoc, TaskType, CompletedTask
from src.prompts.agent_prompts import PLANNING_PROMPT, format_completed_tasks

logger = logging.getLogger(__name__)


@dataclass
class ConflictNotification:
    """Conflict notification"""
    conflicting_node_ids: List[str]
    conflict_description: str


@dataclass
class PlanningInput:
    """Planning Agent input"""
    insight_doc: InsightDoc  # Current task state
    new_memory_nodes: Optional[List] = None  # Newly generated memory
    conflict_notification: Optional[ConflictNotification] = None  # Conflict notification


@dataclass
class PlanningOutput:
    """Planning Agent output"""
    task_goal: str
    completed_tasks: List[CompletedTask]  # List of CompletedTask objects
    current_task: str  # Empty string means no task


class PlanningAgent(BaseAgent):
    """
    Planning Agent

    Incremental planning strategy, decides only the next step each time
    """

    def __init__(self, llm_client, temperature: float = 0.6, top_p: float = 0.95):
        """
        Initialize Planning Agent

        Args:
            llm_client: LLMClient instance
            temperature: Temperature parameter
            top_p: Sampling parameter
        """
        super().__init__(llm_client)
        self.temperature = temperature
        self.top_p = top_p
        logger.info(f"Planning Agent initialized successfully (temp={temperature}, top_p={top_p})")

    @classmethod
    def from_config(cls, llm_client, config) -> "PlanningAgent":
        """Create Agent from config"""
        return cls(
            llm_client=llm_client,
            temperature=config.PLANNING_AGENT_TEMPERATURE,
            top_p=config.PLANNING_AGENT_TOP_P
        )

    def run(self, input_data: PlanningInput) -> PlanningOutput:
        """
        Execute task planning

        Args:
            input_data: PlanningInput instance

        Returns:
            PlanningOutput instance
        """
        prompt = self._build_prompt(input_data)

        # Log LLM input
        logger.debug("="*80)
        logger.debug("Planning Agent LLM Input:")
        logger.debug(prompt)
        logger.debug("="*80)

        response = self.llm_client.call(prompt, temperature=self.temperature, top_p=self.top_p, stop=None)

        # Log LLM raw response
        logger.debug("="*80)
        logger.debug("Planning Agent LLM Raw Response:")
        logger.debug(response)
        logger.debug("="*80)

        return self._parse_response(response)

    def _build_prompt(self, input_data: PlanningInput) -> str:
        """
        Build prompt

        Args:
            input_data: PlanningInput instance

        Returns:
            Complete prompt
        """
        # Format completed tasks
        completed_tasks = [
            {
                "type": task.type.value,
                "description": task.description,
                "status": task.status,
                "context": task.context
            }
            for task in input_data.insight_doc.completed_tasks
        ]
        completed_tasks_str = format_completed_tasks(completed_tasks)

        # Format current pending task (important!)
        current_task_str = "(none)"
        if input_data.insight_doc.current_task:
            current_task_str = f"Currently executing: {input_data.insight_doc.current_task}"

        # Format new memory nodes
        new_memory_str = "(none)"
        if input_data.new_memory_nodes:
            lines = [f"Generated {len(input_data.new_memory_nodes)} new memory node(s):\n"]
            for i, node_info in enumerate(input_data.new_memory_nodes, 1):
                if isinstance(node_info, str):
                    # Compatible with old format (ID only)
                    lines.append(f"{i}. Node ID: {node_info}")
                else:
                    # New format (with detailed information)
                    lines.append(f"{i}. Topic: {node_info.get('context', 'N/A')}")
                    lines.append(f"   Keywords: {', '.join(node_info.get('keywords', []))}")
                    # Do NOT truncate summary - pass complete content
                    summary = node_info.get('summary', '')
                    lines.append(f"   Summary: {summary}")
            new_memory_str = "\n".join(lines)

        # Format conflict notification
        conflict_str = "(none)"
        if input_data.conflict_notification:
            conflict_str = (
                f"Conflict detected: nodes {', '.join(input_data.conflict_notification.conflicting_node_ids[:2])} etc. have conflicts\n"
                f"Conflict description: {input_data.conflict_notification.conflict_description}"
            )

        return PLANNING_PROMPT.format(
            task_goal=input_data.insight_doc.task_goal,
            completed_tasks=completed_tasks_str,
            current_task=current_task_str,
            new_memory_nodes=new_memory_str,
            conflict_notification=conflict_str
        )

    def _parse_response(self, response: str) -> PlanningOutput:
        """
        Parse LLM response

        Args:
            response: LLM response string

        Returns:
            PlanningOutput instance
        """
        try:
            data = self._parse_json_response(response)

            task_goal = data.get("task_goal", "")
            current_task = data.get("current_task", "")

            # Convert dictionary list to CompletedTask object list
            completed_tasks_data = data.get("completed_tasks", [])
            completed_tasks = [
                CompletedTask(
                    type=TaskType(task_dict.get("type", "NORMAL")),
                    description=task_dict.get("description", ""),
                    status=task_dict.get("status", "success"),
                    context=task_dict.get("context", "")
                )
                for task_dict in completed_tasks_data
            ]

            logger.info(
                f"Planning completed: completed={len(completed_tasks)}, current_task={'yes' if current_task else 'no'}"
            )

            return PlanningOutput(
                task_goal=task_goal,
                completed_tasks=completed_tasks,
                current_task=current_task
            )

        except Exception as e:
            logger.error(f"Failed to parse planning response: {str(e)}")
            raise
