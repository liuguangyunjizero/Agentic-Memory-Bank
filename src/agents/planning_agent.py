"""
Task orchestration agent using incremental planning strategy.
Decides only the next action at each call based on current state and feedback.
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
    """Alert indicating detected contradictions requiring validation."""
    conflicting_node_ids: List[str]
    conflict_description: str


@dataclass
class PlanningInput:
    """Planning context including task history, new knowledge, and conflict signals."""
    insight_doc: InsightDoc
    new_memory_nodes: Optional[List] = None
    conflict_notification: Optional[ConflictNotification] = None


@dataclass
class PlanningOutput:
    """Next step decision with updated goal and task history."""
    task_goal: str
    completed_tasks: List[CompletedTask]
    current_task: str
    current_task_keywords: List[str]


class PlanningAgent(BaseAgent):
    """
    Generates single next-step decisions rather than full task sequences.
    Reacts to emerging information by adjusting strategy after each execution cycle.
    """

    def __init__(self, llm_client, temperature: float = 0.6, top_p: float = 0.95):
        """
        Configure reasoning parameters for adaptive decision making.
        Higher temperature allows creative responses to unexpected states.
        """
        super().__init__(llm_client)
        self.temperature = temperature
        self.top_p = top_p
        logger.info(f"Planning Agent initialized successfully (temp={temperature}, top_p={top_p})")

    @classmethod
    def from_config(cls, llm_client, config) -> "PlanningAgent":
        """Build agent from centralized configuration object."""
        return cls(
            llm_client=llm_client,
            temperature=config.PLANNING_AGENT_TEMPERATURE,
            top_p=config.PLANNING_AGENT_TOP_P
        )

    def run(self, input_data: PlanningInput) -> PlanningOutput:
        """
        Determine next action based on fresh execution state.
        Logs full prompt and response to trace planning decisions.
        """
        prompt = self._build_prompt(input_data)

        logger.debug("="*80)
        logger.debug("Planning Agent LLM Input:")
        logger.debug(prompt)
        logger.debug("="*80)

        response = self.llm_client.call(prompt, temperature=self.temperature, top_p=self.top_p, stop=None)

        logger.debug("="*80)
        logger.debug("Planning Agent LLM Raw Response:")
        logger.debug(response)
        logger.debug("="*80)

        return self._parse_response(response)

    def _build_prompt(self, input_data: PlanningInput) -> str:
        """
        Assemble current state into planning template.
        Includes full summaries without truncation to inform better decisions.
        Formats conflicts and new nodes to guide priority selection.
        """
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

        current_task_str = "(none)"
        if input_data.insight_doc.current_task:
            current_task_str = f"Currently executing: {input_data.insight_doc.current_task}"
        current_keywords = input_data.insight_doc.current_task_keywords
        current_keywords_str = ", ".join(current_keywords) if current_keywords else "(none)"

        new_memory_str = "(none)"
        if input_data.new_memory_nodes:
            lines = [f"Generated {len(input_data.new_memory_nodes)} new memory node(s):\n"]
            for i, node_info in enumerate(input_data.new_memory_nodes, 1):
                if isinstance(node_info, str):
                    lines.append(f"{i}. Node ID: {node_info}")
                else:
                    lines.append(f"{i}. Topic: {node_info.get('context', 'N/A')}")
                    lines.append(f"   Keywords: {', '.join(node_info.get('keywords', []))}")
                    summary = node_info.get('summary', '')
                    lines.append(f"   Summary: {summary}")
            new_memory_str = "\n".join(lines)

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
            current_task_keywords=current_keywords_str,
            new_memory_nodes=new_memory_str,
            conflict_notification=conflict_str
        )

    def _parse_response(self, response: str) -> PlanningOutput:
        """
        Extract next task and updated history from JSON response.
        Converts task dictionaries to CompletedTask objects for type safety.
        Raises exception on parse failure to signal planning breakdown.
        """
        try:
            data = self._parse_json_response(response)

            task_goal = data.get("task_goal", "")
            current_task = data.get("current_task", "")

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

            keywords = data.get("current_task_keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            keywords = [kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip()]

            return PlanningOutput(
                task_goal=task_goal,
                completed_tasks=completed_tasks,
                current_task=current_task,
                current_task_keywords=keywords
            )

        except Exception as e:
            logger.error(f"Failed to parse planning response: {str(e)}")
            raise
