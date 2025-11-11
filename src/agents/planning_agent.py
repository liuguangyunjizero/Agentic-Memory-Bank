"""
计划 Agent

职责：采用增量式规划，每次只决定下一步任务
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
    """冲突通知"""
    conflicting_node_ids: List[str]
    conflict_description: str


@dataclass
class PlanningInput:
    """计划 Agent 输入"""
    insight_doc: InsightDoc  # 当前任务状态
    new_memory_nodes: Optional[List] = None  # 新生成的记忆
    conflict_notification: Optional[ConflictNotification] = None  # 冲突通知


@dataclass
class PlanningOutput:
    """计划 Agent 输出"""
    task_goal: str
    completed_tasks: List[CompletedTask]  # ✅ 修复：使用 CompletedTask 对象列表
    current_task: str  # 空字符串表示无任务


class PlanningAgent(BaseAgent):
    """
    计划 Agent

    增量式规划策略，每次只决定下一步
    """

    def __init__(self, llm_client, temperature: float = 0.6, top_p: float = 0.95):
        """
        初始化计划 Agent

        Args:
            llm_client: LLMClient 实例
            temperature: 温度参数
            top_p: 采样参数
        """
        super().__init__(llm_client)
        self.temperature = temperature
        self.top_p = top_p
        logger.info(f"Planning Agent initialized successfully (temp={temperature}, top_p={top_p})")

    @classmethod
    def from_config(cls, llm_client, config) -> "PlanningAgent":
        """从配置创建Agent"""
        return cls(
            llm_client=llm_client,
            temperature=config.PLANNING_AGENT_TEMPERATURE,
            top_p=config.PLANNING_AGENT_TOP_P
        )

    def run(self, input_data: PlanningInput) -> PlanningOutput:
        """
        执行任务规划

        Args:
            input_data: PlanningInput 实例

        Returns:
            PlanningOutput 实例
        """
        prompt = self._build_prompt(input_data)

        # 记录LLM输入
        logger.debug("="*80)
        logger.debug("📥 Planning Agent LLM Input:")
        logger.debug(prompt)
        logger.debug("="*80)

        response = self.llm_client.call(prompt, temperature=self.temperature, top_p=self.top_p, stop=None)

        # 记录LLM原始响应
        logger.debug("="*80)
        logger.debug("📤 Planning Agent LLM Raw Response:")
        logger.debug(response)
        logger.debug("="*80)

        return self._parse_response(response)

    def _build_prompt(self, input_data: PlanningInput) -> str:
        """
        构建 prompt

        Args:
            input_data: PlanningInput 实例

        Returns:
            完整 prompt
        """
        # 格式化已完成任务
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

        # 格式化当前待办任务（重要！）
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
                    # ✅ Fixed: Do NOT truncate summary - pass complete content
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
        解析 LLM 响应

        Args:
            response: LLM 响应字符串

        Returns:
            PlanningOutput 实例
        """
        try:
            data = self._parse_json_response(response)

            task_goal = data.get("task_goal", "")
            current_task = data.get("current_task", "")

            # ✅ 修复：将字典列表转换为 CompletedTask 对象列表
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
