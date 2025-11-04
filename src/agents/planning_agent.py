"""
计划 Agent

职责：采用增量式规划，每次只决定下一步任务

参考：规范文档第5.5节
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
    pending_tasks: List[str]  # 通常 0-1 个


class PlanningAgent(BaseAgent):
    """
    计划 Agent

    增量式规划策略，每次只决定下一步
    """

    def __init__(self, llm_client):
        """
        初始化计划 Agent

        Args:
            llm_client: LLMClient 实例
        """
        super().__init__(llm_client)
        logger.info("计划Agent初始化完成")

    @classmethod
    def from_config(cls, llm_client, config) -> "PlanningAgent":
        """从配置创建Agent"""
        return cls(llm_client=llm_client)

    def run(self, input_data: PlanningInput) -> PlanningOutput:
        """
        执行任务规划

        Args:
            input_data: PlanningInput 实例

        Returns:
            PlanningOutput 实例
        """
        prompt = self._build_prompt(input_data)

        logger.debug("调用LLM进行任务规划...")
        response = self._call_llm(prompt)

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

        # 格式化新记忆节点
        new_memory_str = "（无）"
        if input_data.new_memory_nodes:
            lines = [f"生成了 {len(input_data.new_memory_nodes)} 个新的记忆节点：\n"]
            for i, node_info in enumerate(input_data.new_memory_nodes, 1):
                if isinstance(node_info, str):
                    # 兼容旧格式（只有ID）
                    lines.append(f"{i}. 节点ID: {node_info}")
                else:
                    # 新格式（包含详细信息）
                    lines.append(f"{i}. 主题: {node_info.get('context', 'N/A')}")
                    lines.append(f"   关键词: {', '.join(node_info.get('keywords', []))}")
                    summary = node_info.get('summary', '')
                    if len(summary) > 150:
                        summary = summary[:150] + "..."
                    lines.append(f"   摘要: {summary}")
            new_memory_str = "\n".join(lines)

        # 格式化冲突通知
        conflict_str = "（无）"
        if input_data.conflict_notification:
            conflict_str = (
                f"检测到冲突：节点 {', '.join(input_data.conflict_notification.conflicting_node_ids[:2])} 等存在冲突\n"
                f"冲突描述：{input_data.conflict_notification.conflict_description}"
            )

        return PLANNING_PROMPT.format(
            task_goal=input_data.insight_doc.task_goal,
            completed_tasks=completed_tasks_str,
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
            pending_tasks = data.get("pending_tasks", [])

            # ✅ 修复：将字典列表转换为 CompletedTask 对象列表
            completed_tasks_data = data.get("completed_tasks", [])
            completed_tasks = [
                CompletedTask(
                    type=TaskType(task_dict.get("type", "NORMAL")),
                    description=task_dict.get("description", ""),
                    status=task_dict.get("status", "成功"),
                    context=task_dict.get("context", "")
                )
                for task_dict in completed_tasks_data
            ]

            logger.info(
                f"规划完成: 已完成={len(completed_tasks)}, 待办={len(pending_tasks)}"
            )

            return PlanningOutput(
                task_goal=task_goal,
                completed_tasks=completed_tasks,
                pending_tasks=pending_tasks
            )

        except Exception as e:
            logger.error(f"解析规划响应失败: {str(e)}")
            raise
