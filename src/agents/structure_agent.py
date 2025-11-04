"""
结构化 Agent

职责：对单个主题的内容进行结构化压缩

参考：规范文档第5.2节
"""

import logging
from typing import List
from dataclasses import dataclass
from src.agents.base_agent import BaseAgent
from src.prompts.agent_prompts import STRUCTURE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class StructureInput:
    """结构化 Agent 输入"""
    content: str  # 单个 cluster 的原始内容
    context: str  # cluster 的主题描述（参考）
    keywords: List[str]  # cluster 的关键词（参考）


@dataclass
class StructureOutput:
    """结构化 Agent 输出"""
    summary: str  # 结构化的详细摘要


class StructureAgent(BaseAgent):
    """
    结构化 Agent

    将原始内容压缩成结构化摘要
    """

    def __init__(self, llm_client):
        """
        初始化结构化 Agent

        Args:
            llm_client: LLMClient 实例
        """
        super().__init__(llm_client)
        logger.info("结构化Agent初始化完成")

    @classmethod
    def from_config(cls, llm_client, config) -> "StructureAgent":
        """从配置创建Agent"""
        return cls(llm_client=llm_client)

    def run(self, input_data: StructureInput) -> StructureOutput:
        """
        生成结构化摘要

        Args:
            input_data: StructureInput 实例

        Returns:
            StructureOutput 实例
        """
        prompt = self._build_prompt(input_data)

        logger.debug("调用LLM进行结构化...")
        response = self._call_llm(prompt)

        return self._parse_response(response)

    def _build_prompt(self, input_data: StructureInput) -> str:
        """
        构建 prompt

        Args:
            input_data: StructureInput 实例

        Returns:
            完整 prompt
        """
        return STRUCTURE_PROMPT.format(
            context=input_data.context,
            keywords=", ".join(input_data.keywords),
            content=input_data.content
        )

    def _parse_response(self, response: str) -> StructureOutput:
        """
        解析 LLM 响应

        Args:
            response: LLM 响应字符串

        Returns:
            StructureOutput 实例
        """
        try:
            data = self._parse_json_response(response)
            summary = data.get("summary", "")

            if not summary:
                logger.warning("LLM返回空摘要，使用原始响应")
                summary = response

            return StructureOutput(summary=summary)

        except Exception as e:
            logger.error(f"解析结构化响应失败: {str(e)}")
            # 返回原始响应作为摘要
            return StructureOutput(summary=response)
