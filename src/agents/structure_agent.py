"""
结构化 Agent

职责：对单个主题的内容进行结构化压缩
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
    current_task: str  # 当前子任务（帮助判断哪些信息对当前任务重要）


@dataclass
class StructureOutput:
    """结构化 Agent 输出"""
    summary: str  # 结构化的详细摘要


class StructureAgent(BaseAgent):
    """
    结构化 Agent

    将原始内容压缩成结构化摘要
    """

    def __init__(self, llm_client, temperature: float = 0.1, top_p: float = 0.8):
        """
        初始化结构化 Agent

        Args:
            llm_client: LLMClient 实例
            temperature: 温度参数（默认0.1，用于精确保留数据）
            top_p: 采样参数（默认0.8，用于精确保留数据）
        """
        super().__init__(llm_client)
        self.temperature = temperature
        self.top_p = top_p
        logger.info(f"Structure Agent initialized successfully (temp={temperature}, top_p={top_p})")

    @classmethod
    def from_config(cls, llm_client, config) -> "StructureAgent":
        """从配置创建Agent"""
        return cls(
            llm_client=llm_client,
            temperature=config.STRUCTURE_AGENT_TEMPERATURE,
            top_p=config.STRUCTURE_AGENT_TOP_P
        )

    def run(self, input_data: StructureInput) -> StructureOutput:
        """
        生成结构化摘要

        Args:
            input_data: StructureInput 实例

        Returns:
            StructureOutput 实例
        """
        prompt = self._build_prompt(input_data)

        # 使用配置的temperature和top_p来减少hallucination
        # 确保<answer>标签内容被准确复制
        response = self._call_llm_with_params(prompt, temperature=self.temperature, top_p=self.top_p)

        return self._parse_response(response)

    def _call_llm_with_params(self, prompt: str, temperature: float, top_p: float) -> str:
        """
        使用指定参数调用LLM

        Args:
            prompt: 输入prompt
            temperature: 温度参数
            top_p: 采样参数

        Returns:
            LLM响应
        """
        # 记录LLM输入
        logger.debug("="*80)
        logger.debug("📥 Structure Agent LLM Input:")
        logger.debug(prompt)
        logger.debug("="*80)

        try:
            response = self.llm_client.call(prompt, temperature=temperature, top_p=top_p, stop=None)

            # 记录LLM原始响应
            logger.debug("="*80)
            logger.debug("📤 Structure Agent LLM Raw Response:")
            logger.debug(response)
            logger.debug("="*80)

            return response
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise

    def _build_prompt(self, input_data: StructureInput) -> str:
        """
        构建 prompt

        Args:
            input_data: StructureInput 实例

        Returns:
            完整 prompt
        """
        return STRUCTURE_PROMPT.format(
            current_task=input_data.current_task,
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
                logger.warning("LLM returned empty summary, using raw response")
                summary = response

            return StructureOutput(summary=summary)

        except Exception as e:
            logger.error(f"Failed to parse structure response: {str(e)}")
            # 返回原始响应作为摘要
            return StructureOutput(summary=response)
