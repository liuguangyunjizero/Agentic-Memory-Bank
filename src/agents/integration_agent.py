"""
记忆整合 Agent

职责：基于冲突节点和验证结果，生成整合后的新节点

参考：规范文档第5.4节
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from src.agents.base_agent import BaseAgent
from src.prompts.agent_prompts import INTEGRATION_PROMPT, format_nodes_to_merge

logger = logging.getLogger(__name__)


@dataclass
class NodeWithNeighbors:
    """包含邻居信息的节点"""
    id: str
    summary: str
    context: str
    keywords: List[str]
    neighbors: List[Dict[str, Any]]  # [{"id": ..., "context": ..., "keywords": [...]}, ...]


@dataclass
class IntegrationInput:
    """整合 Agent 输入"""
    nodes_to_merge: List[NodeWithNeighbors]  # 待合并的冲突节点
    validation_result: str  # 外部框架的验证结果


@dataclass
class IntegrationOutput:
    """整合 Agent 输出"""
    merged_node: Dict[str, Any]  # {"summary": ..., "context": ..., "keywords": [...]}
    neighbor_updates: Dict[str, Dict]  # {neighbor_id: {"context": ..., "keywords": [...]}}
    interaction_tree_description: str  # 合并操作描述


class IntegrationAgent(BaseAgent):
    """
    记忆整合 Agent

    整合多个冲突节点的内容
    """

    def __init__(self, llm_client, temperature: float = 0.2, top_p: float = 0.85):
        """
        初始化整合 Agent

        Args:
            llm_client: LLMClient 实例
            temperature: 温度参数
            top_p: 采样参数
        """
        super().__init__(llm_client)
        self.temperature = temperature
        self.top_p = top_p
        logger.info(f"记忆整合Agent初始化完成 (temp={temperature}, top_p={top_p})")

    @classmethod
    def from_config(cls, llm_client, config) -> "IntegrationAgent":
        """从配置创建Agent"""
        return cls(
            llm_client=llm_client,
            temperature=config.INTEGRATION_AGENT_TEMPERATURE,
            top_p=config.INTEGRATION_AGENT_TOP_P
        )

    def run(self, input_data: IntegrationInput) -> IntegrationOutput:
        """
        整合冲突节点

        Args:
            input_data: IntegrationInput 实例

        Returns:
            IntegrationOutput 实例
        """
        if not input_data.nodes_to_merge:
            raise ValueError("待合并节点列表不能为空")

        prompt = self._build_prompt(input_data)

        logger.debug(f"调用LLM整合 {len(input_data.nodes_to_merge)} 个节点 "
                    f"(temp={self.temperature}, top_p={self.top_p})...")
        response = self.llm_client.call(prompt, temperature=self.temperature, top_p=self.top_p)

        # 记录LLM原始响应
        logger.debug("="*80)
        logger.debug("📤 Integration Agent LLM原始响应:")
        logger.debug(response)
        logger.debug("="*80)

        return self._parse_response(response)

    def _build_prompt(self, input_data: IntegrationInput) -> str:
        """
        构建 prompt

        Args:
            input_data: IntegrationInput 实例

        Returns:
            完整 prompt
        """
        # 格式化待合并节点
        nodes = [
            {
                "id": node.id,
                "summary": node.summary,
                "context": node.context,
                "keywords": node.keywords,
                "neighbors": node.neighbors
            }
            for node in input_data.nodes_to_merge
        ]
        nodes_str = format_nodes_to_merge(nodes)

        return INTEGRATION_PROMPT.format(
            validation_result=input_data.validation_result,
            nodes_to_merge=nodes_str
        )

    def _parse_response(self, response: str) -> IntegrationOutput:
        """
        解析 LLM 响应

        Args:
            response: LLM 响应字符串

        Returns:
            IntegrationOutput 实例
        """
        try:
            data = self._parse_json_response(response)

            merged_node = data.get("merged_node", {})
            neighbor_updates = data.get("neighbor_updates", {})
            description = data.get("interaction_tree_description", "节点合并")

            logger.info(f"整合完成: 生成新节点，{len(neighbor_updates)} 个邻居需要更新")

            return IntegrationOutput(
                merged_node=merged_node,
                neighbor_updates=neighbor_updates,
                interaction_tree_description=description
            )

        except Exception as e:
            logger.error(f"解析整合响应失败: {str(e)}")
            raise
