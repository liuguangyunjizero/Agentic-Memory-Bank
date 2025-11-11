"""
记忆分析 Agent

职责：判断新节点与现有节点的关系（conflict/related/unrelated）
"""

import logging
from typing import List, Optional
from dataclasses import dataclass
from src.agents.base_agent import BaseAgent
from src.prompts.agent_prompts import ANALYSIS_PROMPT, format_candidates

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """节点信息（不包含 embedding）"""
    id: Optional[str] = None  # 新节点无 id
    summary: str = ""
    context: str = ""
    keywords: List[str] = None
    merge_description: Optional[str] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


@dataclass
class Relationship:
    """节点关系"""
    existing_node_id: str
    relationship: str  # "conflict" | "related" | "unrelated"
    reasoning: str  # 判断理由

    # conflict 特有字段
    conflict_description: Optional[str] = None


@dataclass
class AnalysisInput:
    """分析 Agent 输入"""
    new_node: NodeInfo  # 新节点
    candidate_nodes: List[NodeInfo]  # 候选节点


@dataclass
class AnalysisOutput:
    """分析 Agent 输出"""
    relationships: List[Relationship]


class AnalysisAgent(BaseAgent):
    """
    记忆分析 Agent

    判断优先级：conflict > related > unrelated
    """

    def __init__(self, llm_client, temperature: float = 0.4, top_p: float = 0.9):
        """
        初始化分析 Agent

        Args:
            llm_client: LLMClient 实例
            temperature: 温度参数
            top_p: 采样参数
        """
        super().__init__(llm_client)
        self.temperature = temperature
        self.top_p = top_p
        logger.info(f"Memory Analysis Agent initialized successfully (temp={temperature}, top_p={top_p})")

    @classmethod
    def from_config(cls, llm_client, config) -> "AnalysisAgent":
        """从配置创建Agent"""
        return cls(
            llm_client=llm_client,
            temperature=config.ANALYSIS_AGENT_TEMPERATURE,
            top_p=config.ANALYSIS_AGENT_TOP_P
        )

    def run(self, input_data: AnalysisInput) -> AnalysisOutput:
        """
        分析节点关系

        Args:
            input_data: AnalysisInput 实例

        Returns:
            AnalysisOutput 实例
        """
        if not input_data.candidate_nodes:
            logger.warning("Candidate node list is empty, returning empty relationships")
            return AnalysisOutput(relationships=[])

        prompt = self._build_prompt(input_data)

        # 记录LLM输入
        logger.debug("="*80)
        logger.debug("📥 Analysis Agent LLM Input:")
        logger.debug(prompt)
        logger.debug("="*80)

        response = self.llm_client.call(prompt, temperature=self.temperature, top_p=self.top_p, stop=None)

        # 记录LLM原始响应
        logger.debug("="*80)
        logger.debug("📤 Analysis Agent LLM Raw Response:")
        logger.debug(response)
        logger.debug("="*80)

        return self._parse_response(response)

    def _build_prompt(self, input_data: AnalysisInput) -> str:
        """
        构建 prompt

        Args:
            input_data: AnalysisInput 实例

        Returns:
            完整 prompt
        """
        # 格式化候选节点
        candidates = [
            {
                "id": node.id,
                "summary": node.summary,
                "context": node.context,
                "keywords": node.keywords
            }
            for node in input_data.candidate_nodes
        ]
        candidates_str = format_candidates(candidates)

        return ANALYSIS_PROMPT.format(
            new_summary=input_data.new_node.summary,
            new_context=input_data.new_node.context,
            new_keywords=", ".join(input_data.new_node.keywords),
            candidates=candidates_str
        )

    def _parse_response(self, response: str) -> AnalysisOutput:
        """
        解析 LLM 响应

        Args:
            response: LLM 响应字符串

        Returns:
            AnalysisOutput 实例
        """
        try:
            data = self._parse_json_response(response)

            # 如果返回的是单个对象，转换为列表
            if isinstance(data, dict):
                data = [data]

            relationships = []
            for rel_data in data:
                relationship = Relationship(
                    existing_node_id=rel_data.get("existing_node_id", ""),
                    relationship=rel_data.get("relationship", "unrelated"),
                    reasoning=rel_data.get("reasoning", ""),
                    conflict_description=rel_data.get("conflict_description")
                )
                relationships.append(relationship)

            logger.info(
                f"Analysis completed: {sum(1 for r in relationships if r.relationship == 'conflict')} conflict, "
                f"{sum(1 for r in relationships if r.relationship == 'related')} related, "
                f"{sum(1 for r in relationships if r.relationship == 'unrelated')} unrelated"
            )

            return AnalysisOutput(relationships=relationships)

        except Exception as e:
            logger.error(f"Failed to parse analysis response: {str(e)}")
            # 返回默认的无关关系
            return AnalysisOutput(relationships=[])
