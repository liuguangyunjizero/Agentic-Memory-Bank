"""
记忆分析 Agent

职责：判断新节点与现有节点的关系（conflict/related/unrelated）

参考：规范文档第5.3节
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

    # related 特有字段
    context_update_new: Optional[str] = None
    context_update_existing: Optional[str] = None
    keywords_update_new: Optional[List[str]] = None
    keywords_update_existing: Optional[List[str]] = None


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

    def __init__(self, llm_client):
        """
        初始化分析 Agent

        Args:
            llm_client: LLMClient 实例
        """
        super().__init__(llm_client)
        logger.info("记忆分析Agent初始化完成")

    @classmethod
    def from_config(cls, llm_client, config) -> "AnalysisAgent":
        """从配置创建Agent"""
        return cls(llm_client=llm_client)

    def run(self, input_data: AnalysisInput) -> AnalysisOutput:
        """
        分析节点关系

        Args:
            input_data: AnalysisInput 实例

        Returns:
            AnalysisOutput 实例
        """
        if not input_data.candidate_nodes:
            logger.warning("候选节点列表为空，返回空关系")
            return AnalysisOutput(relationships=[])

        prompt = self._build_prompt(input_data)

        logger.debug(f"调用LLM分析 {len(input_data.candidate_nodes)} 个候选节点的关系...")
        response = self._call_llm(prompt)

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
                    conflict_description=rel_data.get("conflict_description"),
                    context_update_new=rel_data.get("context_update_new"),
                    context_update_existing=rel_data.get("context_update_existing"),
                    keywords_update_new=rel_data.get("keywords_update_new"),
                    keywords_update_existing=rel_data.get("keywords_update_existing")
                )
                relationships.append(relationship)

            logger.info(
                f"分析完成: {sum(1 for r in relationships if r.relationship == 'conflict')} conflict, "
                f"{sum(1 for r in relationships if r.relationship == 'related')} related, "
                f"{sum(1 for r in relationships if r.relationship == 'unrelated')} unrelated"
            )

            return AnalysisOutput(relationships=relationships)

        except Exception as e:
            logger.error(f"解析分析响应失败: {str(e)}")
            # 返回默认的无关关系
            return AnalysisOutput(relationships=[])
