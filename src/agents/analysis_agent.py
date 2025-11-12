"""
Memory Analysis Agent

Responsibility: Determine relationship between new node and existing nodes (conflict/related/unrelated)
"""

import logging
from typing import List, Optional
from dataclasses import dataclass
from src.agents.base_agent import BaseAgent
from src.prompts.agent_prompts import ANALYSIS_PROMPT, format_candidates

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Node information (without embedding)"""
    id: Optional[str] = None  # New node has no id
    summary: str = ""
    context: str = ""
    keywords: List[str] = None
    merge_description: Optional[str] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


@dataclass
class Relationship:
    """Node relationship"""
    existing_node_id: str
    relationship: str  # "conflict" | "related" | "unrelated"
    reasoning: str  # Reasoning for determination

    # conflict-specific field
    conflict_description: Optional[str] = None


@dataclass
class AnalysisInput:
    """Analysis Agent input"""
    new_node: NodeInfo  # New node
    candidate_nodes: List[NodeInfo]  # Candidate nodes


@dataclass
class AnalysisOutput:
    """Analysis Agent output"""
    relationships: List[Relationship]


class AnalysisAgent(BaseAgent):
    """
    Memory Analysis Agent

    Determination priority: conflict > related > unrelated
    """

    def __init__(self, llm_client, temperature: float = 0.4, top_p: float = 0.9):
        """
        Initialize Analysis Agent

        Args:
            llm_client: LLMClient instance
            temperature: Temperature parameter
            top_p: Sampling parameter
        """
        super().__init__(llm_client)
        self.temperature = temperature
        self.top_p = top_p
        logger.info(f"Memory Analysis Agent initialized successfully (temp={temperature}, top_p={top_p})")

    @classmethod
    def from_config(cls, llm_client, config) -> "AnalysisAgent":
        """Create Agent from config"""
        return cls(
            llm_client=llm_client,
            temperature=config.ANALYSIS_AGENT_TEMPERATURE,
            top_p=config.ANALYSIS_AGENT_TOP_P
        )

    def run(self, input_data: AnalysisInput) -> AnalysisOutput:
        """
        Analyze node relationships

        Args:
            input_data: AnalysisInput instance

        Returns:
            AnalysisOutput instance
        """
        if not input_data.candidate_nodes:
            logger.warning("Candidate node list is empty, returning empty relationships")
            return AnalysisOutput(relationships=[])

        prompt = self._build_prompt(input_data)

        # Log LLM input
        logger.debug("="*80)
        logger.debug("Analysis Agent LLM Input:")
        logger.debug(prompt)
        logger.debug("="*80)

        response = self.llm_client.call(prompt, temperature=self.temperature, top_p=self.top_p, stop=None)

        # Log LLM raw response
        logger.debug("="*80)
        logger.debug("Analysis Agent LLM Raw Response:")
        logger.debug(response)
        logger.debug("="*80)

        return self._parse_response(response)

    def _build_prompt(self, input_data: AnalysisInput) -> str:
        """
        Build prompt

        Args:
            input_data: AnalysisInput instance

        Returns:
            Complete prompt
        """
        # Format candidate nodes
        candidates = [
            {
                "id": node.id,
                "summary": node.summary,
                "context": node.context,
                "keywords": node.keywords,
                "merge_description": node.merge_description
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
        Parse LLM response

        Args:
            response: LLM response string

        Returns:
            AnalysisOutput instance
        """
        try:
            data = self._parse_json_response(response)

            # If single object returned, convert to list
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
            # Return default unrelated relationship
            return AnalysisOutput(relationships=[])
