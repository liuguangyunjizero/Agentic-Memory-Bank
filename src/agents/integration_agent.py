"""
Memory Integration Agent

Responsibility: Generate integrated new node based on conflicting nodes and validation results
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.agents.base_agent import BaseAgent
from src.prompts.agent_prompts import INTEGRATION_PROMPT, format_nodes_to_merge

logger = logging.getLogger(__name__)


@dataclass
class NodeWithNeighbors:
    """Node with neighbor information"""
    id: str
    summary: str
    context: str
    keywords: List[str]
    neighbors: List[Dict[str, Any]]  # [{"id": ..., "context": ..., "keywords": [...]}, ...]
    merge_description: Optional[str] = None
    core_information: str = ""
    supporting_evidence: str = ""
    structure_summary: str = ""
    acquisition_logic: Optional[str] = None


@dataclass
class IntegrationInput:
    """Integration Agent input"""
    nodes_to_merge: List[NodeWithNeighbors]  # Conflicting nodes to merge
    validation_result: str  # Validation result from external framework


@dataclass
class IntegrationOutput:
    """Integration Agent output"""
    merged_node: Dict[str, Any]
    merge_description: str  # Description of merge operation


class IntegrationAgent(BaseAgent):
    """
    Memory Integration Agent

    Integrates content from multiple conflicting nodes
    """

    def __init__(self, llm_client, temperature: float = 0.2, top_p: float = 0.85):
        """
        Initialize Integration Agent

        Args:
            llm_client: LLMClient instance
            temperature: Temperature parameter
            top_p: Sampling parameter
        """
        super().__init__(llm_client)
        self.temperature = temperature
        self.top_p = top_p
        logger.info(f"Memory Integration Agent initialized successfully (temp={temperature}, top_p={top_p})")

    @classmethod
    def from_config(cls, llm_client, config) -> "IntegrationAgent":
        """Create Agent from config"""
        return cls(
            llm_client=llm_client,
            temperature=config.INTEGRATION_AGENT_TEMPERATURE,
            top_p=config.INTEGRATION_AGENT_TOP_P
        )

    def run(self, input_data: IntegrationInput) -> IntegrationOutput:
        """
        Integrate conflicting nodes

        Args:
            input_data: IntegrationInput instance

        Returns:
            IntegrationOutput instance
        """
        if not input_data.nodes_to_merge:
            raise ValueError("Node list to merge cannot be empty")

        prompt = self._build_prompt(input_data)

        # Log LLM input
        logger.debug("="*80)
        logger.debug("Integration Agent LLM Input:")
        logger.debug(prompt)
        logger.debug("="*80)

        response = self.llm_client.call(prompt, temperature=self.temperature, top_p=self.top_p, stop=None)

        # Log LLM raw response
        logger.debug("="*80)
        logger.debug("Integration Agent LLM Raw Response:")
        logger.debug(response)
        logger.debug("="*80)

        return self._parse_response(response)

    def _build_prompt(self, input_data: IntegrationInput) -> str:
        """
        Build prompt

        Args:
            input_data: IntegrationInput instance

        Returns:
            Complete prompt
        """
        # Format nodes to merge
        nodes = [
            {
                "id": node.id,
                "summary": node.summary,
                "context": node.context,
                "keywords": node.keywords,
                "core_information": node.core_information,
                "supporting_evidence": node.supporting_evidence,
                "structure_summary": node.structure_summary,
                "acquisition_logic": node.acquisition_logic,
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
        Parse LLM response

        Args:
            response: LLM response string

        Returns:
            IntegrationOutput instance
        """
        try:
            data = self._parse_json_response(response)

            merged_node = data.get("merged_node", {}) or {}
            description = data.get("merge_description", "Node merge")

            # Ensure merged node fields align with Structure Agent output
            context = merged_node.get("context", "General context").strip()
            keywords = merged_node.get("keywords") or []
            if not isinstance(keywords, list):
                keywords = []
            keywords = [kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip()]

            core_information = merged_node.get("core_information", "").strip()
            supporting_evidence = merged_node.get("supporting_evidence", "").strip()
            structure_summary = merged_node.get("structure_summary", "").strip()
            acquisition_logic = merged_node.get("acquisition_logic", "N/A").strip() or "N/A"
            summary = merged_node.get("summary", "").strip()

            summary_sections = []
            if core_information:
                summary_sections.append(f"**Core Information**:\n{core_information}")
            if structure_summary:
                summary_sections.append(f"**Structure Summary**:\n{structure_summary}")
            if supporting_evidence:
                summary_sections.append(f"**Supporting Evidence**:\n{supporting_evidence}")
            if acquisition_logic and acquisition_logic.upper() != "N/A":
                summary_sections.append(f"**Acquisition Logic**:\n{acquisition_logic}")
            if summary_sections:
                summary = "\n\n".join(summary_sections)
            elif not summary:
                summary = core_information or "Integrated summary"

            sanitized_node = {
                "summary": summary,
                "context": context,
                "keywords": keywords or ["general"],
                "core_information": core_information or summary,
                "supporting_evidence": supporting_evidence,
                "structure_summary": structure_summary or summary,
                "acquisition_logic": acquisition_logic
            }

            logger.info("Integration completed: new node generated")

            return IntegrationOutput(
                merged_node=sanitized_node,
                merge_description=description
            )

        except Exception as e:
            logger.error(f"Failed to parse integration response: {str(e)}")
            raise
