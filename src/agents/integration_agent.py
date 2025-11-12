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


@dataclass
class IntegrationInput:
    """Integration Agent input"""
    nodes_to_merge: List[NodeWithNeighbors]  # Conflicting nodes to merge
    validation_result: str  # Validation result from external framework


@dataclass
class IntegrationOutput:
    """Integration Agent output"""
    merged_node: Dict[str, Any]  # {"summary": ..., "context": ..., "keywords": [...]}
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

            merged_node = data.get("merged_node", {})
            description = data.get("merge_description", "Node merge")

            logger.info(f"Integration completed: new node generated")

            return IntegrationOutput(
                merged_node=merged_node,
                merge_description=description
            )

        except Exception as e:
            logger.error(f"Failed to parse integration response: {str(e)}")
            raise
