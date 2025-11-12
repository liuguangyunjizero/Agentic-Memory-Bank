"""
Structure Agent

Responsibility: Structured compression of content for a single topic
"""

import logging
from typing import List
from dataclasses import dataclass
from src.agents.base_agent import BaseAgent
from src.prompts.agent_prompts import STRUCTURE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class StructureInput:
    """Structure Agent input"""
    content: str  # Original content of a single cluster
    context: str  # Topic description of cluster (for reference)
    keywords: List[str]  # Keywords of cluster (for reference)
    current_task: str  # Current subtask (helps determine which info is important for current task)


@dataclass
class StructureOutput:
    """Structure Agent output"""
    summary: str  # Structured detailed summary


class StructureAgent(BaseAgent):
    """
    Structure Agent

    Compresses original content into structured summary
    """

    def __init__(self, llm_client, temperature: float = 0.1, top_p: float = 0.8):
        """
        Initialize Structure Agent

        Args:
            llm_client: LLMClient instance
            temperature: Temperature parameter (default 0.1, for precise data preservation)
            top_p: Sampling parameter (default 0.8, for precise data preservation)
        """
        super().__init__(llm_client)
        self.temperature = temperature
        self.top_p = top_p
        logger.info(f"Structure Agent initialized successfully (temp={temperature}, top_p={top_p})")

    @classmethod
    def from_config(cls, llm_client, config) -> "StructureAgent":
        """Create Agent from config"""
        return cls(
            llm_client=llm_client,
            temperature=config.STRUCTURE_AGENT_TEMPERATURE,
            top_p=config.STRUCTURE_AGENT_TOP_P
        )

    def run(self, input_data: StructureInput) -> StructureOutput:
        """
        Generate structured summary

        Args:
            input_data: StructureInput instance

        Returns:
            StructureOutput instance
        """
        prompt = self._build_prompt(input_data)

        # Use configured temperature and top_p to reduce hallucination
        # Ensure <answer> tag content is accurately copied
        response = self._call_llm_with_params(prompt, temperature=self.temperature, top_p=self.top_p)

        return self._parse_response(response)

    def _call_llm_with_params(self, prompt: str, temperature: float, top_p: float) -> str:
        """
        Call LLM with specified parameters

        Args:
            prompt: Input prompt
            temperature: Temperature parameter
            top_p: Sampling parameter

        Returns:
            LLM response
        """
        # Log LLM input
        logger.debug("="*80)
        logger.debug("Structure Agent LLM Input:")
        logger.debug(prompt)
        logger.debug("="*80)

        try:
            response = self.llm_client.call(prompt, temperature=temperature, top_p=top_p, stop=None)

            # Log LLM raw response
            logger.debug("="*80)
            logger.debug("Structure Agent LLM Raw Response:")
            logger.debug(response)
            logger.debug("="*80)

            return response
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise

    def _build_prompt(self, input_data: StructureInput) -> str:
        """
        Build prompt

        Args:
            input_data: StructureInput instance

        Returns:
            Complete prompt
        """
        return STRUCTURE_PROMPT.format(
            current_task=input_data.current_task,
            context=input_data.context,
            keywords=", ".join(input_data.keywords),
            content=input_data.content
        )

    def _parse_response(self, response: str) -> StructureOutput:
        """
        Parse LLM response

        Args:
            response: LLM response string

        Returns:
            StructureOutput instance
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
            # Return raw response as summary
            return StructureOutput(summary=response)
