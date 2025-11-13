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
    content: str  # Original content chunk (verbatim)


@dataclass
class StructureOutput:
    """Structure Agent output"""
    summary: str  # Structured detailed summary
    context: str
    keywords: List[str]
    core_information: str
    supporting_evidence: str
    structure_summary: str
    acquisition_logic: str


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
        return STRUCTURE_PROMPT.format(content=input_data.content)

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

            context = data.get("context", "").strip()
            keywords = data.get("keywords") or []
            if not isinstance(keywords, list):
                keywords = []
            keywords = [kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip()]

            core_information = data.get("core_information", "").strip()
            supporting_evidence = data.get("supporting_evidence", "").strip()
            structure_summary = data.get("structure_summary", "").strip()
            acquisition_logic = data.get("acquisition_logic", "N/A").strip() or "N/A"

            summary_sections = []
            if core_information:
                summary_sections.append(f"**Core Information**:\n{core_information}")
            if structure_summary:
                summary_sections.append(f"**Structure Summary**:\n{structure_summary}")
            if supporting_evidence:
                summary_sections.append(f"**Supporting Evidence**:\n{supporting_evidence}")
            if acquisition_logic and acquisition_logic.upper() != "N/A":
                summary_sections.append(f"**Acquisition Logic**:\n{acquisition_logic}")

            summary = "\n\n".join(summary_sections).strip()
            if not summary:
                summary = response

            return StructureOutput(
                summary=summary,
                context=context or "General context",
                keywords=keywords or ["general"],
                core_information=core_information or summary,
                supporting_evidence=supporting_evidence,
                structure_summary=structure_summary or summary,
                acquisition_logic=acquisition_logic
            )

        except Exception as e:
            logger.error(f"Failed to parse structure response: {str(e)}")
            # Return raw response as summary
            return StructureOutput(
                summary=response,
                context="General context",
                keywords=["general"],
                core_information=response,
                supporting_evidence="",
                structure_summary=response,
                acquisition_logic="N/A"
            )
