"""
Compression agent that transforms raw chunks into structured node records.
Extracts key information, evidence, and reasoning patterns for graph storage.
"""

import logging
from typing import List
from dataclasses import dataclass
from src.agents.base_agent import BaseAgent
from src.prompts.agent_prompts import STRUCTURE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class StructureInput:
    """Raw content chunk ready for structured extraction."""
    content: str


@dataclass
class StructureOutput:
    """Structured node containing extracted semantic components."""
    summary: str
    context: str
    keywords: List[str]
    core_information: str
    supporting_evidence: str
    structure_summary: str
    acquisition_logic: str


class StructureAgent(BaseAgent):
    """
    Transforms verbatim chunks into queryable graph nodes.
    Uses very low temperature to ensure faithful extraction without invention.
    """

    def __init__(self, llm_client, temperature: float = 0.1, top_p: float = 0.8):
        """
        Configure extraction parameters for maximum precision.
        Low temperature minimizes hallucination risk when copying answer blocks.
        """
        super().__init__(llm_client)
        self.temperature = temperature
        self.top_p = top_p
        logger.info(f"Structure Agent initialized successfully (temp={temperature}, top_p={top_p})")

    @classmethod
    def from_config(cls, llm_client, config) -> "StructureAgent":
        """Build agent from centralized configuration object."""
        return cls(
            llm_client=llm_client,
            temperature=config.STRUCTURE_AGENT_TEMPERATURE,
            top_p=config.STRUCTURE_AGENT_TOP_P
        )

    def run(self, input_data: StructureInput) -> StructureOutput:
        """
        Execute structured extraction on a single chunk.
        Returns populated node ready for graph insertion.
        """
        prompt = self._build_prompt(input_data)

        response = self._call_llm_with_params(prompt, temperature=self.temperature, top_p=self.top_p)

        return self._parse_response(response)

    def _call_llm_with_params(self, prompt: str, temperature: float, top_p: float) -> str:
        """
        Send extraction request with detailed logging for debugging.
        Logs both input prompt and raw response to trace extraction issues.
        """
        logger.debug("="*80)
        logger.debug("Structure Agent LLM Input:")
        logger.debug(prompt)
        logger.debug("="*80)

        try:
            response = self.llm_client.call(prompt, temperature=temperature, top_p=top_p, stop=None)

            logger.debug("="*80)
            logger.debug("Structure Agent LLM Raw Response:")
            logger.debug(response)
            logger.debug("="*80)

            return response
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise

    def _build_prompt(self, input_data: StructureInput) -> str:
        """Inject chunk content into the extraction template."""
        return STRUCTURE_PROMPT.format(content=input_data.content)

    def _parse_response(self, response: str) -> StructureOutput:
        """
        Extract and validate structured fields from JSON response.
        Falls back to using raw response as summary if parsing fails.
        Ensures all required fields have default values to prevent downstream errors.
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
            return StructureOutput(
                summary=response,
                context="General context",
                keywords=["general"],
                core_information=response,
                supporting_evidence="",
                structure_summary=response,
                acquisition_logic="N/A"
            )
