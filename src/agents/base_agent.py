"""
Agent Base Class

Base class for all LLM-driven Agents
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all agents"""

    def __init__(self, llm_client):
        """
        Initialize Agent

        Args:
            llm_client: LLMClient instance
        """
        from src.utils.llm_client import LLMClient

        # Type check (supports duck typing for testing)
        if not (isinstance(llm_client, LLMClient) or hasattr(llm_client, 'call')):
            raise TypeError("llm_client must be an LLMClient instance or an object with a call method")

        self.llm_client = llm_client

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM

        Args:
            prompt: Input prompt

        Returns:
            LLM response
        """
        try:
            response = self.llm_client.call(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise

    def _fix_json_string(self, json_str: str) -> str:
        """
        Attempt to fix common JSON string errors

        Args:
            json_str: Potentially malformed JSON string

        Returns:
            Fixed JSON string
        """
        # 1. Remove extra whitespace before/after JSON
        json_str = json_str.strip()

        # 2. Check and add missing closing braces
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)

        # 3. Check and add missing closing brackets
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)

        # 4. Attempt to fix trailing commas (JSON doesn't allow trailing commas)
        import re
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        return json_str

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response

        Args:
            response: String returned by LLM

        Returns:
            Parsed dictionary
        """
        try:
            # Try direct parsing
            return json.loads(response)
        except json.JSONDecodeError as e:
            # Try extracting ```json code block
            if "```json" in response:
                try:
                    parts = response.split("```json", 1)
                    if len(parts) > 1:
                        json_part = parts[1].split("```", 1)
                        if len(json_part) > 0:
                            json_str = json_part[0].strip()
                            # Try to fix JSON
                            json_str = self._fix_json_string(json_str)
                            return json.loads(json_str)
                except (IndexError, json.JSONDecodeError) as e:
                    pass

            # Try extracting ``` code block (without json marker)
            if "```" in response and "```json" not in response:
                try:
                    parts = response.split("```", 2)
                    if len(parts) >= 3:
                        json_str = parts[1].strip()
                        # Try to fix JSON
                        json_str = self._fix_json_string(json_str)
                        return json.loads(json_str)
                except (IndexError, json.JSONDecodeError) as e:
                    pass

            # Try extracting JSON object
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                try:
                    json_str = response[start:end]
                    # Try to fix JSON
                    json_str = self._fix_json_string(json_str)
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    pass

            # Try extracting array
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                try:
                    json_str = response[start:end]
                    # Try to fix JSON
                    json_str = self._fix_json_string(json_str)
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    pass

            # Improved: Show more detailed error info and response head/tail
            logger.error(f"JSON parsing failed, response length: {len(response)} characters")
            logger.error(f"First 500 characters of response: {response[:500]}")
            logger.error(f"Last 500 characters of response: {response[-500:]}")

            raise ValueError(f"Unable to parse JSON response")
