"""
Foundation class providing common functionality for all LLM-powered agents.
Handles LLM communication and robust JSON parsing with multiple fallback strategies.
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Shared infrastructure for agents that interact with language models.
    Provides JSON extraction and error recovery capabilities.
    """

    def __init__(self, llm_client):
        """
        Establish connection to the language model client.
        Accepts any object with a callable 'call' method for flexibility in testing.
        """
        from src.utils.llm_client import LLMClient

        if not (isinstance(llm_client, LLMClient) or hasattr(llm_client, 'call')):
            raise TypeError("llm_client must be an LLMClient instance or an object with a call method")

        self.llm_client = llm_client

    def _call_llm(self, prompt: str) -> str:
        """
        Execute a language model request and return the raw text response.
        Propagates exceptions to allow caller-specific error handling.
        """
        try:
            response = self.llm_client.call(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise

    def _fix_json_string(self, json_str: str) -> str:
        """
        Apply automatic repairs to common JSON syntax issues.
        Handles missing closing delimiters and trailing commas that LLMs often produce.
        """
        json_str = json_str.strip()

        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)

        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)

        import re
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        return json_str

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from LLM output with multiple fallback strategies.
        Tries direct parsing, code block extraction, and substring search in sequence.
        Provides detailed logging on failure to aid debugging.
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            if "```json" in response:
                try:
                    parts = response.split("```json", 1)
                    if len(parts) > 1:
                        json_part = parts[1].split("```", 1)
                        if len(json_part) > 0:
                            json_str = json_part[0].strip()
                            json_str = self._fix_json_string(json_str)
                            return json.loads(json_str)
                except (IndexError, json.JSONDecodeError) as e:
                    pass

            if "```" in response and "```json" not in response:
                try:
                    parts = response.split("```", 2)
                    if len(parts) >= 3:
                        json_str = parts[1].strip()
                        json_str = self._fix_json_string(json_str)
                        return json.loads(json_str)
                except (IndexError, json.JSONDecodeError) as e:
                    pass

            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                try:
                    json_str = response[start:end]
                    json_str = self._fix_json_string(json_str)
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    pass

            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                try:
                    json_str = response[start:end]
                    json_str = self._fix_json_string(json_str)
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    pass

            logger.error(f"JSON parsing failed, response length: {len(response)} characters")
            logger.error(f"First 500 characters of response: {response[:500]}")
            logger.error(f"Last 500 characters of response: {response[-500:]}")

            raise ValueError(f"Unable to parse JSON response")
