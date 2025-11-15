"""
Web page content extraction tool using Jina Reader API with LLM-based filtering.
Fetches clean markdown content and extracts information relevant to the specified goal.
"""

import logging
import json
from typing import Dict, Any
import requests

logger = logging.getLogger(__name__)


class VisitTool:
    """
    Retrieves and processes web page content with intelligent extraction.
    Uses Jina's reader service for clean markdown conversion and optional LLM filtering.
    """

    name = "visit"
    description = "Visit a webpage and extract relevant information based on your goal."

    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
            },
            "goal": {
                "type": "string",
                "description": "What you want to find on this webpage"
            }
        },
        "required": ["url", "goal"]
    }

    def __init__(self, llm_client=None, jina_api_key: str = None, max_content_length: int = 95000, temperature: float = 0.2, top_p: float = 0.85):
        """
        Initialize with API credentials and LLM parameters for extraction.
        Low temperature ensures focused, relevant extraction.
        """
        if not jina_api_key or jina_api_key == "your-jina-api-key-here":
            raise ValueError(
                "Jina API key not configured. Please set JINA_API_KEY in .env file.\n"
                "Register at: https://jina.ai/"
            )

        self.llm_client = llm_client
        self.jina_api_key = jina_api_key
        self.max_content_length = max_content_length
        self.temperature = temperature
        self.top_p = top_p

        logger.info(f"VisitTool initialized successfully (using Jina Reader API, temp={temperature}, top_p={top_p})")

    def call(self, params: Dict[str, Any]) -> str:
        """
        Fetch and process one or more web pages based on extraction goal.
        Returns structured JSON with relevant content and evidence.
        """
        url = params.get("url")
        goal = params.get("goal", "")

        if not url:
            return "Error: url parameter is required"

        if isinstance(url, str):
            logger.info(f"Visiting webpage: {url}")
            return self._visit_single_url(url, goal)
        else:
            logger.info(f"Visiting {len(url)} webpages")
            results = []
            for u in url:
                try:
                    result = self._visit_single_url(u, goal)
                    results.append(result)
                except Exception as e:
                    error_msg = f"Error visiting {u}: {str(e)}"
                    logger.warning(error_msg)
                    results.append(json.dumps({"url": u, "error": error_msg}, ensure_ascii=False))

            return "\n=======\n".join(results)

    def _visit_single_url(self, url: str, goal: str) -> str:
        """
        Process a single web page through Jina Reader and LLM extraction pipeline.
        """
        content = self._jina_reader(url, goal)

        output = {
            "url": url,
            "goal": goal,
            "content": content,
            "content_length": len(content)
        }

        logger.info(f"Webpage visit completed: {len(content)} characters")
        return json.dumps(output, ensure_ascii=False, indent=2)

    def _truncate_to_tokens(self, text: str, max_tokens: int = 95000) -> str:
        """
        Limit content length using token-based counting via tiktoken.
        Falls back to character-based estimation if tiktoken unavailable.
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")

            tokens = encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text

            logger.warning(f"Content too long ({len(tokens)} tokens), truncating to {max_tokens} tokens")
            truncated_tokens = tokens[:max_tokens]
            return encoding.decode(truncated_tokens)

        except ImportError:
            logger.debug("tiktoken not available, using character-based truncation")
            max_chars = max_tokens * 3
            if len(text) <= max_chars:
                return text
            logger.warning(f"Content too long, truncating to ~{max_tokens} tokens ({max_chars} chars)")
            return text[:max_chars]

    def _jina_reader(self, url: str, goal: str) -> str:
        """
        Fetch page via Jina Reader API and optionally extract relevant content with LLM.
        Retries failed requests up to 3 times before raising error.
        """
        max_retries = 3
        timeout = 60

        last_error = None
        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.jina_api_key}",
                    "X-Return-Format": "markdown"
                }

                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout
                )

                if response.status_code == 200:
                    webpage_content = response.text

                    webpage_content = self._truncate_to_tokens(webpage_content, max_tokens=95000)

                    if self.llm_client and goal:
                        extracted_content = self._extract_with_llm(webpage_content, goal, url)
                        return extracted_content

                    return webpage_content

                else:
                    error_msg = f"Jina Reader returned error status code: {response.status_code}"
                    logger.warning(f"Attempt {attempt+1}/{max_retries}: {error_msg}")
                    last_error = ValueError(error_msg)

            except Exception as e:
                logger.warning(f"Jina Reader attempt {attempt+1}/{max_retries} failed: {str(e)}")
                last_error = e

        error_msg = f"Failed to visit {url} (after {max_retries} retries): {str(last_error)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from last_error

    def _extract_with_llm(self, full_text: str, goal: str, url: str = "") -> str:
        """
        Use LLM to identify and extract goal-relevant information from full page content.
        Returns structured evidence and summary formatted for easy consumption.
        """
        from src.prompts.agent_prompts import VISIT_EXTRACTION_PROMPT

        prompt = VISIT_EXTRACTION_PROMPT.format(
            goal=goal,
            webpage_content=full_text
        )

        try:
            response = self.llm_client.call(prompt, temperature=self.temperature, top_p=self.top_p, stop=None)

            try:
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]

                data = json.loads(response.strip())

                url_text = url if url else "this webpage"
                result = f"""The useful information in {url_text} for user goal "{goal}" as follows:

Evidence in page:
{data.get('evidence', 'N/A')}

Summary:
{data.get('summary', 'N/A')}
"""
                return result

            except json.JSONDecodeError:
                logger.warning("LLM response is not valid JSON, returning raw response")
                return response

        except Exception as e:
            logger.error(f"LLM extraction failed: {str(e)}")
            raise RuntimeError(f"Failed to extract webpage content using LLM: {str(e)}") from e

    def __repr__(self) -> str:
        """Show tool name and extraction mode."""
        return f"VisitTool(name={self.name}, mode=Jina Reader API)"
