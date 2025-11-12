"""
LLM Client Module

Unified LLM calling interface supporting:
- DeepSeek API
- OpenAI API
- Local models
- Automatic retry mechanism
- Token counting estimation
"""

import time
import logging
from typing import List, Dict, Optional, Union
from openai import OpenAI

# Sentinel value to distinguish "not passed" from "explicitly passed None"
_UNSET = object()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM Client (reference: WebResummer)

    Unified calling interface supporting multiple LLM providers
    """

    def __init__(
        self,
        provider: str = "deepseek",
        api_key: str = "",
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        temperature: float = 0.6,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        max_retries: int = 10,
        timeout: int = 300
    ):
        """
        Initialize LLM Client

        Args:
            provider: LLM provider (deepseek/openai/local)
            api_key: API key
            base_url: API endpoint URL
            model: Model name
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            top_p: Sampling parameter
            max_retries: Maximum retry attempts
            timeout: Request timeout (seconds)
        """
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        self.timeout = timeout

        # Create OpenAI client (compatible with all providers)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        logger.info(
            f"LLM client initialized successfully: provider={provider}, "
            f"model={model}, base_url={base_url}"
        )

    @classmethod
    def from_config(cls, config) -> "LLMClient":
        """
        Create LLM client from Config object

        Args:
            config: Config instance

        Returns:
            LLMClient instance
        """
        return cls(
            provider=config.LLM_PROVIDER,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            top_p=config.LLM_TOP_P
        )

    def call(
        self,
        messages: Union[List[Dict[str, str]], str],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop=_UNSET,  # Use sentinel to distinguish "not passed" from "passed None"
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Call LLM and return response (with retry mechanism)

        Args:
            messages: Message list [{"role": "user", "content": "..."}] or single string
            temperature: Temperature parameter (optional, overrides default)
            top_p: Sampling parameter (optional)
            stop: Stop sequence list (optional)
                - Not passed: use default stop sequences ["\n<tool_response", "<tool_response>"]
                - Pass None: don't use any stop sequences
                - Pass list: use specified stop sequences
            max_tokens: Maximum tokens (optional)

        Returns:
            LLM response content

        Reference: WebResummer's retry mechanism
        """
        # If messages is a string, convert to message list
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Use passed parameters or defaults
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        # Fixed: Use sentinel to properly handle stop parameter
        # - If not passed (default _UNSET): use default stop sequences for ReAct Agent
        # - If explicitly passed None: don't use any stop sequences (for other Agents)
        # - If passed a list: use that list
        if stop is _UNSET:
            stop = ["\n<tool_response", "<tool_response>"]
        # else: use whatever was passed (None or a list)

        # Retry loop
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    max_tokens=max_tokens,
                    timeout=self.timeout
                )

                content = response.choices[0].message.content

                if content:
                    return content
                else:
                    logger.warning(f"LLM returned empty response (attempt {attempt + 1})")

            except Exception as e:
                logger.error(
                    f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}"
                )

                if attempt == self.max_retries - 1:
                    logger.error(f"Maximum retry attempts reached, call failed")
                    return f"Error: Failed after {self.max_retries} attempts - {str(e)}"

                # Exponential backoff
                sleep_time = min(2 ** attempt, 30)
                logger.info(f"Waiting {sleep_time} seconds before retry...")
                time.sleep(sleep_time)

        return "Error: No response after retries"

    def count_tokens(self, messages: Union[List[Dict[str, str]], str]) -> int:
        """
        Estimate token count

        Args:
            messages: Message list or string

        Returns:
            Estimated token count

        Note:
            - Prioritizes using tiktoken library for precise counting
            - If tiktoken is unavailable, uses simple estimate: ~1 token per 3 characters
        """
        try:
            import tiktoken

            # Use cl100k_base encoder (used by GPT-3.5/GPT-4)
            # Generally applicable to DeepSeek as well
            encoding = tiktoken.get_encoding("cl100k_base")

            # If string, calculate directly
            if isinstance(messages, str):
                return len(encoding.encode(messages))

            # If message list, sum all content
            total_tokens = 0
            for msg in messages:
                if isinstance(msg, dict):
                    # Calculate tokens for role and content
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    total_tokens += len(encoding.encode(role))
                    total_tokens += len(encoding.encode(content))
                    # Add 3-4 tokens per message (format overhead)
                    total_tokens += 4

            return total_tokens

        except ImportError:
            # tiktoken unavailable, use simple estimation
            logger.debug("tiktoken not available, using simple estimation")

            # If string, calculate directly
            if isinstance(messages, str):
                # Simple estimate: ~1 character = 1 token for Chinese, ~4 characters = 1 token for English
                # Using average: ~3 characters = 1 token
                return len(messages) // 3

            # If message list, sum all content
            total_chars = sum(
                len(msg.get("content", ""))
                for msg in messages
                if isinstance(msg, dict)
            )
            return total_chars // 3
