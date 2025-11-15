"""
Unified client for interacting with multiple LLM providers through OpenAI-compatible APIs.
Handles retries, token counting, and parameter management across different backends.
"""

import time
import logging
from typing import List, Dict, Optional, Union
from openai import OpenAI

_UNSET = object()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMClient:
    """
    Abstraction layer over OpenAI SDK that works with DeepSeek, OpenAI, and local models.
    Provides automatic retry with exponential backoff and flexible parameter overriding.
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
        Configure connection parameters and create the underlying OpenAI client.
        All providers must expose OpenAI-compatible chat completion endpoints.
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
        Factory method that extracts connection details from configuration object.
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
        stop=_UNSET,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Execute chat completion request with automatic retry and exponential backoff.
        Uses sentinel pattern for stop parameter to distinguish between omitted and explicit None.
        Returns error string on complete failure rather than raising exception.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        if stop is _UNSET:
            stop = ["\n<tool_response", "<tool_response>"]

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

                sleep_time = min(2 ** attempt, 30)
                logger.info(f"Waiting {sleep_time} seconds before retry...")
                time.sleep(sleep_time)

        return "Error: No response after retries"

    def count_tokens(self, messages: Union[List[Dict[str, str]], str]) -> int:
        """
        Calculate approximate token count for input text or message history.
        Uses tiktoken for precision when available, falls back to character-based estimation.
        Accounts for message formatting overhead in chat completion calls.
        """
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")

            if isinstance(messages, str):
                return len(encoding.encode(messages))

            total_tokens = 0
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    total_tokens += len(encoding.encode(role))
                    total_tokens += len(encoding.encode(content))
                    total_tokens += 4

            return total_tokens

        except ImportError:
            logger.debug("tiktoken not available, using simple estimation")

            if isinstance(messages, str):
                return len(messages) // 3

            total_chars = sum(
                len(msg.get("content", ""))
                for msg in messages
                if isinstance(msg, dict)
            )
            return total_chars // 3
