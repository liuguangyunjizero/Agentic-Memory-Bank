"""
Configuration Management Module

Loads and manages all system configuration parameters with support for:
- Environment variable loading
- Multiple LLM providers (DeepSeek/OpenAI/Local)
- Cross-platform path handling
"""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Global configuration class for the system."""

    def __init__(self):
        """Initialize configuration from environment variables."""

        self.PROJECT_ROOT = Path(__file__).parent.parent

        # LLM provider configuration
        self.LLM_PROVIDER: Literal["deepseek", "openai", "local"] = os.getenv(
            "LLM_PROVIDER", "deepseek"
        ).lower()

        # DeepSeek configuration
        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
        self.DEEPSEEK_BASE_URL = os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com/v1"
        )
        self.DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        # OpenAI configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_BASE_URL = os.getenv(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1"
        )
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

        # Local model configuration
        self.LOCAL_BASE_URL = os.getenv(
            "LOCAL_BASE_URL",
            "http://127.0.0.1:6001/v1"
        )
        self.LOCAL_MODEL = os.getenv("LOCAL_MODEL", "default")

        # General LLM parameters
        self.LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))
        self.LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        self.LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))

        # Set current LLM config based on provider
        self._set_current_llm_config()

        # Embedding configuration
        self.EMBEDDING_MODEL = os.getenv(
            "EMBEDDING_MODEL",
            "all-MiniLM-L6-v2"
        )

        # Retrieval configuration
        self.RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
        self.RETRIEVAL_ALPHA = float(os.getenv("RETRIEVAL_ALPHA", "0.5"))

        # Merge configuration
        self.MAX_MERGE_DEPTH = int(os.getenv("MAX_MERGE_DEPTH", "3"))
        self.REPORT_CONFLICTS_IN_CONTEXT_LOADING = bool(
            os.getenv("REPORT_CONFLICTS_IN_CONTEXT_LOADING", "True").lower() == "true"
        )

        # Search API configuration
        self.SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
        self.JINA_API_KEY = os.getenv("JINA_API_KEY", "")

        # ReAct configuration
        self.MAX_LLM_CALL_PER_RUN = int(os.getenv("MAX_LLM_CALL_PER_RUN", "60"))
        self.MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "128000"))

        # Path configuration (cross-platform support via pathlib)
        self.TEMP_DIR = self.PROJECT_ROOT / os.getenv("TEMP_DIR", "data/temp")
        self.STORAGE_DIR = self.PROJECT_ROOT / os.getenv("STORAGE_DIR", "data/storage")

        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.STORAGE_DIR.mkdir(parents=True, exist_ok=True)

        # Agent context window configuration
        self.CLASSIFICATION_AGENT_WINDOW = int(
            os.getenv("CLASSIFICATION_AGENT_WINDOW", "32000")
        )
        self.STRUCTURE_AGENT_WINDOW = int(
            os.getenv("STRUCTURE_AGENT_WINDOW", "32000")
        )
        self.ANALYSIS_AGENT_WINDOW = int(
            os.getenv("ANALYSIS_AGENT_WINDOW", "32000")
        )
        self.INTEGRATION_AGENT_WINDOW = int(
            os.getenv("INTEGRATION_AGENT_WINDOW", "32000")
        )
        self.PLANNING_AGENT_WINDOW = int(
            os.getenv("PLANNING_AGENT_WINDOW", "32000")
        )

        # Agent-specific LLM parameters
        self.CLASSIFICATION_AGENT_TEMPERATURE = float(
            os.getenv("CLASSIFICATION_AGENT_TEMPERATURE", "0.4")
        )
        self.CLASSIFICATION_AGENT_TOP_P = float(
            os.getenv("CLASSIFICATION_AGENT_TOP_P", "0.9")
        )

        self.STRUCTURE_AGENT_TEMPERATURE = float(
            os.getenv("STRUCTURE_AGENT_TEMPERATURE", "0.1")
        )
        self.STRUCTURE_AGENT_TOP_P = float(
            os.getenv("STRUCTURE_AGENT_TOP_P", "0.8")
        )

        self.ANALYSIS_AGENT_TEMPERATURE = float(
            os.getenv("ANALYSIS_AGENT_TEMPERATURE", "0.4")
        )
        self.ANALYSIS_AGENT_TOP_P = float(
            os.getenv("ANALYSIS_AGENT_TOP_P", "0.9")
        )

        self.INTEGRATION_AGENT_TEMPERATURE = float(
            os.getenv("INTEGRATION_AGENT_TEMPERATURE", "0.2")
        )
        self.INTEGRATION_AGENT_TOP_P = float(
            os.getenv("INTEGRATION_AGENT_TOP_P", "0.85")
        )

        self.PLANNING_AGENT_TEMPERATURE = float(
            os.getenv("PLANNING_AGENT_TEMPERATURE", "0.6")
        )
        self.PLANNING_AGENT_TOP_P = float(
            os.getenv("PLANNING_AGENT_TOP_P", "0.95")
        )

        self.REACT_AGENT_TEMPERATURE = float(
            os.getenv("REACT_AGENT_TEMPERATURE", "0.6")
        )
        self.REACT_AGENT_TOP_P = float(
            os.getenv("REACT_AGENT_TOP_P", "0.95")
        )

        self.VISIT_EXTRACTION_TEMPERATURE = float(
            os.getenv("VISIT_EXTRACTION_TEMPERATURE", "0.2")
        )
        self.VISIT_EXTRACTION_TOP_P = float(
            os.getenv("VISIT_EXTRACTION_TOP_P", "0.85")
        )

        # Long text processing
        self.CHUNK_RATIO = float(os.getenv("CHUNK_RATIO", "0.9"))

        # Logging configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    def _set_current_llm_config(self):
        """Set current LLM configuration based on LLM_PROVIDER."""
        if self.LLM_PROVIDER == "deepseek":
            self.LLM_API_KEY = self.DEEPSEEK_API_KEY
            self.LLM_BASE_URL = self.DEEPSEEK_BASE_URL
            self.LLM_MODEL = self.DEEPSEEK_MODEL
        elif self.LLM_PROVIDER == "openai":
            self.LLM_API_KEY = self.OPENAI_API_KEY
            self.LLM_BASE_URL = self.OPENAI_BASE_URL
            self.LLM_MODEL = self.OPENAI_MODEL
        elif self.LLM_PROVIDER == "local":
            self.LLM_API_KEY = "EMPTY"
            self.LLM_BASE_URL = self.LOCAL_BASE_URL
            self.LLM_MODEL = self.LOCAL_MODEL
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {self.LLM_PROVIDER}. "
                f"Use: deepseek, openai, or local"
            )

    def get_llm_config(self) -> dict:
        """
        Get current LLM configuration as a dictionary.
        Returns dict containing API key, base URL, model, and other parameters.
        """
        return {
            "provider": self.LLM_PROVIDER,
            "api_key": self.LLM_API_KEY,
            "base_url": self.LLM_BASE_URL,
            "model": self.LLM_MODEL,
            "temperature": self.LLM_TEMPERATURE,
            "max_tokens": self.LLM_MAX_TOKENS,
            "top_p": self.LLM_TOP_P
        }

    def __repr__(self) -> str:
        """Return configuration summary."""
        return (
            f"Config(\n"
            f"  LLM Provider: {self.LLM_PROVIDER}\n"
            f"  LLM Model: {self.LLM_MODEL}\n"
            f"  Embedding Model: {self.EMBEDDING_MODEL}\n"
            f"  Temp Dir: {self.TEMP_DIR}\n"
            f"  Storage Dir: {self.STORAGE_DIR}\n"
            f")"
        )
