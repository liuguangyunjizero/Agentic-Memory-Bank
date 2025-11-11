"""
配置管理模块

负责加载和管理所有系统配置参数，支持：
- 环境变量加载
- 多 LLM 提供商配置（DeepSeek/OpenAI/本地模型）
- 跨平台路径处理
"""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class Config:
    """全局配置类"""

    def __init__(self):
        """初始化配置，从环境变量读取所有参数"""

        # ===== 项目根目录 =====
        self.PROJECT_ROOT = Path(__file__).parent.parent

        # ===== LLM 提供商配置 =====
        self.LLM_PROVIDER: Literal["deepseek", "openai", "local"] = os.getenv(
            "LLM_PROVIDER", "deepseek"
        ).lower()

        # DeepSeek 配置
        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
        self.DEEPSEEK_BASE_URL = os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com/v1"
        )
        self.DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        # OpenAI 配置
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_BASE_URL = os.getenv(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1"
        )
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

        # 本地模型配置
        self.LOCAL_BASE_URL = os.getenv(
            "LOCAL_BASE_URL",
            "http://127.0.0.1:6001/v1"
        )
        self.LOCAL_MODEL = os.getenv("LOCAL_MODEL", "default")

        # LLM 通用参数
        self.LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))
        self.LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        self.LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))

        # ===== 当前 LLM 配置（根据 provider 自动选择）=====
        self._set_current_llm_config()

        # ===== Embedding 配置 =====
        self.EMBEDDING_MODEL = os.getenv(
            "EMBEDDING_MODEL",
            "all-MiniLM-L6-v2"
        )

        # ===== 检索配置 =====
        self.RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
        self.RETRIEVAL_ALPHA = float(os.getenv("RETRIEVAL_ALPHA", "0.5"))

        # ===== 搜索API配置 =====
        self.SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
        self.JINA_API_KEY = os.getenv("JINA_API_KEY", "")

        # ===== ReAct 配置 =====
        self.MAX_LLM_CALL_PER_RUN = int(os.getenv("MAX_LLM_CALL_PER_RUN", "60"))
        self.MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "32000"))

        # ===== 路径配置（使用 pathlib 确保跨平台）=====
        self.TEMP_DIR = self.PROJECT_ROOT / os.getenv("TEMP_DIR", "data/temp")
        self.STORAGE_DIR = self.PROJECT_ROOT / os.getenv("STORAGE_DIR", "data/storage")

        # 确保目录存在
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.STORAGE_DIR.mkdir(parents=True, exist_ok=True)

        # ===== Agent 窗口配置 =====
        self.CLASSIFICATION_AGENT_WINDOW = int(
            os.getenv("CLASSIFICATION_AGENT_WINDOW", "8000")
        )
        self.STRUCTURE_AGENT_WINDOW = int(
            os.getenv("STRUCTURE_AGENT_WINDOW", "8000")
        )
        self.ANALYSIS_AGENT_WINDOW = int(
            os.getenv("ANALYSIS_AGENT_WINDOW", "8000")
        )
        self.INTEGRATION_AGENT_WINDOW = int(
            os.getenv("INTEGRATION_AGENT_WINDOW", "8000")
        )
        self.PLANNING_AGENT_WINDOW = int(
            os.getenv("PLANNING_AGENT_WINDOW", "8000")
        )

        # ===== Agent LLM 参数配置 =====
        # Classification Agent
        self.CLASSIFICATION_AGENT_TEMPERATURE = float(
            os.getenv("CLASSIFICATION_AGENT_TEMPERATURE", "0.4")
        )
        self.CLASSIFICATION_AGENT_TOP_P = float(
            os.getenv("CLASSIFICATION_AGENT_TOP_P", "0.9")
        )

        # Structure Agent
        self.STRUCTURE_AGENT_TEMPERATURE = float(
            os.getenv("STRUCTURE_AGENT_TEMPERATURE", "0.1")
        )
        self.STRUCTURE_AGENT_TOP_P = float(
            os.getenv("STRUCTURE_AGENT_TOP_P", "0.8")
        )

        # Analysis Agent
        self.ANALYSIS_AGENT_TEMPERATURE = float(
            os.getenv("ANALYSIS_AGENT_TEMPERATURE", "0.4")
        )
        self.ANALYSIS_AGENT_TOP_P = float(
            os.getenv("ANALYSIS_AGENT_TOP_P", "0.9")
        )

        # Integration Agent
        self.INTEGRATION_AGENT_TEMPERATURE = float(
            os.getenv("INTEGRATION_AGENT_TEMPERATURE", "0.2")
        )
        self.INTEGRATION_AGENT_TOP_P = float(
            os.getenv("INTEGRATION_AGENT_TOP_P", "0.85")
        )

        # Planning Agent
        self.PLANNING_AGENT_TEMPERATURE = float(
            os.getenv("PLANNING_AGENT_TEMPERATURE", "0.6")
        )
        self.PLANNING_AGENT_TOP_P = float(
            os.getenv("PLANNING_AGENT_TOP_P", "0.95")
        )

        # ReAct Agent
        self.REACT_AGENT_TEMPERATURE = float(
            os.getenv("REACT_AGENT_TEMPERATURE", "0.6")
        )
        self.REACT_AGENT_TOP_P = float(
            os.getenv("REACT_AGENT_TOP_P", "0.95")
        )

        # Visit Tool Extraction (LLM-based content extraction)
        self.VISIT_EXTRACTION_TEMPERATURE = float(
            os.getenv("VISIT_EXTRACTION_TEMPERATURE", "0.2")
        )
        self.VISIT_EXTRACTION_TOP_P = float(
            os.getenv("VISIT_EXTRACTION_TOP_P", "0.85")
        )

        # ===== 超长文本处理 =====
        self.CHUNK_RATIO = float(os.getenv("CHUNK_RATIO", "0.9"))

        # ===== 日志配置 =====
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    def _set_current_llm_config(self):
        """根据 LLM_PROVIDER 设置当前使用的 LLM 配置"""
        if self.LLM_PROVIDER == "deepseek":
            self.LLM_API_KEY = self.DEEPSEEK_API_KEY
            self.LLM_BASE_URL = self.DEEPSEEK_BASE_URL
            self.LLM_MODEL = self.DEEPSEEK_MODEL
        elif self.LLM_PROVIDER == "openai":
            self.LLM_API_KEY = self.OPENAI_API_KEY
            self.LLM_BASE_URL = self.OPENAI_BASE_URL
            self.LLM_MODEL = self.OPENAI_MODEL
        elif self.LLM_PROVIDER == "local":
            self.LLM_API_KEY = "EMPTY"  # 本地模型通常不需要真实 key
            self.LLM_BASE_URL = self.LOCAL_BASE_URL
            self.LLM_MODEL = self.LOCAL_MODEL
        else:
            raise ValueError(
                f"不支持的 LLM_PROVIDER: {self.LLM_PROVIDER}. "
                f"请使用: deepseek, openai, local"
            )

    def get_llm_config(self) -> dict:
        """
        获取当前 LLM 配置字典

        Returns:
            包含 API key, base_url, model 等配置的字典
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
        """返回配置摘要"""
        return (
            f"Config(\n"
            f"  LLM Provider: {self.LLM_PROVIDER}\n"
            f"  LLM Model: {self.LLM_MODEL}\n"
            f"  Embedding Model: {self.EMBEDDING_MODEL}\n"
            f"  Temp Dir: {self.TEMP_DIR}\n"
            f"  Storage Dir: {self.STORAGE_DIR}\n"
            f")"
        )


# 全局配置实例（可选）
# config = Config()
