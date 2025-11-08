"""
LLM 客户端模块

统一的 LLM 调用接口，支持：
- DeepSeek API
- OpenAI API
- 本地模型
- 自动重试机制
- Token 计数估算

参考：WebResummer 的 call_server 方法
"""

import time
import logging
from typing import List, Dict, Optional, Union
from openai import OpenAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM 客户端（参考 WebResummer）

    支持多个 LLM 提供商的统一调用接口
    """

    def __init__(
        self,
        provider: str = "deepseek",
        api_key: str = "EMPTY",
        base_url: str = "http://127.0.0.1:6001/v1",
        model: str = "default",
        temperature: float = 0.6,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        max_retries: int = 10,
        timeout: int = 300
    ):
        """
        初始化 LLM 客户端

        Args:
            provider: LLM 提供商 (deepseek/openai/local)
            api_key: API 密钥
            base_url: API 端点 URL
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            top_p: 采样参数
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
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

        # 创建 OpenAI 客户端（兼容所有提供商）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        logger.info(
            f"LLM 客户端初始化完成: provider={provider}, "
            f"model={model}, base_url={base_url}"
        )

    @classmethod
    def from_config(cls, config) -> "LLMClient":
        """
        从 Config 对象创建 LLM 客户端

        Args:
            config: Config 实例

        Returns:
            LLMClient 实例
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
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        调用 LLM 并返回响应（带重试机制）

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}] 或单个字符串
            temperature: 温度参数（可选，覆盖默认值）
            top_p: 采样参数（可选）
            stop: 停止词列表（可选）
            max_tokens: 最大 token 数（可选）

        Returns:
            LLM 响应内容

        参考：WebResummer 的重试机制
        """
        # 如果 messages 是字符串，转换为消息列表
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # 使用传入的参数或默认值
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        stop = stop or ["\n<tool_response", "<tool_response>"]

        # 重试循环
        for attempt in range(self.max_retries):
            try:
                msg_count = len(messages)
                total_chars = sum(len(msg.get("content", "")) for msg in messages if isinstance(msg, dict))
                logger.debug(
                    f"LLM 调用 (尝试 {attempt + 1}/{self.max_retries}): "
                    f"消息数={msg_count}, 总字符数={total_chars}, "
                    f"temp={temperature}, top_p={top_p}"
                )

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
                    logger.debug(f"LLM 响应成功: 长度={len(content)} 字符")
                    return content
                else:
                    logger.warning(f"LLM 返回空响应 (尝试 {attempt + 1})")

            except Exception as e:
                logger.error(
                    f"LLM 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}"
                )

                if attempt == self.max_retries - 1:
                    logger.error(f"达到最大重试次数，调用失败")
                    return f"Error: Failed after {self.max_retries} attempts - {str(e)}"

                # 指数退避
                sleep_time = min(2 ** attempt, 30)
                logger.info(f"等待 {sleep_time} 秒后重试...")
                time.sleep(sleep_time)

        return "Error: No response after retries"

    def count_tokens(self, messages: Union[List[Dict[str, str]], str]) -> int:
        """
        估算 token 数量（简单实现）

        Args:
            messages: 消息列表或字符串

        Returns:
            估算的 token 数

        注意：这是一个粗略估算，实际 token 数可能有差异
              对于精确计数，可以使用 tiktoken 库
        """
        # 如果是字符串，直接计算
        if isinstance(messages, str):
            # 简单估算：中文约1个字符1个token，英文约4个字符1个token
            # 这里使用平均值：每3个字符约1个token
            return len(messages) // 3

        # 如果是消息列表，累加所有内容
        total_chars = sum(
            len(msg.get("content", ""))
            for msg in messages
            if isinstance(msg, dict)
        )
        return total_chars // 3

    def test_connection(self) -> bool:
        """
        测试 LLM 连接是否正常

        Returns:
            连接成功返回 True，失败返回 False
        """
        try:
            logger.info("测试 LLM 连接...")
            response = self.call([{"role": "user", "content": "你好"}])

            if "Error" not in response:
                logger.info("LLM 连接测试成功")
                return True
            else:
                logger.error(f"LLM 连接测试失败: {response}")
                return False

        except Exception as e:
            logger.error(f"LLM 连接测试失败: {str(e)}")
            return False

    def __repr__(self) -> str:
        """返回客户端摘要"""
        return (
            f"LLMClient(provider={self.provider}, "
            f"model={self.model}, "
            f"base_url={self.base_url})"
        )
