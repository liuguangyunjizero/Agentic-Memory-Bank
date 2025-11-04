"""
Agent 基础类

所有 LLM 驱动的 Agent 的基类
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BaseAgent:
    """Agent 基类"""

    def __init__(self, llm_client):
        """
        初始化 Agent

        Args:
            llm_client: LLMClient 实例
        """
        from src.utils.llm_client import LLMClient

        # 类型检查（支持duck typing用于测试）
        if not (isinstance(llm_client, LLMClient) or hasattr(llm_client, 'call')):
            raise TypeError("llm_client 必须是 LLMClient 实例或具有 call 方法的对象")

        self.llm_client = llm_client

    def _call_llm(self, prompt: str) -> str:
        """
        调用 LLM

        Args:
            prompt: 输入 prompt

        Returns:
            LLM 响应
        """
        try:
            response = self.llm_client.call(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM 调用失败: {str(e)}")
            raise

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        解析 JSON 响应

        Args:
            response: LLM 返回的字符串

        Returns:
            解析后的字典
        """
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取 ```json 代码块
            if "```json" in response:
                try:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_str)
                except (IndexError, json.JSONDecodeError):
                    pass

            # 尝试提取 ``` 代码块（不带 json 标记）
            if "```" in response:
                try:
                    json_str = response.split("```")[1].split("```")[0].strip()
                    return json.loads(json_str)
                except (IndexError, json.JSONDecodeError):
                    pass

            # 尝试提取 JSON 对象
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                try:
                    return json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass

            # 尝试提取数组
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                try:
                    return json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass

            logger.error(f"JSON 解析失败，原始响应: {response[:500]}...")
            raise ValueError(f"无法解析 JSON 响应")
