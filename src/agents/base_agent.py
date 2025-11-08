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

    def _fix_json_string(self, json_str: str) -> str:
        """
        尝试修复常见的 JSON 字符串错误

        Args:
            json_str: 可能有错误的 JSON 字符串

        Returns:
            修复后的 JSON 字符串
        """
        # 1. 移除 JSON 前后的多余空白
        json_str = json_str.strip()

        # 2. 检查并补全缺失的结束括号
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
            logger.debug(f"补全了 {open_braces - close_braces} 个结束大括号")

        # 3. 检查并补全缺失的结束方括号
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
            logger.debug(f"补全了 {open_brackets - close_brackets} 个结束方括号")

        # 4. 尝试修复末尾的逗号（JSON 不允许末尾逗号）
        import re
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        return json_str

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
        except json.JSONDecodeError as e:
            logger.debug(f"直接JSON解析失败: {str(e)}, 尝试提取代码块...")

            # 尝试提取 ```json 代码块
            if "```json" in response:
                try:
                    parts = response.split("```json", 1)
                    if len(parts) > 1:
                        json_part = parts[1].split("```", 1)
                        if len(json_part) > 0:
                            json_str = json_part[0].strip()
                            # ✅ 尝试修复 JSON
                            json_str = self._fix_json_string(json_str)
                            return json.loads(json_str)
                except (IndexError, json.JSONDecodeError) as e:
                    logger.debug(f"提取 ```json 块失败: {str(e)}, 尝试其他方法...")
                    pass

            # 尝试提取 ``` 代码块（不带 json 标记）
            if "```" in response and "```json" not in response:
                try:
                    parts = response.split("```", 2)
                    if len(parts) >= 3:
                        json_str = parts[1].strip()
                        # ✅ 尝试修复 JSON
                        json_str = self._fix_json_string(json_str)
                        return json.loads(json_str)
                except (IndexError, json.JSONDecodeError) as e:
                    logger.debug(f"提取 ``` 块失败: {str(e)}, 尝试其他方法...")
                    pass

            # 尝试提取 JSON 对象
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                try:
                    json_str = response[start:end]
                    # ✅ 尝试修复 JSON
                    json_str = self._fix_json_string(json_str)
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.debug(f"提取 JSON 对象失败: {str(e)}, 尝试其他方法...")
                    pass

            # 尝试提取数组
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                try:
                    json_str = response[start:end]
                    # ✅ 尝试修复 JSON
                    json_str = self._fix_json_string(json_str)
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.debug(f"提取 JSON 数组失败: {str(e)}")
                    pass

            # ✅ 改进：显示更详细的错误信息和响应的首尾部分
            logger.error(f"JSON 解析失败，响应长度: {len(response)} 字符")
            logger.error(f"响应前500字符: {response[:500]}")
            logger.error(f"响应后500字符: {response[-500:]}")

            raise ValueError(f"无法解析 JSON 响应")

