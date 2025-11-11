"""
ReAct Agent

多轮对话Agent，支持Think-Act-Observe循环和工具调用。
"""

import logging
import json
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class MultiTurnReactAgent:
    """
    多轮ReAct Agent

    特点：
    - Think-Act-Observe循环
    - 工具调用（search, visit, deep_retrieval）
    - 停止条件：<answer>标签
    - 上下文管理：Token计数 + 超限处理
    """

    def __init__(
        self,
        llm_client,
        tools: Dict[str, Any],
        system_message: str,
        max_iterations: int = 60,
        max_context_tokens: int = 32000,
        temperature: float = 0.6,
        top_p: float = 0.95
    ):
        """
        初始化ReAct Agent

        Args:
            llm_client: LLMClient 实例
            tools: 工具字典 {tool_name: tool_instance}
            system_message: System Prompt
            max_iterations: 最大迭代次数
            max_context_tokens: 最大上下文Token数
            temperature: 温度参数
            top_p: 采样参数
        """
        self.llm_client = llm_client
        self.tools = tools
        self.system_message = system_message
        self.max_iterations = max_iterations
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature
        self.top_p = top_p
        logger.info(
            f"MultiTurnReactAgent initialized successfully: "
            f"tools={list(tools.keys())}, "
            f"max_iterations={max_iterations}, "
            f"temp={temperature}, top_p={top_p}"
        )

    def run(self, question: str) -> Dict[str, Any]:
        """
        执行ReAct循环

        Args:
            question: 用户问题（可能已被Adapter增强过）

        Returns:
            执行结果：{
                "question": str,
                "prediction": str,  # 提取的答案
                "messages": List,   # 完整轨迹
                "termination": str, # 终止原因
                "iterations_used": int  # 使用的迭代次数
            }
        """
        logger.info(f"Starting ReAct loop (task length: {len(question)} characters)")
        # Initial task display handled by main.py

        # 1. 初始化
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": question}
        ]
        full_trajectory = messages.copy()
        iterations_left = self.max_iterations

        # 2. 主循环
        while iterations_left > 0:
            iterations_used = self.max_iterations - iterations_left
            iterations_left -= 1

            # 2.1 调用LLM
            try:
                response = self.llm_client.call(
                    messages,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
            except Exception as e:
                logger.error(f"LLM call failed: {str(e)}")
                return {
                    "question": question,
                    "prediction": "Error: LLM call failed",
                    "messages": full_trajectory,
                    "termination": "error",
                    "iterations_used": iterations_used
                }

            # 2.2 清理意外的tool_response标签
            # Note: stop parameter should prevent this, but DeepSeek API may not fully honor it
            # This is a safety check to ensure clean responses
            if '<tool_response>' in response:
                pos = response.find('<tool_response>')
                logger.warning(
                    f"Cleaned unexpected <tool_response> tag at position {pos} "
                    f"(response length: {len(response)}). "
                    f"Stop parameter may not be working correctly with this API."
                )
                logger.debug(f"Context: ...{response[max(0, pos-50):min(len(response), pos+50)]}...")
                response = response[:pos]

            # 2.3 打印LLM响应（ReAct原始输出）
            # (Response display handled by main.py patch)

            # 2.4 添加到消息历史
            messages.append({"role": "assistant", "content": response.strip()})
            full_trajectory.append({"role": "assistant", "content": response.strip()})

            # 2.5 检查工具调用
            if '<tool_call>' in response and '</tool_call>' in response:
                tool_result = self._handle_tool_call(response)

                # 打印工具响应
                # (Tool response display handled by main.py or logged to file)
                messages.append({"role": "user", "content": tool_result})
                full_trajectory.append({"role": "user", "content": tool_result})

            # 2.6 检查答案
            elif '<answer>' in response and '</answer>' in response:
                answer = self._extract_answer(response)
                if answer:
                    # (Answer obtained - logged)
                    logger.info(f"Answer obtained: {answer[:300]}{'...' if len(answer) > 300 else ''}")
                    return {
                        "question": question,
                        "prediction": answer,
                        "messages": full_trajectory,
                        "termination": "answer",
                        "iterations_used": iterations_used + 1
                    }

            # 2.7 Token计数和上下文管理
            token_count = self.llm_client.count_tokens(str(messages))

            # 2.8 超限处理
            if token_count > self.max_context_tokens:
                logger.warning(f"Token limit exceeded: {token_count} > {self.max_context_tokens}")

                # 强制要求生成答案
                force_answer_msg = (
                    "You have now reached the maximum context length. "
                    "Please provide your final answer immediately using the <answer></answer> format."
                )
                messages.append({"role": "user", "content": force_answer_msg})
                full_trajectory.append({"role": "user", "content": force_answer_msg})

                response = self.llm_client.call(messages)
                messages.append({"role": "assistant", "content": response.strip()})
                full_trajectory.append({"role": "assistant", "content": response.strip()})

                answer = self._extract_answer(response)
                return {
                    "question": question,
                    "prediction": answer if answer else "No answer (token limit)",
                    "messages": full_trajectory,
                    "termination": "token_limit",
                    "iterations_used": iterations_used + 1
                }

        # 3. 超出迭代次数
        logger.warning(f"Reached maximum iterations: {self.max_iterations}")
        answer = self._extract_answer(messages[-1]['content']) if messages else None
        return {
            "question": question,
            "prediction": answer if answer else "No answer found",
            "messages": full_trajectory,
            "termination": "max_iterations",
            "iterations_used": self.max_iterations
        }

    def _handle_tool_call(self, response: str) -> str:
        """
        处理工具调用

        Args:
            response: LLM响应（包含<tool_call>标签）

        Returns:
            工具响应（包含<tool_response>标签）
        """
        try:
            # 1. 提取JSON
            tool_call_str = response.split('<tool_call>')[1].split('</tool_call>')[0]
            tool_call = json.loads(tool_call_str.strip())

            # 2. 执行工具
            tool_name = tool_call.get('name', '')
            tool_args = tool_call.get('arguments', {})

            logger.info(f"Executing tool: {tool_name}, arguments: {tool_args}")

            if tool_name in self.tools:
                result = self.tools[tool_name].call(tool_args)
            else:
                result = f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
                logger.error(result)

            return f"<tool_response>{result}</tool_response>"

        except json.JSONDecodeError as e:
            error_msg = f"Error: Invalid JSON in tool call - {str(e)}"
            logger.error(error_msg)
            return f"<tool_response>{error_msg}</tool_response>"
        except Exception as e:
            error_msg = f"Error: Tool call failed - {str(e)}"
            logger.error(error_msg)
            return f"<tool_response>{error_msg}</tool_response>"

    def _extract_answer(self, response: str) -> Optional[str]:
        """
        提取答案

        Args:
            response: LLM响应

        Returns:
            提取的答案，如果没有则返回 None
        """
        try:
            answer = response.split('<answer>')[1].split('</answer>')[0].strip()
            return answer
        except (IndexError, AttributeError):
            return None

    def __repr__(self) -> str:
        """返回Agent摘要"""
        return (
            f"MultiTurnReactAgent("
            f"tools={list(self.tools.keys())}, "
            f"max_iterations={self.max_iterations})"
        )
