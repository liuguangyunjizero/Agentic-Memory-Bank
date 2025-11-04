"""
ReAct Agent

多轮对话Agent，支持Think-Act-Observe循环和工具调用。

参考：WebResummer的MultiTurnReactAgent实现
规范文档：第7.1节
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
        max_context_tokens: int = 32000
    ):
        """
        初始化ReAct Agent

        Args:
            llm_client: LLMClient 实例
            tools: 工具字典 {tool_name: tool_instance}
            system_message: System Prompt
            max_iterations: 最大迭代次数
            max_context_tokens: 最大上下文Token数
        """
        self.llm_client = llm_client
        self.tools = tools
        self.system_message = system_message
        self.max_iterations = max_iterations
        self.max_context_tokens = max_context_tokens
        logger.info(
            f"MultiTurnReactAgent初始化完成: "
            f"tools={list(tools.keys())}, "
            f"max_iterations={max_iterations}"
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
        logger.info(f"开始ReAct循环: {question[:100]}...")
        print(f"\n{'='*80}")
        print(f"🤖 ReAct Agent 开始执行")
        print(f"{'='*80}")
        print(f"任务: {question[:200]}{'...' if len(question) > 200 else ''}")
        print(f"{'='*80}\n")

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

            logger.debug(f"迭代 {iterations_used + 1}/{self.max_iterations}")

            # 2.1 调用LLM
            try:
                response = self.llm_client.call(messages)
            except Exception as e:
                logger.error(f"LLM调用失败: {str(e)}")
                return {
                    "question": question,
                    "prediction": "Error: LLM call failed",
                    "messages": full_trajectory,
                    "termination": "error",
                    "iterations_used": iterations_used
                }

            # 2.2 清理意外的tool_response标签
            if '<tool_response>' in response:
                pos = response.find('<tool_response>')
                response = response[:pos]
                logger.warning("清理了意外的tool_response标签")

            # 2.3 打印LLM响应（ReAct原始输出）
            print(f"\n{'='*80}")
            print(f"📤 ReAct Agent 响应 (迭代 {iterations_used + 1}):")
            print(f"{'='*80}")
            print(response.strip())
            print(f"{'='*80}\n")

            # 2.4 添加到消息历史
            messages.append({"role": "assistant", "content": response.strip()})
            full_trajectory.append({"role": "assistant", "content": response.strip()})

            # 2.5 检查工具调用
            if '<tool_call>' in response and '</tool_call>' in response:
                logger.debug("检测到工具调用")
                tool_result = self._handle_tool_call(response)

                # 打印工具响应
                print(f"\n{'='*80}")
                print(f"🔧 工具响应:")
                print(f"{'='*80}")
                # 截断过长的工具响应（只显示前2000个字符）
                display_result = tool_result[:2000] + "...\n[响应过长，已截断]" if len(tool_result) > 2000 else tool_result
                print(display_result)
                print(f"{'='*80}\n")

                messages.append({"role": "user", "content": tool_result})
                full_trajectory.append({"role": "user", "content": tool_result})

            # 2.6 检查答案
            elif '<answer>' in response and '</answer>' in response:
                answer = self._extract_answer(response)
                if answer:
                    print(f"\n{'='*80}")
                    print(f"✅ ReAct Agent 完成 - 获得最终答案")
                    print(f"{'='*80}\n")
                    logger.info(f"获得答案: {answer[:100]}...")
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
                logger.warning(f"Token超限: {token_count} > {self.max_context_tokens}")

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
        logger.warning(f"达到最大迭代次数: {self.max_iterations}")
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

            logger.info(f"执行工具: {tool_name}, 参数: {tool_args}")

            if tool_name in self.tools:
                result = self.tools[tool_name].call(tool_args)
                logger.debug(f"工具结果长度: {len(result)}")
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
